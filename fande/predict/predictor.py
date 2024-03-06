"""
Predictors objects aim to handle the prediction of energies and forces, evaluate test metrics, make plots, etc.
As an input object it takes the models forces and energies and the data module.
"""


from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import numpy as np

from fande.utils import get_vectors_e, get_vectors_f

import torch
import torchmetrics

from ase.units import Bohr, Hartree

from xtb.ase.calculator import XTB


from ase.visualize import view

from fande import logger

import gpytorch

from ase import Atoms

import warnings
import time

class FandePredictor:
    def __init__(
        self,
        fdm,
        atomic_group_force_model,
        energy_model,
        hparams,
        soap_params
    ):

        self.fdm = fdm
        self.hparams = hparams
        self.soap_params = soap_params


        self.energy_model = energy_model
        self.energy_model_device = torch.device('cuda:0')

        self.ag_force_model = atomic_group_force_model
        self.ag_force_model_device = torch.device('cuda:0')
        self.ag_force_model_devices = [0]


        # self.test_X = fdm.test_X
        # self.test_DX = fdm.test_DX
        # self.test_E = fdm.test_E
        # self.test_F = fdm.test_F



        self.n_atoms = 1

        self.batch_size = 1000_000

        self.last_calculated_snapshot = self.fdm.traj_train[0]
        self.last_X = None
        self.last_DX_grouped = None

 
    def predict_forces_single_snapshot_r(self, snapshot, atomic_groups=None):

            # print(np.any(self.last_calculated_snapshot.positions != snapshot.positions))
            n_atoms = len(snapshot) 
            snapshot.wrap(eps=1e-8)
            # print( np.allclose(self.last_calculated_snapshot.positions, snapshot.positions))
            time_invariants = 0
            time_start = 0

            if not np.allclose(self.last_calculated_snapshot.positions, snapshot.positions) or True:
                # print("Start invariants forces...")
                # record start time
                time_start = time.time()
                X, DX_grouped = self.fdm.calculate_snapshot_invariants_librascal(snapshot)
                # record end time
                time_invariants = time.time()
                self.last_calculated_snapshot = snapshot
                self.last_X, self.last_DX_grouped = X, DX_grouped

                # print("End invariants forces...")
            else:
                X, DX_grouped = self.last_X, self.last_DX_grouped

            # print(snap_DX)
            # predictions_grouped = []
            forces = np.zeros((n_atoms, 3))
            forces_variance = np.zeros((n_atoms, 3))

            time_start_prediction = time.time()
            
            print("Time for invariants (call from forces): ", (time_invariants - time_start) * 1000)

            if self.ag_force_model is None:
                # warnings.warn("Atomic group force model is not defined. Cannot predict forces. Returning zeros.")
                # print("Atomic group force model is not defined. Cannot predict forces. Returning zeros.")
                return forces, forces_variance

            print("Predicting forces...")

            for idx, model in enumerate(self.ag_force_model.models):              

                # res = trainer_f.predict(model, test_dl)[0]

                if model.device != self.ag_force_model_device:
                    # this is related to the issue that gpytorch does not move the cache to the device properly
                    print("removing cache for forces model (THIS IS EXPENSIVE) ")
                    model.train()
                    model.eval()
                    print("Cache moved to the proper device.")
        
                print("Trying to predict forces with ...", DX_grouped)

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    res = model(DX_grouped[idx].cpu())

                # predictions_torch = res.mean
                print("Predicted forces...", res)

                predictions = res.mean.cpu().detach().numpy()

                predictions_variance = res.variance.cpu().detach().numpy()
                         
                n_atoms_in_group = len(atomic_groups[idx])
                pred_forces = predictions.reshape((n_atoms_in_group, 3))
                pred_forces_variance = predictions_variance.reshape((n_atoms_in_group, 3))
                # predictions_grouped.append(pred_forces)

                forces[sorted(atomic_groups[idx])] = pred_forces
                forces_variance[sorted(atomic_groups[idx])] = pred_forces_variance
            

            return forces, forces_variance


    def predict_energy_single_snapshot_r(self, snapshot):
            """
            Predicts energy and energy variance for a single snapshot.
            
            Parameters
            ----------
            snapshot : ase.Atoms
                Snapshot to predict energy for.
            Returns
            -------
            float
                Predicted energy.
            float
                Predicted energy variance.
            """ 

            # print(np.any(self.last_calculated_snapshot.positions != snapshot.positions))
            n_atoms = len(snapshot) 
            snapshot.wrap(eps=1e-8)
            # print( np.allclose(self.last_calculated_snapshot.positions, snapshot.positions))

            # record start time
            time_start = time.time()

            if not np.allclose(self.last_calculated_snapshot.positions, snapshot.positions):
                # print("Start invariants energy...")
                X, DX_grouped = self.fdm.calculate_snapshot_invariants_librascal(snapshot)
                self.last_calculated_snapshot = snapshot.copy()
                self.last_X, self.last_DX_grouped = X, DX_grouped
                # print("End invariants energy...")
            else:
                X, DX_grouped = self.last_X, self.last_DX_grouped

            time_invariants = time.time()

            energy = 0.0
            energy_variance = 0.0
            model = self.energy_model

            if model is None:
                # warnings.warn("Atomic group force model is not defined. Cannot predict forces. Returning zeros.")
                # print("Energy model is not defined. Cannot predict energy. Returning zeros.")
                return energy, energy_variance

            # Bug from PyTorch, it's necessary to clean cache with .train() call when moving to another device: 
            # https://github.com/cornellius-gp/gpytorch/issues/1619


            # print("Energy model device: ", self.energy_model.device)
            model.to(self.energy_model_device)
            x_sum = X.sum(axis=-2).to(self.energy_model_device)
            
            if model.device != self.energy_model_device:
                # this is related to the issue that gpytorch does not move the cache to the device properly
                print("removing cache for energy model (THIS IS EXPENSIVE) ")
                model.train()
                model.eval()
                print("Cache moved to the proper device.")

            time_start_prediction = time.time()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                res = model(x_sum) 

            time_end_prediction = time.time()

            predicted_mean = res.mean.cpu().detach().numpy()
            predicted_variance = res.variance.cpu().detach().numpy()

            energy = predicted_mean
            energy_variance = predicted_variance

            time_end = time.time()

            print("Energy model summary: ")
            print("Time invariants: ", (time_invariants - time_start) * 1000)
            print("Time prediction: ", (time_end_prediction - time_start_prediction)*1000)
            print("Time moving on device: ", (time_end - time_end_prediction)*1000)
            print("Time total: ", (time_end - time_start)*1000)
        
            return energy, energy_variance


    def test_errors(self, view_worst_atoms=False):
        """
        Provides test errors metrics for the models. SHOULD BE REMOVED FROM PREDICTOR AND MOVED TO A SEPARATE CLASS.

        Parameters
        ----------
        plot : bool, optional
            If True, plots the errors, by default False
        view_worst_atoms : bool, optional
            If True, provides the indices of the worst atoms, by default False

        Returns
        -------
        dict
            Dictionary containing the errors metrics 

        """

        logger.info("Calculating errors...")

        for i in range(len(self.ag_force_model.train_data_loaders)):
            plt.hist( self.ag_force_model.train_data_loaders[i].dataset[:][1].cpu(), label=str(i), alpha=0.7 )    
            plt.legend()
            plt.savefig("SELECTED_TRAINING_FORCES_" + str(i) + ".png")
            plt.close()
        
        # try to make it work with the self.ag_force_model
        predictions_errors = []

        per_model_RMSE = []
        per_model_MAE = []

        for idx, model in enumerate(self.ag_force_model.models):       

            model = model.cuda()
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                res = model(self.fdm.test_DX[idx].cuda())

            predictions_torch = res.mean

            # print(predictions_torch)     
            # print()

            predictions_errors_idx = predictions_torch.detach().cpu() - self.fdm.test_F[idx].detach().cpu()
            predictions_errors_idx = predictions_errors_idx.numpy()
            predictions_errors.append(predictions_errors_idx)

            rmse = np.sqrt( np.mean(  predictions_errors_idx**2) )
            mae = np.mean(  abs(predictions_errors_idx))

            per_model_RMSE.append(rmse)
            per_model_MAE.append(mae)

            plt.figure(figsize=(15, 6), dpi=80)
            plt.title(f"RMSE:  {rmse}, MAE:  {mae}, Total length:{predictions_torch.shape[0]}" )
            plt.plot(predictions_torch.detach().cpu(), label="GP predictions group " + str(idx))
            plt.plot(self.fdm.test_F[idx].detach().cpu(), label="true force group " + str(idx))
            plt.legend()
            # plt.xlim([0,100])
            plt.savefig("PRED_vs_TRUE_val_group_" + str(idx) + ".png")
            plt.close()


        for idx, pred_err in enumerate(predictions_errors):
            plt.plot( pred_err, label="Atomic group " + str(idx) )
            plt.legend()
            plt.savefig("predictions_errors_group_" + str(idx) + ".png")
            plt.close()

            plt.hist(pred_err, bins=30, label="Atomic group " + str(idx))
            plt.legend()
            plt.savefig("hist_predictions_errors_group_" + str(idx) + ".png")
            plt.close()

            # plt.plot( self.fdm.test_F[idx].detach().cpu(), label="Test forces " + str(idx) )
            # plt.plot( predictions_torch.detach().cpu(), label="Predicted forces " + str(idx) )
            # plt.legend()
            # plt.savefig("TEST_group_" + str(idx) + ".png")
            # plt.close()

            print("Error metrics for atomic group ", idx)
            print("RMSE: ", np.sqrt( np.mean(pred_err**2) ) )
            print("MAE: ", np.mean( abs(pred_err)) )
            print("Max error: ", max(abs(pred_err)))


        print("Analyzing where predictions are the worst...")
        Warning("This part is not working yet")
       
        # def k_largest_index_argsort(a, k):
        #     # Helper function to find indices for k largest values
        #     idx = np.argsort(a.ravel())[:-k-1:-1]
        #     return np.column_stack(np.unravel_index(idx, a.shape))

        # n_test_snaps = 20
        # n_atoms = 264

        # abs_errors = abs(predictions_errors).reshape(n_test_snaps, n_atoms,-1)

        # worst_indices = k_largest_index_argsort(abs_errors, 20)
        # worst_atoms = worst_indices[:,1]
        # worst_snapshots = worst_indices[:,1]

        # print("Atomic indices with worst predictions:", worst_atoms)

        # if view_worst_atoms:
        #     view( [ self.fdm.traj_test[0][worst_atoms], self.fdm.traj_test[0] ]  ) #+ self.fdm.traj_test[0])

        #     # make plot of worst errors (debug needed)
        #     errors_list = [abs_errors[tuple(i)] for i in worst_indices]
        #     xs = np.arange(len(errors_list))
        #     ys = errors_list
        #     plt.plot(xs, ys, marker='x', linestyle='None')
        #     ind=0
        #     for x,y in zip(xs,ys):

        #         label = "{:.2f}".format(y)

        #         plt.annotate(worst_indices[ind], # this is the text
        #                     (x,y), # these are the coordinates to position the label
        #                     textcoords="offset points", # how to position the text
        #                     xytext=(0,10), # distance from text to points (x,y)
        #                     ha='center') # horizontal alignment can be left, right or center
        #         ind=ind+1
        #     plt.show()


        # return abs_errors, worst_indices

        return per_model_RMSE, per_model_MAE


    # def get_xtb_energy(self, atoms):

    #     atoms_ = atoms.copy()
    #     atoms_.calc = self.xtb_calc

    #     res_ = atoms_.get_potential_energy()

    #     del atoms_

    #     return res_


    # def get_xtb_forces(self, atoms):

    #     atoms_ = atoms.copy()
    #     atoms_.calc = self.xtb_calc

    #     res_ = atoms_.get_forces()

    #     del atoms_

    #     return res_

    def move_models_to_device(self, device):
        #to add: move also ag_models to device...

        if str(device)[0:4]=='cuda' and not str(device)[-1].isdigit():
            raise ValueError("Please provide a cuda device with a specific id, e.g. cuda:0")

        self.energy_model_device = device

        self.energy_model = self.energy_model.to(device)
        self.energy_model.model.likelihood = self.energy_model.model.likelihood.to(device)
        self.energy_model.model.model = self.energy_model.model.model.to(device)
        self.energy_model.model.model.train()
        self.energy_model.model.likelihood.train()
        self.energy_model.model.model.eval()
        self.energy_model.model.likelihood.eval()

        self.ag_force_model_device = device

        for model in self.ag_force_model.models:
            model.model = model.model.to(device)
            model.likelihood = model.likelihood.to(device)
            # model.model = model.model.model.to(device)
            model.model.train()
            model.likelihood.train()
            model.model.eval()
            model.likelihood.eval()

        return


    def empty_cuda_cache(self):
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        return