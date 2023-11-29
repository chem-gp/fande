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

        # self.model_e = model_e

        self.energy_model = energy_model

        self.ag_force_model = atomic_group_force_model


        self.test_X = fdm.test_X
        self.test_DX = fdm.test_DX
        self.test_E = fdm.test_E
        self.test_F = fdm.test_F

        self.test_traj = fdm.traj_test

        self.n_atoms = 1

        self.batch_size = 1000_000

 
    def predict_and_plot_energies(self):

        raise NotImplementedError

        test_x_e, test_y_e = get_vectors_e(
            self.test_X, self.test_F, self.n_molecules, self.n_atoms
        )

        test = TensorDataset(test_x_e, test_y_e)
        test_dl = DataLoader(test, batch_size=self.batch_size)

        res = self.trainer_e.predict(self.model_e, test_dl)[0]

        predictions_torch = res.mean
        variances_torch = res.variance

        predictions = res.mean.cpu().detach().numpy()
        actual_values = test_y_e.cpu().detach().numpy()
        plt.rcParams["figure.figsize"] = (30, 10)
        plt.plot(predictions, color="blue", label="predictions", linewidth=0.4)
        plt.plot(actual_values, color="red", label="actual values", linewidth=0.4)
        plt.legend()
        plt.ylabel("Energy")
        plt.xlabel("Index")
        # plt.xlim(3000,4000)
        # plt.ylim(0, 1.0)
        # wandb.log({"all prediction": wandb.Image(plt)})
        plt.show()

        self.e_mae = torchmetrics.functional.mean_absolute_error(
            predictions_torch.cpu(), test_y_e.cpu()
        )
        self.e_mse = torchmetrics.functional.mean_squared_error(
            predictions_torch.cpu(), test_y_e.cpu()
        )
        print("\nEnergies MAE: %5.4f" % self.e_mae.item())
        print("Energies MSE: %5.4f" % self.e_mse.item())

        return

    def predict_and_plot_forces(self):

        # if self.hparams["device"] == "gpu":
        #     self.model_f = self.model_f.cuda()  # PL moves params to cpu (what a mess!)

        test = TensorDataset(self.test_DX, self.test_F)
        test_dl = DataLoader(test, batch_size=self.batch_size)
        res = self.trainer_f.predict(self.model_f, test_dl)[0]

        predictions_torch = res.mean

        # variances_torch = res.variance
        # print(variances_torch, res.confidence_region())

        # lower, upper = res.confidence_region()
        # lower = 0.1 * lower.cpu().detach().numpy()
        # upper = 0.1 * upper.cpu().detach().numpy()

        # lower = lower.tolist()
        # upper = upper.tolist()
        # print(lower, upper)

        predictions = res.mean.cpu().detach().numpy()
        actual_values = self.test_F.cpu().detach().numpy()
        plt.rcParams["figure.figsize"] = (30, 10)
        plt.plot(predictions, color="blue", label="predictions", linewidth=0.2)
        plt.plot(actual_values, color="red", label="actual values", linewidth=0.2)

        plt.legend()
        # plt.xlim(3000,4000)
        # plt.ylim(0, 1.0)
        # wandb.log({"all prediction": wandb.Image(plt)})
        plt.show()
        f_mae = torchmetrics.functional.mean_absolute_error(
            torch.tensor(predictions), torch.tensor(actual_values)
        )
        f_mse = torchmetrics.functional.mean_squared_error(
            torch.tensor(predictions), torch.tensor(actual_values)
        )
        print("Forces MAE: %5.4f" % f_mae.item())
        print("Forces MSE: %5.4f" % f_mse.item())
        # print("Cumulative uncertainty: %5.4f" % np.sum(upper_forces[:,fatom] - lower_forces[:,fatom]) )


        predicted_forces = predictions.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)       
        # upper_forces = upper.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)
        # lower_forces = upper.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)      
        actual_forces = self.test_F.numpy()
        actual_forces = actual_forces.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)

        predicted_forces = np.concatenate( (predicted_forces[:,:,0], predicted_forces[:,:,1], predicted_forces[:,:,2]) )
        # upper_forces = np.concatenate( (upper_forces[:,:,0], upper_forces[:,:,1], upper_forces[:,:,2]) )
        # lower_forces = np.concatenate( (lower_forces[:,:,0], lower_forces[:,:,1], lower_forces[:,:,2]) )
        actual_forces = np.concatenate( (actual_forces[:,:,0], actual_forces[:,:,1], actual_forces[:,:,2]) )

        predicted_energies = predictions[-self.n_molecules :]
        # upper_energies = upper[-self.n_molecules :]
        # lower_energies = lower[-self.n_molecules :]

        actual_energies = self.test_E

        # l = self.test_shape[0]
        pred_forces = predicted_forces
        test_forces = actual_forces

        pred_energies = predicted_energies
        test_energies = actual_energies.cpu().detach().numpy()

        # FMIN = self.forces_energies.min()
        # FMAX = self.forces_energies.max()

        FMIN = self.test_F.min()
        FMAX = self.test_F.max()

        x_axis = range(200)
        # lower_energies = 0.1*np.ones(400)
        # upper_energies = 0.2*np.ones(400)
        # print(lower_energies, upper_energies)
        plt.rcParams["figure.figsize"] = (30, 5)
        plt.plot(pred_energies, color="blue", label="predictions", linewidth=0.4)
        plt.plot(test_energies, color="red", label="actual values", linewidth=0.4)
        # plt.fill_between(
        #     x_axis,
        #     pred_energies - lower_energies,
        #     pred_energies + upper_energies,
        #     color="b",
        #     alpha=0.1,
        #     label="Confidence of prediction",
        # )
        plt.title(f"Energies")
        plt.legend()
        # plt.axvspan(0, l, facecolor='purple', alpha=0.05)
        # plt.ylim(FMIN, FMAX)
        # wandb.log({"energy": wandb.Image(plt)})
        plt.show()

        # variances_torch = res.variance
        # print(type(res.confidence_region()))

        x_axis = np.concatenate(
            (
                np.arange(0, self.n_molecules, 100),
                np.arange(0, self.n_molecules, 100),
                np.arange(0, self.n_molecules, 100),
            )
        )

        full_x = np.arange(0, 3 * self.n_molecules, 100).tolist()
        # lower_forces = pred_forces - lower_forces
        # upper_forces = pred_forces + upper_forces

        x_axis_forces = np.arange(3 * self.n_molecules).tolist()

        mol = self.test_traj[0]
        plt.rcParams["figure.figsize"] = (30, 5)
        l = self.n_molecules
        for fatom in range(self.n_atoms):

            plt.plot(
                pred_forces[:, fatom], color="blue", label="predictions", linewidth=0.3
            )
            plt.plot(
                test_forces[:, fatom], color="red", label="actual values", linewidth=0.3
            )

            # print(lower_forces[:,fatom],upper_forces[:,fatom])

            # plt.fill_between(
            #     x_axis_forces,
            #     lower_forces[:,fatom],
            #     upper_forces[:,fatom],
            #     color="b",
            #     alpha=0.1,
            #     label="Confidence of prediction"
            # )

            plt.title(f"Forces, atom {fatom} : {mol[fatom].symbol}")
            plt.legend()
            plt.axvspan(0, l, facecolor="purple", alpha=0.05)
            plt.axvspan(l, 2 * l, facecolor="green", alpha=0.05)
            plt.axvspan(2 * l, 3 * l, facecolor="orange", alpha=0.05)
            plt.ylim(FMIN, FMAX)
            plt.text(l / 2, FMAX / 2, r"$F_x$", fontsize=44, alpha=0.1)
            plt.text(3 * l / 2, FMAX / 2, r"$F_y$", fontsize=44, alpha=0.1)
            plt.text(5 * l / 2, FMAX / 2, r"$F_z$", fontsize=44, alpha=0.1)
            # wandb.log({f"atom {fatom} : {mol[fatom].symbol}": wandb.Image(plt) })
            # plt.xticks(full_x, x_axis) # check(there's some error)

            plt.show()
            
            self.f_mae = torchmetrics.functional.mean_absolute_error(
                torch.tensor(pred_forces[:, fatom]), torch.tensor(test_forces[:, fatom])
            )
            self.f_mse = torchmetrics.functional.mean_squared_error(
                torch.tensor(pred_forces[:, fatom]), torch.tensor(test_forces[:, fatom])
            )
            print("Forces MAE: %5.4f" % self.f_mae.item())
            print("Forces MSE: %5.4f" % self.f_mse.item())
            # print("Cumulative uncertainty: %5.4f" % np.sum(upper_forces[:,fatom] - lower_forces[:,fatom]) )

        return
    

    def predict_and_plot_forces_r(self):

            # if self.hparams["device"] == "gpu":
            #     self.model_f = self.model_f.cuda()  # PL moves params to cpu (what a mess!)

            for idx, model in enumerate(self.ag_force_model.models): 

                test = TensorDataset(self.fdm.test_DX, self.fdm.test_F)
                test_dl = DataLoader(test, batch_size=self.batch_size)
                res = self.trainer_f.predict(self.model_f, test_dl)[0]

                predictions_torch = res.mean

                print(self.test_F.shape)


                predictions = res.mean.cpu().detach().numpy()
                actual_values = self.fdm.test_F.cpu().detach().numpy()
                plt.rcParams["figure.figsize"] = (30, 10)

                predictions_xyz = np.concatenate( (predictions[0::3], predictions[1::3], predictions[2::3]))
                actual_values_xyz = np.concatenate( (actual_values[0::3], actual_values[1::3], actual_values[2::3]))

                plt.plot(predictions_xyz, color="blue", label="predictions", linewidth=0.4)
                plt.plot(actual_values_xyz, color="red", label="actual values", linewidth=0.4)

                plt.legend()
                # plt.xlim(3000,4000)
                # plt.ylim(0, 1.0)
                # wandb.log({"all prediction": wandb.Image(plt)})
                plt.show()
                f_mae = torchmetrics.functional.mean_absolute_error(
                    torch.tensor(predictions), torch.tensor(actual_values)
                )
                f_mse = torchmetrics.functional.mean_squared_error(
                    torch.tensor(predictions), torch.tensor(actual_values)
                )
                print("Forces MAE: %5.4f" % f_mae.item())
                print("Forces MSE: %5.4f" % f_mse.item())
            # print("Cumulative uncertainty: %5.4f" % np.sum(upper_forces[:,fatom] - lower_forces[:,fatom]) )


            return



    def predict_forces_single_snapshot_r(self, snapshot, atomic_groups=None):

            n_atoms = len(snapshot)
            X, DX_grouped = self.fdm.calculate_snapshot_invariants_librascal(snapshot)
            atomic_groups = self.fdm.atomic_groups_train
            # snap_DX = self.fdm.snap_DX

            # print(snap_DX)
            # predictions_grouped = []
            forces = np.zeros((n_atoms, 3))
            forces_variance = np.zeros((n_atoms, 3))
            for idx, model in enumerate(self.ag_force_model.models):              

                # res = trainer_f.predict(model, test_dl)[0]
                model = model.cuda()
                model.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    res = model(DX_grouped[idx].cuda()) # should you move to a device with specific id? for now it works...
        
                # predictions_torch = res.mean

                predictions = res.mean.cpu().detach().numpy()

                predictions_variance = res.variance.cpu().detach().numpy()
                         
                n_atoms_in_group = len(atomic_groups[idx])
                pred_forces = predictions.reshape((n_atoms_in_group, 3))
                pred_forces_variance = predictions_variance.reshape((n_atoms_in_group, 3))
                # predictions_grouped.append(pred_forces)

                forces[sorted(atomic_groups[idx])] = pred_forces
                forces_variance[sorted(atomic_groups[idx])] = pred_forces_variance
            

            return forces, forces_variance


    def test_errors(self, view_worst_atoms=False):
        """
        Provides test errors metrics for the models.

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


    def predict_energy_single(self,snapshot):
        # Check with XTB values:
        # e_ = self.get_xtb_energy(snapshot)
        # e_var_ = np.zeros_like(e_)

        e_ = 0.0
        e_var_ = 0.0

        return e_, e_var_  


    def predict_forces_single(self, snapshot):

        x, dx = self.soap_single(snapshot)

        # dx = dx.view(3*self.n_atoms,-1).transpose(0,1)
        # dx = dx[:,:-1, :].squeeze()

        y_dummy = torch.zeros(3*self.n_atoms)

        test = TensorDataset(dx, y_dummy)
        test_dl = DataLoader(test, batch_size=self.batch_size)
        res = self.trainer_f.predict(self.model_f, test_dl)[0]

        predictions_torch = res.mean
        # variances_torch = res.variance

        f_ = predictions_torch.cpu().detach().numpy()
        # f_var_ = variances_torch.cpu().detach().numpy()

        f_ = f_.reshape(3, self.n_atoms).transpose(1, 0)
        # f_var_ = f_var_.reshape(3, self.n_atoms).transpose(1, 0)     

        # f_ = f_.reshape(self.n_atoms, 3)
        # f_var_ = f_var_.reshape(self.n_atoms, 3)
        #
        # print("Normalizing factor", self.fdm.normalizing_factor)

        # moved to the PredictorASE:
        # f_ = f_ * self.fdm.normalizing_factor * Hartree / Bohr
        # f_var_ = f_var_ * self.fdm.normalizing_factor * Hartree / Bohr

        # f_var_ = 0.0

        # f_ = f_ * self.fdm.normalizing_factor
        # f_var_ = f_var_ * self.fdm.normalizing_factor

        # Check with XTB values:
        # f_ = self.get_xtb_forces(snapshot)
        # f_var_ = np.zeros_like(f_)


        return f_, f_var_



    def get_xtb_energy(self, atoms):

        atoms_ = atoms.copy()
        atoms_.calc = self.xtb_calc

        res_ = atoms_.get_potential_energy()

        del atoms_

        return res_


    def get_xtb_forces(self, atoms):

        atoms_ = atoms.copy()
        atoms_.calc = self.xtb_calc

        res_ = atoms_.get_forces()

        del atoms_

        return res_