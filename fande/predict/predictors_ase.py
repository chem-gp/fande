from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import numpy as np

from fande.compute.compute_soap import InvariantsComputer

from fande.utils import get_vectors_e, get_vectors_f

from dscribe.descriptors import SOAP

import torch
import torchmetrics

from ase.units import Bohr, Hartree

from xtb.ase.calculator import XTB


from ase.visualize import view


class SimplePredictorASE:
    def __init__(self, hparams, model_e, trainer_e, model_f, trainer_f):
        
        self.hparams = hparams
        self.trainer_e = trainer_e
        self.model_e = model_e
        self.trainer_f = trainer_f
        self.model_f = model_f

        self.soap_computer = InvariantsComputer(hparams)

        self.n_molecules = 1
        self.n_atoms = 12
        self.batch_size = 100_000

        self.soap_full = None



    def predict_single_energy(self, snapshot, positions):

        x = self.soap_computer.soap_single_snapshot(snapshot, positions)
        self.soap_full = x

        x = x.view(3*self.n_atoms+1, self.n_molecules, -1).transpose(0,1)
        x = x[:, -1, :]

        y_dummy = torch.tensor( [0.0] )

        test = TensorDataset(x, y_dummy)
        test_dl = DataLoader(test, batch_size=self.batch_size)

        res = self.trainer_e.predict(self.model_e, test_dl)[0]

        predictions_torch = res.mean
        variances_torch = res.variance

        energy = predictions_torch.cpu().detach().numpy()
        variance = variances_torch.cpu().detach().numpy()

        return energy, variance


    def predict_single_forces(self, snapshot, positions):

        if self.soap_full is not None:
            x = self.soap_full
        else:
            x = self.soap_computer.soap_single_snapshot(snapshot, positions)
            self.soap_full = x


        x = x.view(3*self.n_atoms+1, self.n_molecules,-1).transpose(0,1)
        x = x[:,:-1, :].squeeze()

        y_dummy = torch.zeros(3*self.n_atoms)

        test = TensorDataset(x, y_dummy)
        test_dl = DataLoader(test, batch_size=self.batch_size)
        res = self.trainer_f.predict(self.model_f, test_dl)[0]

        predictions_torch = res.mean
        variances_torch = res.variance

        f_ = predictions_torch.cpu().detach().numpy()
        f_var_ = variances_torch.cpu().detach().numpy()

        return f_, f_var_


    # def predict_single_forces(self, snapshot, positions):

    #     if self.soap_full is not None:
    #         x = self.soap_full
    #     else:
    #         x = self.soap_computer.soap_single_snapshot(snapshot, positions)
    #         self.soap_full = x


    #     x = x.view(3*self.n_atoms+1, self.n_molecules,-1).transpose(0,1)
    #     x = x[:,:-1, :].squeeze()

    #     y_dummy = torch.zeros(3*self.n_atoms)

    #     test = TensorDataset(x, y_dummy)
    #     test_dl = DataLoader(test, batch_size=self.batch_size)
    #     res = self.trainer_f.predict(self.model_f, test_dl)[0]

    #     predictions_torch = res.mean
    #     variances_torch = res.variance

    #     f_ = predictions_torch.cpu().detach().numpy()
    #     f_var_ = variances_torch.cpu().detach().numpy()

    #     return f_, f_var_

    def unnormalize_energy(self):
        ...

    def unnormalize_forces(self):
        ...


class PredictorASE:
    def __init__(
        self,
        fdm,
        model_e,
        model_f,
        trainer_e,
        trainer_f,
        hparams,
        soap_params
    ):

        self.fdm = fdm
        self.hparams = hparams
        self.soap_params = soap_params

        self.model_e = model_e
        self.model_f = model_f

        self.trainer_e = trainer_e
        self.trainer_f = trainer_f



        self.test_X = fdm.test_X
        self.test_DX = fdm.test_DX
        self.test_E = fdm.test_E
        self.test_F = fdm.test_F

        self.test_traj = fdm.traj_test

        self.n_molecules = fdm.test_E.shape[0]

        self.n_atoms = 1

        self.batch_size = 1000_000

        self.xtb_calc = XTB(method="GFN2-xTB")

    def predict_and_plot_energies(self):

        # print(self.n_molecules, self.n_atoms)
        test_x_e, test_y_e = get_vectors_e(
            self.test_X, self.test_F, self.n_molecules, self.n_atoms
        )

        # print(test_y_e.shape)
        test = TensorDataset(test_x_e, test_y_e)
        test_dl = DataLoader(test, batch_size=self.batch_size)

        res = self.trainer_e.predict(self.model_e, test_dl)[0]

        predictions_torch = res.mean
        variances_torch = res.variance
        # print(variances_torch)

        # variances_torch = res.variance#res.confidence_region()
        # print(variances_torch)

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

            # print("HI")

            predictions = res.mean.cpu().detach().numpy()
            actual_values = self.test_F.cpu().detach().numpy()
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


            # predicted_forces = predictions.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)       
            # # upper_forces = upper.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)
            # # lower_forces = upper.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)      
            # actual_forces = self.test_F.numpy()
            # actual_forces = actual_forces.reshape(3, self.n_atoms, -1).transpose(2, 1, 0)

            # predicted_forces = np.concatenate( (predicted_forces[:,:,0], predicted_forces[:,:,1], predicted_forces[:,:,2]) )
            # # upper_forces = np.concatenate( (upper_forces[:,:,0], upper_forces[:,:,1], upper_forces[:,:,2]) )
            # # lower_forces = np.concatenate( (lower_forces[:,:,0], lower_forces[:,:,1], lower_forces[:,:,2]) )
            # actual_forces = np.concatenate( (actual_forces[:,:,0], actual_forces[:,:,1], actual_forces[:,:,2]) )

            # predicted_energies = predictions[-self.n_molecules :]
            # # upper_energies = upper[-self.n_molecules :]
            # # lower_energies = lower[-self.n_molecules :]

            # actual_energies = self.test_E

            # # l = self.test_shape[0]
            # pred_forces = predicted_forces
            # test_forces = actual_forces

            # pred_energies = predicted_energies
            # test_energies = actual_energies.cpu().detach().numpy()

            # # FMIN = self.forces_energies.min()
            # # FMAX = self.forces_energies.max()

            # FMIN = self.test_F.min()
            # FMAX = self.test_F.max()

            # x_axis = range(200)
            # # lower_energies = 0.1*np.ones(400)
            # # upper_energies = 0.2*np.ones(400)
            # # print(lower_energies, upper_energies)
            # plt.rcParams["figure.figsize"] = (30, 5)
            # plt.plot(pred_energies, color="blue", label="predictions", linewidth=0.4)
            # plt.plot(test_energies, color="red", label="actual values", linewidth=0.4)
            # # plt.fill_between(
            # #     x_axis,
            # #     pred_energies - lower_energies,
            # #     pred_energies + upper_energies,
            # #     color="b",
            # #     alpha=0.1,
            # #     label="Confidence of prediction",
            # # )
            # plt.title(f"Energies")
            # plt.legend()
            # # plt.axvspan(0, l, facecolor='purple', alpha=0.05)
            # # plt.ylim(FMIN, FMAX)
            # # wandb.log({"energy": wandb.Image(plt)})
            # plt.show()

            # # variances_torch = res.variance
            # # print(type(res.confidence_region()))

            # x_axis = np.concatenate(
            #     (
            #         np.arange(0, self.n_molecules, 100),
            #         np.arange(0, self.n_molecules, 100),
            #         np.arange(0, self.n_molecules, 100),
            #     )
            # )

            # full_x = np.arange(0, 3 * self.n_molecules, 100).tolist()
            # # lower_forces = pred_forces - lower_forces
            # # upper_forces = pred_forces + upper_forces

            # x_axis_forces = np.arange(3 * self.n_molecules).tolist()

            # mol = self.test_traj[0]
            # plt.rcParams["figure.figsize"] = (30, 5)
            # l = self.n_molecules
            # for fatom in range(self.n_atoms):

            #     plt.plot(
            #         pred_forces[:, fatom], color="blue", label="predictions", linewidth=0.3
            #     )
            #     plt.plot(
            #         test_forces[:, fatom], color="red", label="actual values", linewidth=0.3
            #     )

            #     # print(lower_forces[:,fatom],upper_forces[:,fatom])

            #     # plt.fill_between(
            #     #     x_axis_forces,
            #     #     lower_forces[:,fatom],
            #     #     upper_forces[:,fatom],
            #     #     color="b",
            #     #     alpha=0.1,
            #     #     label="Confidence of prediction"
            #     # )

            #     plt.title(f"Forces, atom {fatom} : {mol[fatom].symbol}")
            #     plt.legend()
            #     plt.axvspan(0, l, facecolor="purple", alpha=0.05)
            #     plt.axvspan(l, 2 * l, facecolor="green", alpha=0.05)
            #     plt.axvspan(2 * l, 3 * l, facecolor="orange", alpha=0.05)
            #     plt.ylim(FMIN, FMAX)
            #     plt.text(l / 2, FMAX / 2, r"$F_x$", fontsize=44, alpha=0.1)
            #     plt.text(3 * l / 2, FMAX / 2, r"$F_y$", fontsize=44, alpha=0.1)
            #     plt.text(5 * l / 2, FMAX / 2, r"$F_z$", fontsize=44, alpha=0.1)
            #     # wandb.log({f"atom {fatom} : {mol[fatom].symbol}": wandb.Image(plt) })
            #     # plt.xticks(full_x, x_axis) # check(there's some error)

            #     plt.show()
                
            #     self.f_mae = torchmetrics.functional.mean_absolute_error(
            #         torch.tensor(pred_forces[:, fatom]), torch.tensor(test_forces[:, fatom])
            #     )
            #     self.f_mse = torchmetrics.functional.mean_squared_error(
            #         torch.tensor(pred_forces[:, fatom]), torch.tensor(test_forces[:, fatom])
            #     )
            #     print("Forces MAE: %5.4f" % self.f_mae.item())
            #     print("Forces MSE: %5.4f" % self.f_mse.item())
            #     # print("Cumulative uncertainty: %5.4f" % np.sum(upper_forces[:,fatom] - lower_forces[:,fatom]) )

            return


    def predict_forces_single_snapshot_r(self, snapshot):

            # if self.hparams["device"] == "gpu":
            #     self.model_f = self.model_f.cuda()  # PL moves params to cpu (what a mess!)

            n_atoms = len(snapshot)

            self.fdm.calculate_snapshot_invariants_librascal(snapshot)

            snap_DX = self.fdm.snap_DX

            snap_DX = torch.tensor(snap_DX, dtype = torch.float32 )
            zeros_F = torch.zeros_like(snap_DX[:,0])

            print(snap_DX.shape, zeros_F.shape)

            test = TensorDataset(snap_DX, zeros_F)
            test_dl = DataLoader(test, batch_size=self.batch_size)

            res = self.trainer_f.predict(self.model_f, test_dl)[0]

            predictions_torch = res.mean

            print("predictions done!")

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

            
            pred_forces = predictions.reshape((n_atoms, 3))

            return pred_forces


    def test_errors(self, plot=False, view_worst_atoms=False):
        ## predictor maximal error with respect to fdm.test_DX and fdm.test_F

        test = TensorDataset(self.fdm.test_DX, self.fdm.test_F)
        test_dl = DataLoader(test, batch_size=self.batch_size)

        res = self.trainer_f.predict(self.model_f, test_dl)[0]

        predictions_torch = res.mean

        predictions_errors = predictions_torch.detach().cpu() - self.fdm.test_F.detach().cpu()
        predictions_errors = predictions_errors.numpy()

        if plot:
            plt.plot(predictions_errors)
            plt.show()

            plt.hist(predictions_errors, bins=30)
            plt.show()


        print("MSE: ", np.mean(predictions_errors**2) )
        print("MAE: ", np.mean( abs(predictions_errors)) )
        print("Max error: ", max(abs(predictions_errors)))


        print("Analyzing where predictions are the worst...")          
        def k_largest_index_argsort(a, k):
            # Helper function to find indices for k largest values
            idx = np.argsort(a.ravel())[:-k-1:-1]
            return np.column_stack(np.unravel_index(idx, a.shape))

        n_test_snaps = 20
        n_atoms = 264

        abs_errors = abs(predictions_errors).reshape(n_test_snaps, n_atoms,-1)

        worst_indices = k_largest_index_argsort(abs_errors, 20)
        worst_atoms = worst_indices[:,1]
        worst_snapshots = worst_indices[:,1]

        print("Atomic indices with worst predictions:", worst_atoms)

        if view_worst_atoms:
            view( [ self.fdm.traj_test[0][worst_atoms], self.fdm.traj_test[0] ]  ) #+ self.fdm.traj_test[0])


        return abs_errors, worst_indices


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

        f_var_ = 0.0

        f_ = f_ * self.fdm.normalizing_factor
        f_var_ = f_var_ * self.fdm.normalizing_factor

        # Check with XTB values:
        # f_ = self.get_xtb_forces(snapshot)
        # f_var_ = np.zeros_like(f_)


        return f_, f_var_



    def soap_single(self, snapshot):

        soap_params = self.soap_params

        species= soap_params['species']
        periodic= soap_params['periodic']
        rcut= soap_params['rcut']
        sigma= soap_params['sigma']
        nmax= soap_params['nmax']
        lmax= soap_params['lmax']
        average= soap_params['average']
        crossover= soap_params['crossover']
        dtype= soap_params['dtype']
        sparse= soap_params['sparse']
        positions = soap_params['positions']

        soap = SOAP(
            species=species,
            periodic=periodic,
            rcut=rcut,
            sigma=sigma,
            nmax=nmax,
            lmax=lmax,
            average=average,
            crossover=crossover,
            dtype=dtype,
            sparse=sparse  
        )

        snap = [snapshot]

        print("Starting SOAP calculation...")
        derivatives, descriptors = soap.derivatives(
            snap,
            positions=[positions] * len(snap),
            n_jobs=1,
            # method="analytical"
        )
        print("SOAP calculation done!")
        derivatives = derivatives.squeeze()
        descriptors = descriptors.squeeze()

        # print(derivatives.shape)

        x = torch.tensor(descriptors)    

        dx = torch.tensor(
                derivatives.transpose(1,0,2).reshape(
            derivatives.shape[0]*derivatives.shape[1], -1
        ))

        # print(dx.shape)


        if self.hparams['device'] == 'gpu':
            x = x.cuda()
            dx = dx.cuda()

        return x, dx


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