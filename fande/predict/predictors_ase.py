from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import numpy as np

from fande.compute.compute_soap import InvariantsComputer

from fande.utils import get_vectors_e, get_vectors_f

import torch
import torchmetrics


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

    def unnormalize_energy(self):
        ...

    def unnormalize_forces(self):
        ...


class PredictorASE:
    def __init__(
        self,
        model_e,
        model_f,
        trainer_e,
        trainer_f,
        test_X,
        test_DX,
        test_E,
        test_F,
        test_data,
        hparams
    ):

        self.hparams = hparams
        self.model_e = model_e
        self.model_f = model_f

        self.trainer_e = trainer_e
        self.trainer_f = trainer_f

        self.test_X = test_X
        self.test_DX = test_DX
        self.test_E = test_E
        self.test_F = test_F

        self.test_traj = test_data['trajectory']

        self.n_molecules = test_E.shape[0]

        self.n_atoms = 17

        self.batch_size = 1000_000

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

        lower, upper = res.confidence_region()
        lower = 0.1 * lower.cpu().detach().numpy()
        upper = 0.1 * upper.cpu().detach().numpy()
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


        # predicted_forces = (
        #     predictions.reshape(self.n_atoms, -1)
        #     .transpose(1, 0)
        # )

        # upper_forces = (
        #     upper.reshape(self.n_atoms, -1)
        #     .transpose(1, 0)
        # )

        # lower_forces = (
        #     upper.reshape(self.n_atoms, -1)
        #     .transpose(1, 0)
        # )

        # actual_forces = (
        #     self.test_F.reshape(self.n_atoms, -1)
        #     .transpose(1, 0)
        # )


        predicted_forces = (
            predictions.reshape(3, self.n_atoms, -1)
            .transpose(2, 1, 0)
        )

        upper_forces = (
            upper.reshape(3, self.n_atoms, -1)
            .transpose(2, 1, 0)
        )

        lower_forces = (
            upper.reshape(3, self.n_atoms, -1)
            .transpose(2, 1, 0)
        )

        actual_forces = (
            self.test_F.reshape(3, self.n_atoms, -1)
            .transpose(2, 1, 0)
        )

        predicted_energies = predictions[-self.n_molecules :]
        upper_energies = upper[-self.n_molecules :]
        lower_energies = lower[-self.n_molecules :]

        actual_energies = self.test_E

        # l = self.test_shape[0]
        pred_forces = predicted_forces
        test_forces = actual_forces.cpu().detach().numpy()

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
        lower_forces = pred_forces - lower_forces
        upper_forces = pred_forces + upper_forces
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

            plt.fill_between(
                x_axis_forces,
                lower_forces[:,fatom],
                upper_forces[:,fatom],
                color="b",
                alpha=0.1,
                label="Confidence of prediction"
            )

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
            print("Cumulative uncertainty: %5.4f" % np.sum(upper_forces[:,fatom] - lower_forces[:,fatom]) )

        return
