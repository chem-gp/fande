import torch
import numpy as np

from dscribe.descriptors import SOAP


class InvariantsComputer:
    def __init__(self, hparams):

        self.hparams = hparams

        self.soap = SOAP(
            species=["H", "C", "O"],
            periodic=False,
            rcut=5.0,
            sigma=0.5,
            nmax=5,
            lmax=5,
            average="outer",
            crossover=True,
            dtype="float64",
        )

    def soap_single_snapshot(self, snapshot, positions=None):

        print("Starting SOAP calculation...")

        mol_traj = [snapshot]

        if positions is not None:
            self.positions = positions
        else:
            pos = [0, 1, 4, 5]

        derivatives, descriptors = self.soap.derivatives(
            mol_traj,
            positions=[self.positions] * len(mol_traj),
            n_jobs=1,
            # method="analytical"
        )

        print("SOAP calculation done!")

        derivatives = np.expand_dims(derivatives, axis=0)
        descriptors = np.expand_dims(descriptors, axis=0)

        derivatives_flattened = derivatives.reshape(
            derivatives.shape[0], derivatives.shape[1], -1, derivatives.shape[-1]
        )
        descriptors_expanded = np.expand_dims(descriptors, 2)

        # print(derivatives_flattened.shape)
        # print(derivatives_flattened.ndim, descriptors_expanded.ndim)
        derivatives_descriptors = np.concatenate(
            (derivatives_flattened, descriptors_expanded), axis=2
        )

        derivatives_descriptors = derivatives_descriptors.squeeze().astype(np.float64)

        derivatives_descriptors_torch = torch.tensor(derivatives_descriptors)

        return derivatives_descriptors_torch
