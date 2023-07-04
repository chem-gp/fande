import numpy as np
from ase import io
import matplotlib.pyplot as plt

import os

from fande.plot import atom_forces_3d


from xtb.ase.calculator import XTB

# from dscribe.descriptors import SOAP

from tqdm import tqdm


class FastLoader:
    def __init__(self, hparams, **kwargs):

        self.hparams = hparams

        self.energies = None
        self.forces = None

        self.energies_norm = None
        self.forces_norm = None

    def rotate_traj(self, mol_traj):
        energies_np = np.zeros(len(mol_traj))
        forces_np = np.zeros((len(mol_traj), len(mol_traj[0]), 3))

        for i, mol in tqdm(enumerate(mol_traj)):
            mol.calc = XTB(method="GFN2-xTB")
            # a = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
            # a.positions = a.positions + 0.01*np.random.rand(n_atoms,3 )

            # a.positions[7] = a.positions[7] + np.array([0.0, 0.0, 1.5*np.random.rand(1)] ) + np.array([1.5,1.5,1.5])
            
            angles = 1000*np.random.rand(3)

            mol.euler_rotate(phi=angles[0], theta=angles[1], psi=angles[2], center=(0, 0, 0))
            # new_traj.append(mol)# hmm.. you can also rotate forces by creating dummy molecules and rotating them
            energies_np[i] = mol.get_potential_energy()
            forces_np[i, :, :] = mol.get_forces()
            mol.calc = None

        io.write("data/dump/rotated_traj.xyz", mol_traj, format="extxyz")

        return mol_traj, energies_np, forces_np
        
    def load_and_caculate(self):

        os.chdir(self.hparams["root_dir"])
        print(os.getcwd())
        mol_traj = io.read("data/dump/mol_trj.xyz", index="15000:17000")

        # energies_np = np.load("data/dump/energies_np.npy")
        # forces_np = np.load("data/dump/forces_np.npy")

        L = 2000

        # mol_traj = (
        #     mol_traj[0 : L - 400] + mol_traj[4400:4800]
        # )  # put some piece trajectory for testing
        mol_traj = mol_traj[0:L]

        print(f"Total length of mol_traj is {len(mol_traj)}")

        mol_traj, energies_np, forces_np = self.rotate_traj(mol_traj)

        # energies_np = energies_np[0:L]
        # forces_np = forces_np[0:L]

        n_samples = len(mol_traj)

        mol = mol_traj[0]


        self.energies = energies_np
        self.forces = forces_np

        ## shuffle traj and energies/forces:
        # np.random.seed(123)

        # shuffled_indices = np.random.permutation(n_samples)

        # mol_traj_shuffled = [mol_traj[ind] for ind in shuffled_indices]
        # io.write("data/dump/mol_trj_shuffled.xyz", mol_traj_shuffled, format="extxyz")
        # energies_np = energies_np[shuffled_indices]
        # forces_np = forces_np[shuffled_indices]

        # print(shuffled_indices)

        # da_dx = derivatives[:, 0, fatom, 0]

        #without normalization!
        # forces = forces_np / (np.max(energies_np) - np.min(energies_np))
        # energies = (energies_np - np.min(energies_np)) / (
        #     np.max(energies_np) - np.min(energies_np)
        # )


        self.energies_norm = self.energies
        self.forces_norm = self.forces

        forces = self.forces
        energies = self.energies


        plt.plot(energies)
        plt.title("Energies")
        plt.show()

        # plt.plot(f)
        # plt.title(r"$F_{x}$")
        # plt.show()

        # atom_forces_3d(mol_traj, forces_np, range(0,11))

        ########################################################################
        ### Compute SOAP
        from dscribe.descriptors import SOAP

        # np.random.seed(33)
        # Setting up the SOAP descriptor
        soap = SOAP(
            species=["H", "C", "O"],
            periodic=False,
            rcut=5.0,
            sigma=0.5,
            nmax=5,
            lmax=5,
            average="outer",
            crossover=True,
            dtype="float64",  # self.hparams['dtype']
        )

        print("Starting SOAP calculation...")
        # pos = [0,1,2,3,4,5,6,7,8,9,10,11]
        pos = [0, 1, 4, 5]
        # pos = [[-0.00358347365689,    0.05151389483731 ,   0.37078570243174]]
        # pos = [1]
        derivatives, descriptors = soap.derivatives(
            mol_traj,
            positions=[pos] * len(mol_traj),
            n_jobs=10,
            # method="analytical"
        )
        print("SOAP calculation done!")

        derivatives_flattened = derivatives.reshape(
            derivatives.shape[0], derivatives.shape[1], -1, derivatives.shape[-1]
        )
        descriptors_expanded = np.expand_dims(descriptors, 2)
        # print(derivatives_flattened.shape)
        # print(derivatives_flattened.ndim, descriptors_expanded.ndim)
        derivatives_descriptors = np.concatenate(
            (derivatives_flattened, descriptors_expanded), axis=2
        )
        # print(derivatives_descriptors.shape) # (1001, 1, 37, 3780)
        # np.array_equal( descriptors, derivatives_descriptors[...,-1,:] )

        print(soap.get_number_of_features())

        derivatives_descriptors = derivatives_descriptors.squeeze().astype(np.float64)
        forces_energies = np.concatenate(
            (forces.reshape(forces.shape[0], -1), energies[:, None]), axis=1
        ).astype(np.float64)
        # np.array_equal(forces_energies[:,-1],energies)

        print(derivatives_descriptors.shape, forces_energies.shape)
        print(derivatives_descriptors.dtype, forces_energies.dtype)

        return derivatives_descriptors, forces_energies, forces_np, energies_np, mol, mol_traj


