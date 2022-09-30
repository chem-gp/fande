# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#datamodules-recommended
from tkinter import NO
from torch.utils.data import DataLoader
import os
import subprocess
from datetime import datetime

from typing import Optional
from pytorch_lightning import LightningDataModule

import torch
import numpy as np


from fande.data import FastLoader

from fande.utils import get_vectors_e, get_vectors_f

from dscribe.descriptors import SOAP

from ase import io

from tqdm import tqdm


from functools import lru_cache



class FandeDataModuleASE(LightningDataModule):
    def __init__(self, training_data, test_data, hparams, atoms_forces = None):
        super().__init__()
        self.hparams.update(hparams)

        self.traj_train = training_data['trajectory']
        self.energies_train = training_data['energies']
        self.forces_train = training_data['forces']

        self.traj_test = test_data['trajectory']
        self.energies_test = test_data['energies']
        self.forces_test = test_data['forces']

        self.atoms_forces = atoms_forces
        if atoms_forces is not None:
            self.forces_train = self.forces_train[:,atoms_forces,:]
            # self.forces_test = self.forces_test[:,atoms_forces,:]


        self.train_DX = None
        self.train_F = None
        self.test_DX = None
        self.test_F = None

        self.train_X = None
        self.train_E = None
        self.test_X = None
        self.test_E = None

        forces_train_norm = self.forces_train / (np.max(self.energies_train) - np.min(self.energies_train))
        energies_train_norm = (self.energies_train - np.min(self.energies_train)) / (
            np.max(self.energies_train) - np.min(self.energies_train)
        )

        forces_test_norm = self.forces_test / (np.max(self.energies_train) - np.min(self.energies_train))
        energies_test_norm = (self.energies_test - np.min(self.energies_train)) / (
            np.max(self.energies_train) - np.min(self.energies_train)
        )

        forces_train_norm = forces_train_norm.reshape(
            forces_train_norm.shape[0] * forces_train_norm.shape[1] * forces_train_norm.shape[2], -1
            ).astype(np.float64)

        forces_test_norm = forces_test_norm.reshape(
            forces_test_norm.shape[0] * forces_test_norm.shape[1] * forces_test_norm.shape[2], -1
            ).astype(np.float64)

        self.train_E = torch.tensor( energies_train_norm )
        self.test_E = torch.tensor( energies_test_norm )
        self.train_F = torch.tensor( forces_train_norm )
        self.test_F = torch.tensor( forces_test_norm )


        self.train_F = self.train_F[:, :].transpose(0, 1).flatten(0, 1)
        self.test_F = self.test_F[:, :].transpose(0, 1).flatten(0, 1)


        self.save_hyperparameters()


        # print(self.forces_train[0,1,:])
        # print(self.forces_test.shape)

        # print(forces_train_norm[0,:] * (np.max(self.energies_train) - np.min(self.energies_train) ) )
        # print(forces_test_norm.shape)

        # derivatives_descriptors = derivatives_descriptors.squeeze().astype(np.float64)
        # forces_energies = np.concatenate(
        #     (forces.reshape(forces.shape[0], -1), energies[:, None]), axis=1
        # ).astype(np.float64)
        # # np.array_equal(forces_energies[:,-1],energies)

        # print(derivatives_descriptors.shape, forces_energies.shape)
        # print(derivatives_descriptors.dtype, forces_energies.dtype)

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        print("hi")
        # called on every GPU

    def train_dataloader(self):
        # transforms = ...
        return DataLoader(self.train, batch_size=100_000)

    def val_dataloader(self):
        # ...
        # transforms = ...
        return DataLoader(self.val, batch_size=100_000)

    def test_dataloader(self):
        # ...
        # transforms = ...
        return DataLoader(self.test, batch_size=100_000)


    def calculate_invariants(self, soap_params):


        soap = SOAP(
            species=["H", "C"],
            periodic=False,
            rcut=2.0,
            sigma=0.5,
            nmax=7,
            lmax=7,
            average="outer",
            crossover=True,
            dtype="float64",
            sparse=False  
        )
        pos = [0,1,2,3,4,5,6]

        traj_train = self.traj_train
        traj_test = self.traj_test


        print(f"Total length of train traj is {len(traj_train)}")
        print("Starting SOAP calculation...")
        derivatives_train, descriptors_train = soap.derivatives(
            traj_train,
            positions=[pos] * len(traj_train),
            n_jobs=10,
            # method="analytical"
        )
        print("SOAP calculation done!")
        derivatives_train = derivatives_train.squeeze()
        descriptors_train = descriptors_train.squeeze()

        print(f"Total length of test traj is {len(traj_test)}")
        print("Starting SOAP calculation...")
        derivatives_test, descriptors_test = soap.derivatives(
            traj_test,
            positions=[pos] * len(traj_test),
            n_jobs=10,
            # method="analytical"
        )
        print("SOAP calculation done!")
        derivatives_test = derivatives_test.squeeze()
        descriptors_test = descriptors_test.squeeze()

        if self.atoms_forces is not None:
            derivatives_train = derivatives_train[:, self.atoms_forces,:, :]
            # derivatives_test = derivatives_test[:, self.atoms_forces,:, :]

        self.train_X = torch.tensor(descriptors_train)      
        self.train_DX = torch.tensor(
                derivatives_train.reshape(
            derivatives_train.shape[0]*derivatives_train.shape[1]*derivatives_train.shape[2], -1
        ))

        self.test_X = torch.tensor(descriptors_test)      
        self.test_DX = torch.tensor(
                derivatives_test.reshape(
            derivatives_test.shape[0]*derivatives_test.shape[1]*derivatives_test.shape[2], -1
        ))

        print(derivatives_train.shape)
        print(descriptors_train.shape)

        # derivatives_flattened = derivatives.reshape(
        #     derivatives.shape[0], derivatives.shape[1], -1, derivatives.shape[-1]
        # )


        # train_X = train_X.to(torch.float32)
        # train_Y = train_Y.to(torch.float32)
        # test_X = test_X.to(torch.float32)
        # test_Y = test_Y.to(torch.float32)


    def get_training_data(self, n_atoms=None):
        """
        return training snapshots, energies and forces
        """
        # check this method!

        training_snapshots = self.mol_traj[0:1600]
        # tensor_Y = self.train_Y
        # tensor_Y = tensor_Y.view(3*n_atoms+1,-1).transpose(0,1)

        # training_forces = tensor_Y[:,:-1].squeeze()
        # training_energies = tensor_Y[:,-1].squeeze()
        # training_forces = training_forces.reshape(-1, n_atoms, 3)

        print( self.forces_energies.shape )

        return training_snapshots, training_energies, training_forces




    def prepare_torch_dataset(self, energies, forces, descriptors, derivatives):
        """
        Prepare train/test datasets from raw computed energies/descriptors and forces/derivatives.
        """
        self.n_atoms = forces.shape[1]

        r_test = 0.2
        r_train = 1 - r_test

        n_samples = energies.shape[0]


        n_train = int(r_train * n_samples)
        n_test = n_samples - n_train

        self.n_train_structures = n_train
        self.n_test_structures = n_test

        energies_train = energies[0:n_train]
        forces_train = forces[0:n_train]

        self.normalizing_const = np.max(energies_train) - np.min(energies_train)
        self.normalizing_shift = np.min(energies_train)

        forces = forces / self.normalizing_const
        energies = ( energies - self.normalizing_shift ) / self.normalizing_const
        self.forces_norm = forces
        self.energies_norm = energies


        energies_train = energies[0:n_train]
        forces_train = forces[0:n_train]
        energies_test = energies[n_train:-1]
        forces_test = energies[n_train:-1]

        derivatives_flattened = derivatives.reshape(
            derivatives.shape[0], derivatives.shape[1], -1, derivatives.shape[-1]
        )
        descriptors_expanded = np.expand_dims(descriptors, 2)

        derivatives_descriptors = np.concatenate(
            (derivatives_flattened, descriptors_expanded), axis=2
        )

        derivatives_descriptors = derivatives_descriptors.squeeze().astype(np.float64)
        
        forces_energies = np.concatenate(
            (forces.reshape(forces.shape[0], -1), energies[:, None]), axis=1
        ).astype(np.float64)

        derivatives_descriptors_torch = torch.tensor(derivatives_descriptors)
        forces_energies_torch = torch.tensor(forces_energies)


        train_X = derivatives_descriptors_torch[0 : n_train, :, :]
        train_Y = forces_energies_torch[0 : n_train, :]


        test_X = derivatives_descriptors_torch[
            n_train : n_samples, :, :
        ]
        test_Y = forces_energies_torch[n_train : n_samples, :]

        test_energies_torch = test_Y[:, -1]
        test_forces_torch = test_Y[:, :-1].reshape(-1, 3, self.n_atoms)

        self.test_shape = test_Y.shape
        self.train_shape = train_Y.shape

        train_X = train_X[:, :, :].transpose(0, 1).flatten(0, 1)
        train_Y = train_Y[:, :].transpose(0, 1).flatten(0, 1)

        test_X = test_X[:, :, :].transpose(0, 1).flatten(0, 1)
        test_Y = test_Y[:, :].transpose(0, 1).flatten(0, 1)


        print("Train set")
        print("Shape: ", train_X.shape, train_Y.shape)
        print("Type: ", train_X.dtype, train_Y.dtype)
        print("Device: ", train_X.device, train_Y.device)

        print("\nTest set")
        print("Shape: ", test_X.shape, test_Y.shape)
        print("Type: ", test_X.dtype, test_Y.dtype)
        print("Device: ", test_X.device, test_Y.device)

        train_X = train_X.to(torch.float32)
        train_Y = train_Y.to(torch.float32)
        test_X = test_X.to(torch.float32)
        test_Y = test_Y.to(torch.float32)

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.train_X_e = train_X[-n_train:]
        self.train_Y_e = train_Y[-n_train:]
        self.test_X_e = test_X[-n_test:]
        self.test_Y_e = test_Y[-n_test:]


        return

    def randomly_rotate(self, traj, forces):


        energies_np = np.zeros(len(traj))
        forces_np = np.zeros((len(traj), len(traj[0]), 3))

        for i, mol in tqdm(enumerate(traj)):
            # mol.calc = XTB(method="GFN2-xTB")
            # a = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
            # a.positions = a.positions + 0.01*np.random.rand(n_atoms,3 )

            # a.positions[7] = a.positions[7] + np.array([0.0, 0.0, 1.5*np.random.rand(1)] ) + np.array([1.5,1.5,1.5])
            
            angles = 1000*np.random.rand(3)

            mol.euler_rotate(phi=angles[0], theta=angles[1], psi=angles[2], center=(0, 0, 0))
            # new_traj.append(mol)# hmm.. you can also rotate forces by creating dummy molecules and rotating them
            # energies_np[i] = mol.get_potential_energy()
            forces_np[i, :, :] = mol.get_forces()
            mol.calc = None

        # io.write("data/dump/rotated_traj.xyz", mol_traj, format="extxyz")

        return traj, forces_np


