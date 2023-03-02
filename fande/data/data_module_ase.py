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

from rascal.representations import SphericalInvariants

import ase
from ase import io

from tqdm import tqdm

from ase.units import Bohr, Hartree

from ase.visualize import view


from functools import lru_cache



class FandeDataModuleASE(LightningDataModule):
    def __init__(self, training_data, test_data, hparams, units='ev_angstrom'):
        super().__init__()
        self.hparams.update(hparams)

        self.traj_train = training_data['trajectory']
        self.energies_train = training_data['energies']
        self.forces_train = training_data['forces']

        # self.traj_train, self.forces_train = self.randomly_rotate(self.traj_train, self.forces_train)

        self.traj_test = test_data['trajectory']
        self.energies_test = test_data['energies']
        self.forces_test = test_data['forces']

        if units=='hartree_bohr':
            print('Converting from Hartree to eV, Hartree/Bohr to eV/Angstrom')
            self.energies_train = self.energies_train * Hartree
            self.energies_test = self.energies_test * Hartree
            self.forces_train = self.forces_train * Hartree / Bohr
            self.forces_test = self.forces_test * Hartree / Bohr

        self.train_DX = None
        self.train_F = None
        self.test_DX = None
        self.test_F = None

        self.train_X = None
        self.train_E = None
        self.test_X = None
        self.test_E = None

        self.normalizing_factor = np.max(self.energies_train) - np.min(self.energies_train)

        forces_train_norm = self.forces_train / self.normalizing_factor
        energies_train_norm = (self.energies_train - np.min(self.energies_train)) / self.normalizing_factor

        forces_test_norm = self.forces_test / self.normalizing_factor
        energies_test_norm = (self.energies_test - np.min(self.energies_train)) / self.normalizing_factor

        self.forces_train_norm = forces_train_norm
        self.forces_test_norm = forces_test_norm

        forces_train_norm = forces_train_norm.transpose(2,1,0).reshape(
            forces_train_norm.shape[0] * forces_train_norm.shape[1] * forces_train_norm.shape[2], -1
            ).astype(np.float64)

        forces_test_norm = forces_test_norm.transpose(2,1,0).reshape(
            forces_test_norm.shape[0] * forces_test_norm.shape[1] * forces_test_norm.shape[2], -1
            ).astype(np.float64)


        self.train_E = torch.tensor( energies_train_norm )
        self.test_E = torch.tensor( energies_test_norm )
        self.train_F = torch.tensor( forces_train_norm )
        self.test_F = torch.tensor( forces_test_norm )


        self.train_F = self.train_F[:, :].squeeze()
        self.test_F = self.test_F[:, :].squeeze()

        self.batch_size = 1_000_000



    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        print("hi")
        # called on every GPU

    def train_dataloader(self):
        # transforms = ...
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        # ...
        # transforms = ...
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        # ...
        # transforms = ...
        return DataLoader(self.test, batch_size=self.batch_size)


    def calculate_invariants_dscribe(self, soap_params):

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

        traj_train = self.traj_train
        traj_test = self.traj_test

        print(f"Total length of train traj is {len(traj_train)}")
        print("Starting SOAP calculation...")
        derivatives_train, descriptors_train = soap.derivatives(
            traj_train,
            positions=[positions] * len(traj_train),
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
            positions=[positions] * len(traj_test),
            n_jobs=10,
            # method="analytical"
        )
        print("SOAP calculation done!")
        derivatives_test = derivatives_test.squeeze()
        descriptors_test = descriptors_test.squeeze()


        self.train_X = torch.tensor(descriptors_train)      
        self.train_DX = torch.tensor(
                derivatives_train.transpose(2,1,0,3).reshape(
            derivatives_train.shape[0]*derivatives_train.shape[1]*derivatives_train.shape[2], -1
        ))

        self.test_X = torch.tensor(descriptors_test)      
        self.test_DX = torch.tensor(
                derivatives_test.transpose(2,1,0,3).reshape(
            derivatives_test.shape[0]*derivatives_test.shape[1]*derivatives_test.shape[2], -1
        ))

        print(derivatives_train.shape)
        print(descriptors_train.shape)

        # train_X = train_X.to(torch.float32)
        # train_Y = train_Y.to(torch.float32)
        # test_X = test_X.to(torch.float32)
        # test_Y = test_Y.to(torch.float32)


    # https://github.com/lab-cosmo/librascal/blob/4e576ae7b9d3740715ab1910def5e1a15ffd1268/tests/python/python_representation_calculator_test.py
    # https://github.com/lab-cosmo/librascal/blob/f45e6052e2ca5e3e5b62f1440a79b8da5eceec96/examples/needs_updating/Spherical_invariants_and_database_exploration.ipynb
    def calculate_invariants_librascal(
            self, 
            soap_params, 
            train_centers_positions=None, 
            train_derivatives_positions=None,
            test_centers_positions=None, 
            test_derivatives_positions=None,
            same_centers_derivatives=False):

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

        hypers = dict(soap_type="PowerSpectrum",
                    interaction_cutoff=3.5,
                    max_radial=6,
                    max_angular=6,
                    gaussian_sigma_constant=0.5,
                    gaussian_sigma_type="Constant",
                    cutoff_function_type="RadialScaling",
                    cutoff_smooth_width=0.5,
                    cutoff_function_parameters=
                            dict(
                                    rate=1,
                                    scale=3.5,
                                    exponent=4
                                ),
                    radial_basis="GTO",
                    normalize=True,
                    #   optimization=
                    #         dict(
                    #                 Spline=dict(
                    #                    accuracy=1.0e-05
                    #                 )
                    #             ),
                    compute_gradients=True
                    )
        
        self.soap_hypers = hypers
        
        traj_train = self.traj_train
        traj_test = self.traj_test

        for f in traj_train:
            f.wrap(eps=1e-18)

        for f in traj_test:
            f.wrap(eps=1e-18)

        n_atoms = len(traj_train[0])

        print(f"Total length of train traj is {len(traj_train)}")
        print(f"Total length of test traj is {len(traj_test)}")
        
        print("Calculating invariants on train trajectory with librascal...")
        soap_train = SphericalInvariants(**hypers)
        managers_train = soap_train.transform(traj_train)
        soap_array_train = managers_train.get_features(soap_train)
        soap_grad_array_train = managers_train.get_features_gradient(soap_train)  

        train_grad_info = managers_train.get_gradients_info()
        # get the information necessary to the computation of gradients. 
        # It has as many rows as dX_dr and each columns correspond to the 
        # index of the structure, the central atom, the neighbor atom and their atomic species.
        # get the derivatives of the representation w.r.t. the atomic positions
        # DX_train = soap_grad_array_train.reshape((grad_info_train.shape[0], 3, -1))
        
        #for now just subsampling the grad_array
        if train_centers_positions is not None and train_derivatives_positions is not None:
            print("Subsampling the gradients for selected positions...")
            a = train_grad_info[:,1]
            b = train_grad_info[:,2]

            if same_centers_derivatives:
                train_indices_sub = np.where(
                    np.in1d(a%n_atoms, train_centers_positions) & 
                    np.in1d(b%n_atoms, train_derivatives_positions) &
                    (a%n_atoms == b%n_atoms) )[0]
            else:
                train_indices_sub = np.where(
                np.in1d(a%n_atoms, train_centers_positions) & 
                np.in1d(b%n_atoms, train_derivatives_positions))[0]
                    
            forces_train_sub = np.zeros((train_grad_info[train_indices_sub].shape[0],3) )
            train_grad_info_sub = train_grad_info[train_indices_sub]

            for ind, gi in enumerate(train_grad_info_sub):
                forces_train_sub[ind] = self.forces_train[gi[0], gi[2]%n_atoms]

            forces_train_flat = forces_train_sub.flatten()

            train_indices_sub_3x = np.empty(( 3*train_indices_sub.size,), dtype=train_indices_sub.dtype)
            train_indices_sub_3x[0::3] = 3*train_indices_sub
            train_indices_sub_3x[1::3] = 3*train_indices_sub+1
            train_indices_sub_3x[2::3] = 3*train_indices_sub+2
            train_DX_np = soap_grad_array_train[train_indices_sub_3x]

        if train_centers_positions is not None and train_derivatives_positions is not None:
            self.train_DX = torch.tensor(train_DX_np, dtype=torch.float32)
            self.train_F = torch.tensor(forces_train_flat, dtype=torch.float32)

        del soap_train, managers_train, soap_grad_array_train, train_DX_np

        print("Calculating invariants on test trajectory with librascal...")
        soap_test = SphericalInvariants(**hypers)
        managers_test = soap_test.transform(traj_test)
        soap_array_test = managers_test.get_features(soap_test)
        soap_grad_array_test = managers_test.get_features_gradient(soap_test)           
        test_grad_info = managers_test.get_gradients_info()
        # DX_test = soap_grad_array_test.reshape((grad_info_test.shape[0], 3, -1))
        #for now just subsampling the grad_array
        if test_centers_positions is not None and test_derivatives_positions is not None:
            print("Subsampling the gradients for selected positions...")
            a = test_grad_info[:,1]
            b = test_grad_info[:,2]

            if same_centers_derivatives:
                test_indices_sub = np.where(
                    np.in1d(a%n_atoms, test_centers_positions) & 
                    np.in1d(b%n_atoms, test_derivatives_positions) &
                    (a%n_atoms == b%n_atoms) )[0]
            else:
                test_indices_sub = np.where(
                np.in1d(a%n_atoms, test_centers_positions) & 
                np.in1d(b%n_atoms, test_derivatives_positions))[0]
                    
            forces_test_sub = np.zeros((test_grad_info[test_indices_sub].shape[0],3) )
            test_grad_info_sub = test_grad_info[test_indices_sub]

            for ind, gi in enumerate(test_grad_info_sub):
                forces_test_sub[ind] = self.forces_test[gi[0], gi[2]%n_atoms]

            forces_test_flat = forces_test_sub.flatten()

            test_indices_sub_3x = np.empty(( 3*test_indices_sub.size,), dtype=test_indices_sub.dtype)
            test_indices_sub_3x[0::3] = 3*test_indices_sub
            test_indices_sub_3x[1::3] = 3*test_indices_sub+1
            test_indices_sub_3x[2::3] = 3*test_indices_sub+2
            test_DX_np = soap_grad_array_test[test_indices_sub_3x]



        if test_centers_positions is not None and test_derivatives_positions is not None:
            self.test_DX = torch.tensor(test_DX_np, dtype=torch.float32)
            self.test_F = torch.tensor(forces_test_flat, dtype=torch.float32)

        del soap_test, managers_test, soap_grad_array_test, test_DX_np



        return train_grad_info_sub, test_grad_info_sub



    def calculate_test_invariants_librascal(
            self, 
            test_centers_positions=None, 
            test_derivatives_positions=None,
            same_centers_derivatives=False):

        
        traj_test = self.traj_test

        for f in traj_test:
            f.wrap(eps=1e-18)

        n_atoms = len(traj_test[0])

        print(f"Total length of test traj is {len(traj_test)}")
        
        hypers = self.soap_hypers 

        print("Calculating invariants on test trajectory with librascal...")
        soap_test = SphericalInvariants(**hypers)
        managers_test = soap_test.transform(traj_test)
        # soap_array_test = managers_test.get_features(soap_test)
        soap_grad_array_test = managers_test.get_features_gradient(soap_test)           
        test_grad_info = managers_test.get_gradients_info()
        # DX_test = soap_grad_array_test.reshape((grad_info_test.shape[0], 3, -1))
        #for now just subsampling the grad_array
        if test_centers_positions is not None and test_derivatives_positions is not None:
            print("Subsampling the gradients for selected positions...")
            a = test_grad_info[:,1]
            b = test_grad_info[:,2]

            if same_centers_derivatives:
                test_indices_sub = np.where(
                    np.in1d(a%n_atoms, test_centers_positions) & 
                    np.in1d(b%n_atoms, test_derivatives_positions) &
                    (a%n_atoms == b%n_atoms) )[0]
            else:
                test_indices_sub = np.where(
                np.in1d(a%n_atoms, test_centers_positions) & 
                np.in1d(b%n_atoms, test_derivatives_positions))[0]
                    
            forces_test_sub = np.zeros((test_grad_info[test_indices_sub].shape[0],3) )
            test_grad_info_sub = test_grad_info[test_indices_sub]


            for ind, gi in enumerate(test_grad_info_sub):
                forces_test_sub[ind] = self.forces_test[gi[0], gi[2]%n_atoms]

            forces_test_flat = forces_test_sub.flatten()


            test_indices_sub_3x = np.empty(( 3*test_indices_sub.size,), dtype=test_indices_sub.dtype)
            test_indices_sub_3x[0::3] = 3*test_indices_sub
            test_indices_sub_3x[1::3] = 3*test_indices_sub+1
            test_indices_sub_3x[2::3] = 3*test_indices_sub+2
            test_DX_np = soap_grad_array_test[test_indices_sub_3x]



        if test_centers_positions is not None and test_derivatives_positions is not None:
            self.test_DX = torch.tensor(test_DX_np, dtype=torch.float32)
            self.test_F = torch.tensor(forces_test_flat, dtype=torch.float32)

        print(self.test_F.shape)

        del soap_test, managers_test, soap_grad_array_test, test_DX_np


        # need to also update forces...


        return test_grad_info_sub
    

    def calculate_snapshot_invariants_librascal(
            self,
            snapshot: ase.Atoms, 
            same_centers_derivatives=True):
        

        traj_test = [snapshot]

        for f in traj_test:
            f.wrap(eps=1e-18)

        n_atoms = len(traj_test[0])

        print(f"Calculating full invariants for a single snapshot...")      
        hypers = self.soap_hypers 
        soap_test = SphericalInvariants(**hypers)
        managers_test = soap_test.transform(traj_test)
        soap_grad_array_test = managers_test.get_features_gradient(soap_test)           
        test_grad_info = managers_test.get_gradients_info()
        if same_centers_derivatives:
            print("Subsampling the gradients for selected positions...")
            a = test_grad_info[:,1]
            b = test_grad_info[:,2]
            test_indices_sub = np.where((a%n_atoms == b%n_atoms))[0]
            
            test_indices_sub_3x = np.empty(( 3*test_indices_sub.size,), dtype=test_indices_sub.dtype)
            test_indices_sub_3x[0::3] = 3*test_indices_sub
            test_indices_sub_3x[1::3] = 3*test_indices_sub+1
            test_indices_sub_3x[2::3] = 3*test_indices_sub+2
            test_DX_np = soap_grad_array_test[test_indices_sub_3x]

            self.snap_DX = torch.tensor(test_DX_np, dtype=torch.float32)

        else:
            raise NotImplementedError

        del soap_test, managers_test, soap_grad_array_test, test_DX_np

        return test_grad_info



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

    def randomly_rotate(self, traj_initial, forces):
        print("Randomly rotating training configuration and forces...")

        traj = traj_initial.copy()

        energies_np = np.zeros(len(traj))
        forces_np = np.zeros((len(traj), len(traj[0]), 3))

        mol_forces = traj[0].copy()
        forces_rotated = np.empty_like(forces)


        for i, mol in tqdm(enumerate(traj)):

            angles = 1000*np.random.rand(3)
            mol.euler_rotate(phi=angles[0], theta=angles[1], psi=angles[2], center=(0, 0, 0))
            
            # hmm.. you can also rotate forces by creating dummy molecules and rotating them
            mol_forces.positions = forces[i]
            mol_forces.euler_rotate(phi=angles[0], theta=angles[1], psi=angles[2], center=(0, 0, 0))
            forces_rotated[i] = mol_forces.positions


        # io.write("data/dump/rotated_traj.xyz", mol_traj, format="extxyz")

        return traj, forces_rotated


    def view_traj(self):

        print("Train+test trajectories:")
        print(f"First {len(self.traj_train)} frames is traning traj, remaining is test traj")

        view(self.traj_train)
        view(self.traj_test)
             
        return


