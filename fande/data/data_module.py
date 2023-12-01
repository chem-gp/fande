# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#datamodules-recommended
# from tkinter import NO
from torch.utils.data import DataLoader
import os
import subprocess
# from datetime import datetime

from typing import Optional
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, TensorDataset, random_split

import torch
import numpy as np


# from fande.data import FastLoader
# from fande.utils import get_vectors_e, get_vectors_f
# from dscribe.descriptors import SOAP

from rascal.representations import SphericalInvariants

import ase
from ase import io
from tqdm import tqdm

# from ase.units import Bohr, Hartree
# from ase.visualize import view
# from functools import lru_cache

import math

from fande import logger



class FandeDataModule(LightningDataModule):
    def __init__(
        self, 
        training_data=None, 
        hparams=None
        ):

        super().__init__()

        if hparams is None:
            hparams = {}
        self.hparams.update(hparams)

        if training_data is not None:
            self.traj_train = training_data['trajectory']
            self.forces_train = training_data['forces']
            self.energies_train = training_data['energies']
            self.trajectory_energies_train = training_data['trajectory_energies']
        

        self.test_data = None
        # self.traj_train, self.forces_train = self.randomly_rotate(self.traj_train, self.forces_train)

        self.test_DX = None
        self.test_F = None

        self.train_DX = None
        self.train_F = None

        self.train_X = None
        self.train_E = None
        self.test_X = None
        self.test_E = None

        self.atomic_groups_train = None
        self.centers_positions_train = None
        self.derivatives_positions_train = None
        self.n_atoms = None

        self.train_indices = None

        self.batch_size = 100_000




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


    # https://github.com/lab-cosmo/librascal/blob/4e576ae7b9d3740715ab1910def5e1a15ffd1268/tests/python/python_representation_calculator_test.py
    # https://github.com/lab-cosmo/librascal/blob/f45e6052e2ca5e3e5b62f1440a79b8da5eceec96/examples/needs_updating/Spherical_invariants_and_database_exploration.ipynb
    def calculate_invariants_librascal_no_derivatives(
            self,
            trajectory, 
            soap_params,
            frames_per_batch=1): 
        """
        Calculate SOAP invariants without derivatives using librascal.

        """

        # species= soap_params['species']
        # periodic= soap_params['periodic']
        interaction_cutoff = soap_params['interaction_cutoff']
        gaussian_sigma_constant= soap_params['gaussian_sigma_constant']
        max_radial= soap_params['max_radial']
        max_angular= soap_params['max_angular']
        cutoff_smooth_width = soap_params['cutoff_smooth_width']
        # average= soap_params['average']
        # crossover= soap_params['crossover']
        # dtype= soap_params['dtype']
        # sparse= soap_params['sparse']
        # positions = soap_params['positions']

        hypers = dict(soap_type="PowerSpectrum",
                    interaction_cutoff=interaction_cutoff,
                    max_radial=max_radial,
                    max_angular=max_angular,
                    gaussian_sigma_constant=gaussian_sigma_constant,
                    gaussian_sigma_type="Constant",
                    cutoff_function_type="RadialScaling",
                    cutoff_smooth_width=cutoff_smooth_width, # 0.1 is way better than 0.5
                    cutoff_function_parameters=
                            dict(
                                    rate=1,
                                    scale=3.5,
                                    exponent=4
                                ),
                    radial_basis="GTO",
                    normalize=True, # setting False makes model untrainable
                    #   optimization=
                    #         dict(
                    #                 Spline=dict(
                    #                    accuracy=1.0e-05
                    #                 )
                    #             ),
                    compute_gradients=True,
                    expansion_by_species_method='structure wise'
                    )
    
        self.soap_hypers_energy = hypers

        traj = trajectory
        for f in traj:
            f.wrap(eps=1e-18)

        n_atoms = len(traj[0])

      
        frames_batches = self.prepare_batches(traj, forces=None, energies=None, frames_per_batch=frames_per_batch)      

        print(f"Total length of traj is {len(traj)}")
        print(f"Total number of batches {len(frames_batches)}")       
        print("Calculating invariants on trajectory with librascal...")
   
        soap = SphericalInvariants(**hypers)

        X_np_batched = []  
        # F_np_batched = [[] * len(frames_batches) for i in range(n_atomic_groups)]
        # grad_info_sub_batched = [[] * len(frames_batches) for i in range(n_atomic_groups)]

        for ind_b, batch in enumerate(tqdm(frames_batches)):
            traj_b = batch['traj']
            # forces_b = batch['forces']
            # energies_b = batch['energies']

            managers = soap.transform(traj_b)
            soap_array = managers.get_features(soap)
            X_np_batched.append(soap_array)

        X = torch.tensor(X_np_batched,dtype=torch.float32)


        return X


    # https://github.com/lab-cosmo/librascal/blob/4e576ae7b9d3740715ab1910def5e1a15ffd1268/tests/python/python_representation_calculator_test.py
    # https://github.com/lab-cosmo/librascal/blob/f45e6052e2ca5e3e5b62f1440a79b8da5eceec96/examples/needs_updating/Spherical_invariants_and_database_exploration.ipynb
    def calculate_invariants_librascal(
            self, 
            soap_params=None,
            atomic_groups = None, 
            centers_positions=None, 
            derivatives_positions=None,
            same_centers_derivatives=False,
            frames_per_batch=1,
            calculation_context=None, # train/test/production contexts
            trajectory=None): 
        """
        Calculate SOAP invariants with derivatives using librascal. If `trajectory` is not specified, the invariants are calculated for the training or test set of frames (`self.traj_train` or `self.traj_test`).

        Parameters:
        -----------
        soap_params: dict
            Dictionary containing the parameters for the SOAP descriptors
        atomic_groups: list
            List of atomic groups to train separate GP models for
        centers_positions: list
            List of positions of the atomic centers
        derivatives_positions: list
            List of indices of the atoms to calculate the derivatives with respect to. On these atoms the forces are evaluated.
        same_centers_derivatives: bool
            If True, the same atomic centers are used for the derivatives as for the descriptors.
        frames_per_batch: int
            Number of frames per batch when calculating invariants. Extremely useful when memory is limited.
        calculation_context: str, ['train', 'test', 'production']
            Whether to calculate the invariants for the training/test set of frames (stored within`self.traj_train` or `self.traj_test`) or for on-the-fly md runs.
        
        Returns:
        --------
        X: torch.tensor
            Tensor containing the SOAP descriptors for the set of frames split by atomic groups specified in `atomic_groups`.
        DX: list(torch.tensor)
            list of torch tensors containing the derivatives of the SOAP descriptors for each atomic group specified in `atomic_groups`.
        
        only for calculation_context='production':
        F: torch.tensor
            Tensor containing the forces for the set of frames split by atomic groups specified in `atomic_groups`.
        """

        if calculation_context is None:
            raise ValueError('Calculation_context: "train" or "test" or "production" must be specified')

        if  calculation_context == 'train':

            if soap_params is None:
                raise ValueError('soap_params must be specified when when training!')

            # species= soap_params['species']
            # periodic= soap_params['periodic']
            interaction_cutoff = soap_params['interaction_cutoff']
            gaussian_sigma_constant= soap_params['gaussian_sigma_constant']
            max_radial= soap_params['max_radial']
            max_angular= soap_params['max_angular']
            cutoff_smooth_width = soap_params['cutoff_smooth_width']
            # average= soap_params['average']
            # crossover= soap_params['crossover']
            # dtype= soap_params['dtype']
            # sparse= soap_params['sparse']
            # positions = soap_params['positions']

            hypers = dict(soap_type="PowerSpectrum",
                        interaction_cutoff=interaction_cutoff,
                        max_radial=max_radial,
                        max_angular=max_angular,
                        gaussian_sigma_constant=gaussian_sigma_constant,
                        gaussian_sigma_type="Constant",
                        cutoff_function_type="RadialScaling",
                        cutoff_smooth_width=cutoff_smooth_width, # 0.1 is way better than 0.5
                        cutoff_function_parameters=
                                dict(
                                        rate=1,
                                        scale=3.5,
                                        exponent=4
                                    ),
                        radial_basis="GTO",
                        normalize=True, # setting False makes model untrainable
                        #   optimization=
                        #         dict(
                        #                 Spline=dict(
                        #                    accuracy=1.0e-05
                        #                 )
                        #             ),
                        compute_gradients=True,
                        expansion_by_species_method='structure wise'
                        )
        
            self.soap_hypers = hypers

        else:
            hypers = self.soap_hypers

        logger.info("Setting context for descriptors calculation to {}".format(calculation_context))

        if calculation_context == "train":
            traj = self.traj_train
            forces = self.forces_train
            self.atomic_groups_train = atomic_groups
            self.centers_positions_train = centers_positions
            self.derivatives_positions_train = derivatives_positions
        elif calculation_context == "test":
            self.traj_test = self.test_data['trajectory']
            self.energies_test = self.test_data['energies']
            self.forces_test = self.test_data['forces']
            traj = self.traj_test
            forces = self.forces_test

        if trajectory is not None and calculation_context == "production":
            traj = trajectory
            forces = np.zeros((len(traj), len(traj[0]), 3))
            atomic_groups = self.atomic_groups_train
            centers_positions = self.centers_positions_train
            derivatives_positions = self.derivatives_positions_train
            # print("Production mode")
            # raise NotImplementedError("Not implemented yet for trajectory input")
   
        for f in traj:
            f.wrap(eps=1e-18)

        n_atoms = len(traj[0])

        tqdm_ = lambda x: x
        if calculation_context == "train":
            self.n_atoms = n_atoms
            tqdm_ = tqdm

        if atomic_groups == 'all':
            atomic_groups = [list(range(n_atoms))]


        n_atomic_groups = len(atomic_groups)


        frames_batches = self.prepare_batches(traj, forces, frames_per_batch=frames_per_batch)

        

        # print(f"Total length of traj is {len(traj)}")
        # print(f"Total number of batches {len(frames_batches)}")       
        # print("Calculating invariants on trajectory with librascal...")


   
        soap = SphericalInvariants(**hypers)

        DX_np_batched = [[] * len(frames_batches) for i in range(n_atomic_groups)]  
        F_np_batched = [[] * len(frames_batches) for i in range(n_atomic_groups)]
        grad_info_sub_batched = [[] * len(frames_batches) for i in range(n_atomic_groups)]

        X_np_batched = []

        for ind_b, batch in enumerate(tqdm_(frames_batches)):
            traj_b = batch['traj']
            forces_b = batch['forces']

            managers = soap.transform(traj_b)
            soap_array = managers.get_features(soap)
            X_np_batched.append(soap_array)

            soap_grad_array = managers.get_features_gradient(soap)
            grad_info = managers.get_gradients_info()
            # get the information necessary to the computation of gradients. 
            # It has as many rows as dX_dr and each columns correspond to the 
            # index of the structure, the central atom, the neighbor atom and their atomic species.
            # get the derivatives of the representation w.r.t. the atomic positions
            # DX_train = soap_grad_array_train.reshape((grad_info_train.shape[0], 3, -1))
            # print("rascal calculations done...")

            #for now just subsampling the grad_array: this part is probably much cheaper to evaluate calculation of invariants
            # if centers_positions is not None and derivatives_positions is not None:
                # print("Subsampling the gradients for selected positions...")
            a = grad_info[:,1]
            b = grad_info[:,2]
            for ind_ag, ag in enumerate(atomic_groups):
                # a very important step where we select the relevant indices:
                # for now we select the coinciding centers and derivatives atoms that also belong to the atomic group
                indices_sub = np.where(
                    np.in1d(a%n_atoms, centers_positions) & 
                    np.in1d(b%n_atoms, derivatives_positions) &
                    (a%n_atoms == b%n_atoms) &
                    np.in1d(b%n_atoms, ag) )[0]
                
                ### DEBUG
                # print(grad_info.shape)
                # indices_sub_ = np.where(
                #     np.in1d(a%n_atoms, centers_positions) & 
                #     np.in1d(b%n_atoms, derivatives_positions) &
                #     (a%n_atoms == b%n_atoms) &
                #     np.in1d(b%n_atoms, ag) 
                #     )[0]
                # # print(a[0:20]%n_atoms)
                # # print(b[0:20]%n_atoms)
                # print(np.count_nonzero((a%n_atoms == b%n_atoms)))
                # # print("Centers", centers_positions)
                # print("Derivatives", derivatives_positions)
                # # print("Atomic group", ag)
                # print("Indices sub", indices_sub_)
                ### DEBUG

                forces_sub = np.zeros((grad_info[indices_sub].shape[0],3) )
                grad_info_sub = grad_info[indices_sub]

                for ind, gi in enumerate(grad_info_sub):
                    forces_sub[ind] = forces_b[gi[0], gi[2]%n_atoms]

                forces_train_flat = forces_sub.flatten()

                indices_sub_3x = np.empty(( 3*indices_sub.size,), dtype=indices_sub.dtype)
                indices_sub_3x[0::3] = 3*indices_sub
                indices_sub_3x[1::3] = 3*indices_sub+1
                indices_sub_3x[2::3] = 3*indices_sub+2
                DX_np = soap_grad_array[indices_sub_3x]

                DX_np_batched[ind_ag].append(DX_np)
                F_np_batched[ind_ag].append(forces_train_flat)
                grad_info_sub_batched[ind_ag].append(grad_info_sub)

                # print(a)
                # print("positions ", centers_positions, derivatives_positions)
                # print()

                
                if not same_centers_derivatives:
                    raise NotImplementedError("Different centers and derivatives not implemented yet")
                    indices_sub = np.where(
                    np.in1d(a%n_atoms, train_centers_positions) & 
                    np.in1d(b%n_atoms, train_derivatives_positions))[0]


        DX_np_grouped = []
        F_np_grouped = []
        for i in range(n_atomic_groups):
            DX_np_grouped.append( np.concatenate( DX_np_batched[i][:]) )
            F_np_grouped.append( np.concatenate( F_np_batched[i][:]) )
        
        # we cannot create a single tensor for all groups, because the number of environments is different for each group
        DX = [torch.tensor(DX_np_grouped[i], dtype=torch.float32) for i in range(n_atomic_groups)]
        F = [torch.tensor(F_np_grouped[i], dtype=torch.float32) for i in range(n_atomic_groups)]
        
        X = torch.tensor(X_np_batched,dtype=torch.float32)

        if calculation_context == "train":
            self.train_DX = DX
            self.train_F = F
            self.train_X = X
        elif calculation_context == "test":
            self.test_DX = DX
            self.test_F = F
            self.test_X = X


        if calculation_context == 'production':
            return X, DX
        else:
            return



    def prepare_batches(
            self, 
            traj, 
            forces=None,
            energies=None, 
            frames_per_batch=10):
        
        n_frames = len(traj)
        n_batches = math.ceil(n_frames/frames_per_batch)

        # print(f"Total number of frames is {n_frames}")
        # print(f"Total number of batches is {n_batches}")

        batches = []
        for i in range(n_batches):
            batch = {}
            batch["traj"] = traj[i*frames_per_batch:(i+1)*frames_per_batch]
            if forces is not None:
                batch["forces"] = forces[i*frames_per_batch:(i+1)*frames_per_batch]
            if energies is not None:
                batch["energies"] = energies[i*frames_per_batch:(i+1)*frames_per_batch]
            batches.append(batch)

        return batches
        


    def calculate_snapshot_invariants_librascal(
            self,
            snapshot: ase.Atoms, 
            same_centers_derivatives=True):

        traj = [snapshot]
        snap_X, snap_DX = self.calculate_invariants_librascal(
            trajectory=traj, 
            same_centers_derivatives=same_centers_derivatives, 
            calculation_context="production")

        return snap_X, snap_DX


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


    def prepare_train_data_loaders(
            self, 
            total_samples_per_group, 
            high_force_samples_per_group):
        """
        Prepare data loaders for training on FORCES. Currently data loaders are created within the FandeDataModule class and then passed to the group model classes.

        Parameters
        ----------
        samples_per_group : list of int
            Number of training random samples to use for each group. If None, all samples are used.
        
        high_force_samples_per_group : list of int
            Number of high force samples to be included for each group. `high_force_samples_per_group` of the tighest (positive) and lowest (negative) forces are selected.

        Returns
        -------
        train_data_loaders : list of torch.utils.data.DataLoader
            List of data loaders for training for each atomic group model.
        """

        train_indices = []

        train_data_loaders = []
        for idx, model in enumerate(self.atomic_groups_train):

            total_training_random_samples = total_samples_per_group[idx]
            high_force_samples = high_force_samples_per_group[idx]
            random_samples = total_training_random_samples - high_force_samples
            total_train_samples = self.train_F[idx].shape[0]


            if total_training_random_samples is None or total_training_random_samples == 'all' or total_train_samples < total_training_random_samples:
                ind_slice = np.sort( np.arange(0, self.train_F[idx].shape[0]) )
                indices = ind_slice
                print(f"Taking ALL {total_train_samples} samples for group {idx}")
            else:
                indices_high_force = torch.concat( 
                    (torch.topk(self.train_F[idx], high_force_samples//2, largest=True)[1],  
                    torch.topk(self.train_F[idx], high_force_samples//2, largest=False)[1]) ).cpu().numpy()

                indices_without_high_force = np.setdiff1d(  np.arange(0, self.train_F[idx].shape[0]), indices_high_force    )

                ind_slice = np.sort(  np.random.choice(indices_without_high_force, random_samples, replace=False) )
 
                indices = np.concatenate((ind_slice, indices_high_force))
                indices = np.unique(indices)

            train_indices.append(indices.tolist())

            train_dataset = TensorDataset(self.train_DX[idx][indices], self.train_F[idx][indices])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
            train_data_loaders.append(train_loader)

            print("Dataloader for group {} created".format(idx))
            print("Number of samples in dataloader: {}".format(len(train_dataset)) )
        
        self.train_indices = train_indices      

        return train_data_loaders


    def prepare_train_data_loaders_energy(self, trajectory=None, train_energies=None, energy_soap_hypers=None):
        """
        Prepare data loaders for training on ENERGIES. Currently data loaders are created within the FandeDataModule class and then passed to the energy model.
        """
        
        if trajectory is not None:
            trajectory = trajectory
        else:
            trajectory = self.traj_train

        if energy_soap_hypers is not None:
            energy_soap_hypers = energy_soap_hypers
        else:
            energy_soap_hypers = self.soap_hypers_energy

        if train_energies is None:
            energies = self.trajectory_energies_train
        else:
            energies = train_energies

        train_X = self.calculate_invariants_librascal_no_derivatives(trajectory, energy_soap_hypers)
        train_X = train_X.sum(axis=1)
        train_E = torch.tensor(energies, dtype=torch.float32)

        train_dataset = TensorDataset(train_X, train_E)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        return train_loader


    def dataloaders_from_trajectory(
            self,
            trajectory_energy,
            trajectory_forces,
            energies = None,
            forces = None,
            atomic_groups = None,
            centers_positions = None,
            derivatives_positions = None,
            energy_soap_hypers = None,
            forces_soap_hypers = None,
            total_forces_samples_per_group = 0,
            high_force_samples_per_group = 0,
            ):

            if atomic_groups is None:
                atomic_groups = self.atomic_groups_train

            if energy_soap_hypers is None:
                energy_soap_hypers = self.soap_hypers_energy

            if forces_soap_hypers is None:
                forces_soap_hypers = self.soap_hypers

            if energies is None:
                energies = [s.calc.get_potential_energy() for s in trajectory_energy]
                self.emax = np.max(energies)
                self.emin = np.min(energies)
                energies = (energies - self.emin)/(self.emax - self.emin)
                energies = np.array(energies)

            if forces is None:
                forces = [s.calc.get_forces() for s in trajectory_forces]
                forces = np.array(forces)

            self.traj_train = trajectory_forces
            self.forces_train = forces
            self.atomic_groups_train = atomic_groups
            self.centers_positions_train = centers_positions
            self.derivatives_positions_train = derivatives_positions


            X_energy = self.calculate_invariants_librascal_no_derivatives(trajectory_energy, energy_soap_hypers)
            # X_forces, DX_forces = self.calculate_invariants_librascal(trajectory_forces, forces_soap_hypers, calculation_context="train")
            self.calculate_invariants_librascal(
                forces_soap_hypers,
                atomic_groups = atomic_groups,
                centers_positions = centers_positions, 
                derivatives_positions = derivatives_positions,
                same_centers_derivatives=True,
                frames_per_batch=1,
                calculation_context="train")


            X_forces = self.train_X
            DX_forces = self.train_DX
            F_forces = self.train_F


            X_energy = X_energy.sum(axis=1)
            train_E = torch.tensor(energies, dtype=torch.float32)
            train_dataset_energy = TensorDataset(X_energy, train_E)
            train_loader_energy = DataLoader(train_dataset_energy, batch_size=self.batch_size)

            #################### Forces sampling: ####################

            train_indices = []

            total_samples_per_group = total_forces_samples_per_group
            high_force_samples_per_group = high_force_samples_per_group

            train_data_loaders_forces = []
            for idx in range(len(self.atomic_groups_train)):

                total_training_random_samples = total_samples_per_group[idx]
                high_force_samples = high_force_samples_per_group[idx]
                random_samples = total_training_random_samples - high_force_samples
                total_train_samples = self.train_F[idx].shape[0]


                if total_training_random_samples is None or total_training_random_samples == 'all' or total_train_samples < total_training_random_samples:
                    ind_slice = np.sort( np.arange(0, self.train_F[idx].shape[0]) )
                    indices = ind_slice
                    print(f"Taking ALL {total_train_samples} samples for group {idx}")
                else:
                    indices_high_force = torch.concat( 
                        (torch.topk(self.train_F[idx], high_force_samples//2, largest=True)[1],  
                        torch.topk(self.train_F[idx], high_force_samples//2, largest=False)[1]) ).cpu().numpy()

                    indices_without_high_force = np.setdiff1d(  np.arange(0, self.train_F[idx].shape[0]), indices_high_force    )

                    ind_slice = np.sort(  np.random.choice(indices_without_high_force, random_samples, replace=False) )
    
                    indices = np.concatenate((ind_slice, indices_high_force))
                    indices = np.unique(indices)

                train_indices.append(indices.tolist())

                train_dataset = TensorDataset(self.train_DX[idx][indices], self.train_F[idx][indices])
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                train_data_loaders_forces.append(train_loader)

                print("Dataloader for group {} created".format(idx))
                print("Number of samples in dataloader: {}".format(len(train_dataset)) )
            
            self.train_indices = train_indices  


            ##################################################

            return train_loader_energy, train_data_loaders_forces