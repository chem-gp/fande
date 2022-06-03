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
import sparse

import wandb

from fande.data import FastLoader

from fande.utils import get_vectors_e, get_vectors_f

from dscribe.descriptors import SOAP

from ase import io
from xtb.ase.calculator import XTB

from tqdm import tqdm


from functools import lru_cache




def create_dump_directory():
    # import string
    # import random
    # printing lowercase
    # letters = string.ascii_lowercase
    # rstr = "_" + "".join(random.choice(letters) for i in range(3))

    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S.%f")

    dirName = "data/dump/" + date_time
    # Create target directory & all intermediate directories if don't exists
    os.makedirs(dirName)

    dump_dir = dirName + "/"

    dump_dir = os.path.abspath(dump_dir)

    print("Created directory to store computational data:")
    print(dump_dir)

    print("All the artifacts are saved there.")
    os.chdir(dump_dir)

    return dump_dir


# import subprocess

# out = subprocess.run(['zenodo_get', '6333600'], capture_output=True)
## https://zenodo.org/record/6333600
## 2-propanol
# print( 'exit status:', out.returncode )
# print( 'stdout:', out.stdout.decode() )
# print( 'stderr:', out.stderr.decode() )
# Maybe use zenodopy
# https://github.com/lgloege/zenodopy


class FandeDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.data_dir = hparams["root_dir"] + "/data/dump"
        self.train_dims = None
        self.vocab_size = 0

        self.train = 1
        self.val = 2
        self.test = 3

        self.n_atoms = None
        self.n_molecules = None
        self.n_train_structures = None
        self.n_test_structures = None
        self.soap_size = None

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.r_test = None

        self.test_shape = None
        self.train_shape = None

        self.forces_energies = None # all stuff flattened

        self.forces = None # not normalized non-flattened numpy array
        self.energies = None # not normalized

        self.forces_norm = None
        self.energies_norm = None

        self.normalizing_const = None
        self.normalizing_energy_shift = None

        self.mol_traj = None

        self.save_hyperparameters()

    def prepare_data(self, dataset_id="6333600"):

        print("Downloading dataset from https://zenodo.org/record/" + dataset_id)
        os.makedirs(self.data_dir, exist_ok=True)
        os.chdir(self.data_dir)
        out = subprocess.run(["zenodo_get", "6333600"], capture_output=True)
        print("exit status:", out.returncode)
        print("stdout:", out.stdout.decode())
        print("stderr:", out.stderr.decode())
        # called only on 1 GPU
        # download_dataset()
        # tokenize()
        # build_vocab()

        # Or load data from the torch format
        # torch.save(derivatives_descriptors_torch, "data/dump/derivatives_descriptors_torch.pt")
        # torch.save(forces_energies_torch, "data/dump/forces_energies_torch.pt")

        # torch.save({
        #       "derivatives_descriptors_torch": derivatives_descriptors_torch,
        #       "forces_energies_torch": forces_energies_torch
        #       }, "data/dump/dataset_torch.pt")

        # dataset_torch = torch.load("data/dump/dataset_torch.pt")

        return

    def setup(self, stage: Optional[str] = None):
        print("hi")
        # called on every GPU
        # vocab = load_vocab()
        # self.vocab_size = len(vocab)

        # self.train, self.val, self.test = load_datasets()
        # self.train_dims = self.train.next_batch.size()

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

    def calculate_invariants(self):

        fl = FastLoader(self.hparams)
        derivatives_descriptors, forces_energies, forces_np, energies_np, mol, mol_traj = fl.load_and_caculate()

        self.forces = forces_np
        self.energies = energies_np

        forces = forces_np / (np.max(energies_np) - np.min(energies_np))
        energies = (energies_np - np.min(energies_np)) / (
            np.max(energies_np) - np.min(energies_np)
        )

        self.forces_norm = forces
        self.energies_norm = energies

        self.normalizing_const = np.max(energies_np) - np.min(energies_np)
        self.normalizing_shift = np.min(energies_np)

        self.mol_traj = mol_traj
        self.forces_energies = forces_energies

        derivatives_descriptors_torch = torch.tensor(derivatives_descriptors)
        forces_energies_torch = torch.tensor(forces_energies)

        # derivatives_descriptors_torch = dataset_torch['derivatives_descriptors_torch']
        # forces_energies_torch = dataset_torch['forces_energies_torch']

        r_train = 0.8
        r_test = 1.0 - r_train
        self.r_test = r_test
        n_samples = forces_energies_torch.shape[0]

        self.n_train_structures = int(r_train * n_samples)
        self.n_test_structures = int(r_test * n_samples)

        train_X = derivatives_descriptors_torch[0 : int(r_train * n_samples), :, :]
        train_Y = forces_energies_torch[0 : int(r_train * n_samples), :]

        trrrrx = train_X
        trrrry = train_Y

        test_X = derivatives_descriptors_torch[
            int(r_train * n_samples) : n_samples, :, :
        ]
        test_Y = forces_energies_torch[int(r_train * n_samples) : n_samples, :]

        test_energies_torch = test_Y[:, -1]
        test_forces_torch = test_Y[:, :-1].reshape(-1, 3, len(mol))

        self.test_shape = test_Y.shape
        self.train_shape = train_Y.shape

        train_X = train_X[:, :, :].transpose(0, 1).flatten(0, 1)
        train_Y = train_Y[:, :].transpose(0, 1).flatten(0, 1)

        test_X = test_X[:, :, :].transpose(0, 1).flatten(0, 1)
        test_Y = test_Y[:, :].transpose(0, 1).flatten(0, 1)

        # train_X = train_X[0:1000]
        # train_Y = train_Y[0:1000]

        print("Train set:")
        print(train_X.shape, train_Y.shape)
        print(train_X.dtype, train_Y.dtype)
        print(train_X.device, train_Y.device)

        print("\nTest set:")
        print(test_X.shape, test_Y.shape)
        print(test_X.dtype, test_Y.dtype)
        print(test_X.device, test_Y.device)

        train_X = train_X.to(torch.float32)
        train_Y = train_Y.to(torch.float32)
        test_X = test_X.to(torch.float32)
        test_Y = test_Y.to(torch.float32)

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

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


    def get_vectors(self):

        raise NotImplementedError

        n_atoms = 12
        n_molecules = 1600
        soap_size = 720

        from fande.utils import get_vectors_e, get_vectors_f

        # transforming back and forth between tensor shapes
        x, y = get_vectors_e(train_X, train_Y, n_molecules, n_atoms)
        Dx, Dy = get_vectors_f(train_X, train_Y, n_molecules, n_atoms)

        print(torch.equal(x, trrrrx[:, -1, :]))
        print(torch.equal(y, trrrry[:, -1]))
        print(torch.equal(Dx, trrrrx[:, :-1, :]))
        print(torch.equal(Dy, trrrry[:-1, :]))

    def calculate_MD(self):
        # to compute md from scratch
        # Calculate MD, forces and energies for some example molecule
        # 100 ps MD can run for ~15 min for simple 12 atom molecule
        raise NotImplementedError

        from fande.data.compute_md import Compute_MD
        import numpy as np
        from ase.build import molecule

        mol = molecule("CH3CH2OCH3")

        md_calc_xtb = Compute_MD(mol, hparams)

        md_traj, energies_np, forces_np = md_calc_xtb.run_xTB_md()
        forces = forces_np / (np.max(energies_np) - np.min(energies_np))
        energies = (energies_np - np.min(energies_np)) / (
            np.max(energies_np) - np.min(energies_np)
        )


    def dump_data_to_file(self, filename):
        torch.save({
            'train_x': self.train_X, 
            'train_y': self.train_Y,
            'test_x': self.test_X,
            'test_y': self.test_Y}, filename)

        print("All data saved to ", os.path.abspath(filename) )

        return

    def load_precomputed_data(self, torch_filename):
        
        os.chdir(self.hparams['root_dir'])

        full_path = os.path.join(self.hparams['root_dir'], torch_filename)
        mol_dataset = torch.load(full_path)

        # mol_dataset2 = torch.load('data/dump/dataset_15000_17000_torch.pt')


        train_x = mol_dataset['train_x'].contiguous()
        train_y = mol_dataset['train_y'].contiguous()
        test_x = mol_dataset['test_x'].contiguous()
        test_y = mol_dataset['test_y'].contiguous()

        # self.train_X_forces = train_x[-1600:]
        # self.train_Y_forces = train_y[-1600:]
        # self.test_X_forces = test_x[-400:]
        # self.test_Y_forces = test_y[-400:]

        self.train_X = train_x
        self.train_Y = train_y
        self.test_X = test_x
        self.test_Y = test_y

        self.train_X_e = train_x[-1600:]
        self.train_Y_e = train_y[-1600:]
        self.test_X_e = test_x[-400:]
        self.test_Y_e = test_y[-400:]

        # train_x2 = mol_dataset2['train_x'].contiguous()
        # train_y2 = mol_dataset2['train_y'].contiguous()
        # test_x2 = mol_dataset2['test_x'].contiguous()
        # test_y2 = mol_dataset2['test_y'].contiguous()

        # train_x2 = train_x2[-1600:]
        # train_y2 = train_y2[-1600:]
        # test_x2 = test_x2[-400:]
        # test_y2 = test_y2[-400:]

        return


    def prepare_torch_dataset(descriptors, derivatives, energies, forces):

        r_test = 0.2
        r_train = 1 - r_train

        energies_train = energies[0:1600]
        forces_train = forces[0:1600]

        self.normalizing_const = np.max(energies_train) - np.min(energies_train)
        self.normalizing_shift = np.min(energies_train)

        forces = forces_np / self.normalizing_const
        energies = ( energies_train - self.normalizing_shift ) / self.normalizing_const
        self.forces_norm = forces
        self.energies_norm = energies


        energies_train = energies[0:1600]
        forces_train = forces[0:1600]
        energies_test = energies[1600:2000]
        forces_test = energies[1600:2000]


###
        derivatives_flattened = derivatives.reshape(
            derivatives.shape[0], derivatives.shape[1], -1, derivatives.shape[-1]
        )
        descriptors_expanded = np.expand_dims(descriptors, 2)

        derivatives_descriptors = np.concatenate(
            (derivatives_flattened, descriptors_expanded), axis=2
        )
        print(soap.get_number_of_features())

        derivatives_descriptors = derivatives_descriptors.squeeze().astype(np.float64)
        forces_energies = np.concatenate(
            (forces.reshape(forces.shape[0], -1), energies[:, None]), axis=1
        ).astype(np.float64)

###



        derivatives_descriptors_torch = torch.tensor(derivatives_descriptors)
        forces_energies_torch = torch.tensor(forces_energies)

        n_samples = energies.shape[0]

        self.n_train_structures = energies_train.shape[0]
        self.n_test_structures = energies_test.shape[0]




        train_X = derivatives_descriptors_torch[0 : int(r_train * n_samples), :, :]
        train_Y = forces_energies_torch[0 : int(r_train * n_samples), :]


        test_X = derivatives_descriptors_torch[
            int(r_train * n_samples) : n_samples, :, :
        ]
        test_Y = forces_energies_torch[int(r_train * n_samples) : n_samples, :]

        test_energies_torch = test_Y[:, -1]
        test_forces_torch = test_Y[:, :-1].reshape(-1, 3, len(mol))

        self.test_shape = test_Y.shape
        self.train_shape = train_Y.shape

        train_X = train_X[:, :, :].transpose(0, 1).flatten(0, 1)
        train_Y = train_Y[:, :].transpose(0, 1).flatten(0, 1)

        test_X = test_X[:, :, :].transpose(0, 1).flatten(0, 1)
        test_Y = test_Y[:, :].transpose(0, 1).flatten(0, 1)

        # train_X = train_X[0:1000]
        # train_Y = train_Y[0:1000]

        print("Train set:")
        print(train_X.shape, train_Y.shape)
        print(train_X.dtype, train_Y.dtype)
        print(train_X.device, train_Y.device)

        print("\nTest set:")
        print(test_X.shape, test_Y.shape)
        print(test_X.dtype, test_Y.dtype)
        print(test_X.device, test_Y.device)

        train_X = train_X.to(torch.float32)
        train_Y = train_Y.to(torch.float32)
        test_X = test_X.to(torch.float32)
        test_Y = test_Y.to(torch.float32)

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y


        return

    @staticmethod
    def cd_to_root_dir(hparams):

        os.chdir(hparams["root_dir"])

        return


    @staticmethod
    @lru_cache(maxsize=10)
    def calculate_invariants(traj_file, index=":", positions=None, out_file_descriptors=None, out_file_derivatives=None):
        # saving derivatives to disk may take a lot of space

        mol_traj = io.read(traj_file, index=index)

        print(f"Total length of MD traj is {len(mol_traj)}")

        soap = SOAP(
            species=["H", "C", "O"],
            periodic=False,
            rcut=5.0,
            sigma=0.5,
            nmax=5,
            lmax=5,
            average="outer",
            crossover=True,
            dtype="float64",
            sparse=False  
        )

        print("Starting SOAP calculation...")

        pos = list(positions)
        derivatives, descriptors = soap.derivatives(
            mol_traj,
            positions=[pos] * len(mol_traj),
            n_jobs=10,
            # method="analytical"
        )
        print("SOAP calculation done!")

        # np.save("data/dump/store/invariants_derivatives_outersum_300K.npy", derivatives)
        # np.save("data/dump/store/invariants_descriptors_outersum_300K.npy", derivatives)

        if out_file_derivatives is not None and out_file_descriptors is not None:
            np.save(out_file_descriptors, descriptors)
            np.save(out_file_derivatives, derivatives)
        
        # sparse.save_npz("data/dump/store/soap_.npz", derivatives)

        return mol_traj, descriptors, derivatives

    @staticmethod
    @lru_cache(maxsize=10)
    def calculate_energies_forces(traj_file, index=":"):
        # calculates energies and forces from xyz trajectory

        mol_traj = io.read(traj_file, index=index)

        print(f"Total length of MD traj is {len(mol_traj)}")

        energies_np = np.zeros( len(mol_traj) )
        forces_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )

        # energies_np = np.memmap("../data/dump/all_energies.npy", dtype='float32', mode='w+', shape=len(mol_traj))
        # forces_np = np.memmap("../data/dump/all_forces.npy", dtype='float32', mode='w+', shape=(len(mol_traj), len(mol_traj[0]), 3))

        for i,mol in tqdm( enumerate(mol_traj) ):
            mol.calc=XTB(method="GFN2-xTB") 
            energies_np[i] = mol.get_potential_energy()
            forces_np[i,:,:] = mol.get_forces()
            mol.calc=None

        # np.save("data/dump/store_conf/forces_np_300K_randrot.npy", forces_np)
        # np.save("data/dump/store_conf/energies_np_300K_randrot.npy", energies_np)

        print(f"Energies/forces were calculated")

        return mol_traj, energies_np, forces_np