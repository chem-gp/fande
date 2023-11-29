import numpy as np

import os

from ase.units import Hartree, Bohr
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from ase import Atoms

from fande.predict import FandePredictor

from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress

import matplotlib.pyplot as plt


from xtb.ase.calculator import XTB

import wandb

import torch

class FandeCalc(Calculator):
    """See for example:
    Calculator to be used with Atomic Simulation Environment.

    See example calculator:
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/emt.py
    """

    implemented_properties = ["energy", "forces", "forces_variance"]
    nolabel = True


    def __init__(self, 
                 predictor: FandePredictor,
                 forces_errors_plot_file=None, 
                 **kwargs):
        Calculator.__init__(self, **kwargs)


        self.predictor = predictor

        self.energy=None
        self.forces=None

        self.nl = None

        self.supporting_calc = None
        self.forces_errors = []
        self.forces_errors_max = []
        self.forces_errors_plot_file = forces_errors_plot_file



    def initialize(self, atoms):
        self.par = {}
        self.numbers = atoms.get_atomic_numbers()
        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((3, 3))
        self.sigma1 = np.empty(len(atoms))
        self.deds = np.empty(len(atoms))
        self.forces_var = np.empty((len(atoms), 3))


    def calculate(self, 
                  atoms=None, 
                  properties=None, 
                  system_changes=all_changes):

        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()
        self.pbc = atoms.get_pbc().copy()
        

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell


        self.energy = 0.0

        natoms = len(self.atoms)
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        # forces, forces_var = self.predictor.predict_forces_single(self.atoms)
        # print("Calculating FORCES!")

        forces, forces_variance = self.predictor.predict_forces_single_snapshot_r(self.atoms.copy())

        self.forces = forces

        self.forces_variance = forces_variance

        # comparing with supporting calculation
        if self.supporting_calc is not None:
            a_ = self.atoms.copy()
            a_.calc = self.supporting_calc
            supporting_forces = a_.get_forces()
            self.forces_errors.append(forces-supporting_forces)
            self.forces_errors_max.append( np.max(np.abs(forces-supporting_forces)) )


            if self.forces_errors_plot_file is not None and len(self.forces_errors)%self.forces_errors_loginterval==0:
                self.make_forces_errors_plot(plot_show=False, plot_file=self.forces_errors_plot_file)


        self.results["energy"] = self.energy
        self.results["energies"] = self.energies
        self.results["free_energy"] = self.energy
        self.results["forces"] = self.forces
        self.results["forces_variance"] = self.forces_variance


        if "stress" in properties:
            raise PropertyNotImplementedError


    def update(self, atoms):
        if (self.energy is None or
            len(self.numbers) != len(atoms) or
            (self.numbers != atoms.get_atomic_numbers()).any()):
            self.initialize(atoms)
            self.calculate(atoms)
        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()
    
    def get_forces_variance(self, atoms):
        self.update(atoms)
        return self.forces_variance.copy()
    

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.energy

    def get_forces_errors(self):
        return np.array(self.forces_errors)
    
    def set_forces_errors_plot_file(self, forces_errors_plot_file, loginterval=10):
        self.forces_errors_plot_file = forces_errors_plot_file
        self.forces_errors_loginterval = loginterval

    def set_atomic_groups(self, atomic_groups, titles=None):
        self.atomic_groups = atomic_groups
        if titles is None:
            titles = [f"Group {n}" for n in range(len(atomic_groups))]
        self.atomic_groups_titles = titles        


    def save_predictor(self, file_name):
        """
        Save the predictor to file. Everything is saved, including the model, the trainer, the hparams, and descriptors. 
        Huge file is generated.

        Sample loading of predictor:
        ``` python
        import torch
        from fande.ase import FandeCalc

        predictor_loaded = torch.load("/data1/simulations/ML_models/predictor.pt")
        fande_calc = FandeCalc(predictor_loaded)

        atoms = predictor_loaded.fdm.traj_train[0].copy()
        atoms.calc = fande_calc

        print( atoms.get_potential_energy(), atoms.get_forces() )
        ```

        """
        print("Saving predictor requires humongous amount of memory! Spare some dozens of GBs!")

        # free up memory to make the artifact smaller
        try:
            del self.predictor.fdm.train_DX, self.predictor.fdm.train_F, self.predictor.fdm.test_DX, self.predictor.fdm.test_F
        except:
            pass

        try:
            del self.predictor.fdm.train_X, self.predictor.fdm.train_E, self.predictor.fdm.test_X, self.predictor.fdm.test_E
        except:
            pass

        try:
            del self.train_DX, self.train_F, self.test_DX, self.test_F
        except:
            pass

        try:
            del self.train_X, self.train_E, self.test_X, self.test_E
        except:
            pass

        try:
            del self.test_X, self.test_E
        except:
            pass

        try:
            del self.test_DX, self.test_F
        except:
            pass

        torch.save(self.predictor, file_name)
        return





