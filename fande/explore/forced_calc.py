from typing import final
from ase.calculators.calculator import Calculator
import numpy as np

from ase.units import Hartree, Bohr
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from ase import Atoms

from fande.predict import SimplePredictor

from xtb.ase.calculator import XTB

from numpy.linalg import norm

import sys

from math import sqrt

class ForcedCalc(Calculator):
    """See for example:
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/emt.py
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = dict(charge=0, mult=1)

    def __init__(self, atoms1, atoms2, force):
        Calculator.__init__(self)

        self.xtb_calc = XTB(method="GFN2-xTB")

        self.atoms1 = atoms1
        self.atoms2 = atoms2

        self.iter = 0
        self.e_path = []

        self.grad_path = np.zeros( (2000,12,3) )

        self.force = force
        # self.force_vec = force * np.linspace( 1.45**(1/2.5), 0.1**(1/2.5), 1000)**2.5  #np.ones(1000)
        # self.force_vec = force * np.linspace( 0.95**(1/2.), 1.7**(1/2.), 1000)**4.  #np.ones(1000)
        self.force_vec = force * np.ones(1000)
        # self.force_vec = np.linspace(10, 10.1, 1000)
        # self.force_vec = np.linspace(1, 200, 1000)
        # self.force_vec = np.linspace(5, 100, 1000)
        # self.force_vec = self.force_vec ** 2
        # self.force_vec = 100*np.random.rand(1000)

    def initialize(self, atoms):
        self.par = {}
        self.numbers = atoms.get_atomic_numbers()
        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((3, 3))
        self.sigma1 = np.empty(len(atoms))
        self.deds = np.empty(len(atoms))
        self.forces_var = np.empty((len(atoms), 3))

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):

        self.iter = self.iter + 1
        if self.iter > 2000:
            sys.exit("Maximum number of iterations reached!")

        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell

        self.energies[:] = 0
        self.sigma1[:] = 0.0
        # self.forces[:] = 0.0
        self.stress[:] = 0.0

        natoms = len(self.atoms)

        # self.forces = self.forces.reshape(3, natoms).T
        # self.forces_var = self.forces_var.reshape(3, natoms).T
        self.energy = self.get_xtb_energy()
        self.e_path.append(self.energy)


        forces_raw = self.get_xtb_forces()


        # print(forces_raw.shape)
        self.grad_path[self.iter-1, :, :] = forces_raw

        # dist = self.atoms.get_distance(2,5)

        atoms1_positions = positions[self.atoms1, :]
        atoms2_positions = positions[self.atoms2, :]


        # diff_vectors = np.zeros( (len(self.atoms1), len(self.atoms2), 3))
        diff_vectors = np.zeros((natoms, natoms, 3))

        for at1 in self.atoms1:
            for at2 in self.atoms2:
                diff_vectors[at1, at2] = positions[at1] - positions[at2]

        # still needs to be checked
        forces_raw[self.atoms1] = forces_raw[self.atoms1] - self.force_vec[
            self.iter
        ] * diff_vectors[self.atoms1, :, :].sum(axis=1)

        forces_raw[self.atoms2] = forces_raw[self.atoms2] + self.force_vec[
            self.iter
        ] * diff_vectors[:, self.atoms2, :].sum(axis=0)


        self.forces = forces_raw

        self.results["energy"] = self.energy
        self.results["energies"] = self.energies
        self.results["free_energy"] = self.energy
        self.results["forces"] = self.forces

        if "stress" in properties:
            raise PropertyNotImplementedError

    def get_xtb_energy(self, atoms=None):

        if atoms is not None:
            atoms_ = atoms.copy()
            atoms_.calc = self.xtb_calc
        else:
            atoms_ = self.atoms.copy()
            atoms_.calc = self.xtb_calc

        res_ = atoms_.get_potential_energy()

        del atoms_

        return res_

    def get_xtb_forces(self, atoms=None):

        if atoms is not None:
            atoms_ = atoms.copy()
            atoms_.calc = self.xtb_calc
        else:
            atoms_ = self.atoms.copy()
            atoms_.calc = self.xtb_calc

        res_ = atoms_.get_forces()

        del atoms_

        return res_
