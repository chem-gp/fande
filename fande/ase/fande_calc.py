from ase.calculators.calculator import Calculator
import numpy as np

from ase.units import Hartree, Bohr
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from ase import Atoms

from fande.predict import PredictorASE
from fande.predict import SimplePredictor

from xtb.ase.calculator import XTB


class FandeCalc(Calculator):
    """See for example:
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/emt.py
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = dict(charge=0, mult=1)

    def __init__(self, predictor, **kwargs):
        Calculator.__init__(self, **kwargs)
        # self.atoms = atoms
        # self.predictor = SimplePredictor(
        #     hparams, model_e, trainer_e, model_f, trainer_f
        # )

        self.predictor = predictor

        # self.xtb_calc = XTB(method="GFN2-xTB")

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

        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self.initialize(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell

        # self.energy, self.energy_var = self.predictor.predict_single_energy(
        #     self.atoms, positions=[0, 1, 4, 5]
        # )
        # self.energy = (
        #     self.energy[0] * self.model_e.normalizing_const
        #     + self.model_e.normalizing_shift
        # )

        # self.energy_var = self.energy_var[0] * self.model_e.normalizing_const

        energy, energy_var = self.predictor.predict_energy_single(self.atoms)

        self.energy = energy

        forces, forces_var = self.predictor.predict_forces_single(self.atoms)
        self.forces = forces

        # print("FORCES calculated!")

        self.energies[:] = 0
        self.sigma1[:] = 0.0
        # self.forces[:] = 0.0
        self.stress[:] = 0.0

        natoms = len(self.atoms)


        # self.forces = self.forces.reshape(3, natoms).T
        # self.forces_var = self.forces_var.reshape(3, natoms).T

        # forces = forces.reshape(len(self.atoms), -1)
        # self.forces = forces * self.model_e.normalizing_const

        # forces_var = forces_var.reshape(len(self.atoms), -1)
        # self.forces_var = forces_var * self.model_e.normalizing_const

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



################################################################
# from ase.build import molecule
# atoms = molecule("CH3CH2OCH3")
# atoms.calc = FandeCalc(hparams, hparams, model_e, trainer_e, model_f, trainer_f)
# print( atoms.get_potential_energy(), atoms.get_forces() )
