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


class FandeCalc(Calculator):
    """See for example:
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/emt.py
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = dict(charge=0, mult=1)

    def __init__(self, hparams, model_e, trainer_e, model_f, trainer_f, **kwargs):
        Calculator.__init__(self, **kwargs)
        # self.atoms = atoms
        self.predictor = SimplePredictor(
            hparams, model_e, trainer_e, model_f, trainer_f
        )

        self.model_e = model_e
        self.model_f = model_f

        self.xtb_calc = XTB(method="GFN2-xTB")

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

        self.energy, self.energy_var = self.predictor.predict_single_energy(
            self.atoms, positions=[0, 1, 4, 5]
        )
        self.energy = (
            self.energy[0] * self.model_e.normalizing_const
            + self.model_e.normalizing_shift
        )

        self.energy_var = self.energy_var[0] * self.model_e.normalizing_const

        forces, forces_var = self.predictor.predict_single_forces(
            self.atoms, positions=[0, 1, 4, 5]
        )
        self.energies[:] = 0
        self.sigma1[:] = 0.0
        # self.forces[:] = 0.0
        self.stress[:] = 0.0

        natoms = len(self.atoms)

        # self.forces = self.forces.reshape(3, natoms).T
        # self.forces_var = self.forces_var.reshape(3, natoms).T

        forces = forces.reshape(len(self.atoms), -1)
        self.forces = forces * self.model_e.normalizing_const

        forces_var = forces_var.reshape(len(self.atoms), -1)
        self.forces_var = forces_var * self.model_e.normalizing_const

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


from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import io
import os


class MDRunner:
    def __init__(self, atoms, traj_filename, log_filename):
        self.atoms = atoms
        self.traj_filename = traj_filename
        self.log_filename = log_filename

    def run(self):
        # atoms.calc = EMT()

        # Set the momenta corresponding to T=300K
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=300)

        # We want to run MD with constant energy using the VelocityVerlet algorithm.
        os.makedirs("data/dump/ase", exist_ok=True)
        dyn = VelocityVerlet(
            self.atoms,
            0.4 * units.fs,
            trajectory=os.path.dirname(self.traj_filename) + "md.traj",
            logfile=self.log_filename,
        )  # 5 fs time step.

        def printenergy(a):
            """Function to print the potential, kinetic and total energy"""
            epot = a.get_potential_energy() / len(a)
            ekin = a.get_kinetic_energy() / len(a)
            print(
                "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
                "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
            )

        dyn.run(10)

        # # Now run the dynamics
        # printenergy(atoms)
        # for i in range(5):
        #     dyn.run(10)
        #     # printenergy(atoms)

        traj = io.read(os.path.dirname(self.traj_filename) + "md.traj", index=":")
        io.write(self.traj_filename, traj, format="xyz")


################################################################
# from ase.build import molecule
# atoms = molecule("CH3CH2OCH3")
# atoms.calc = FandeCalc(hparams, hparams, model_e, trainer_e, model_f, trainer_f)
# print( atoms.get_potential_energy(), atoms.get_forces() )
