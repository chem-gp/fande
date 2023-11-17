# from ase.calculators.calculator import Calculator, all_changes
import numpy as np

import os

from ase.units import Hartree, Bohr
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

from ase import Atoms

from fande.predict import PredictorASE
# from fande.predict import SimplePredictor

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

    # default_parameters = dict(charge=0, mult=1)

    def __init__(self, 
                 predictor: PredictorASE,
                 forces_errors_plot_file=None, 
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        # self.atoms = atoms
        # self.predictor = SimplePredictor(
        #     hparams, model_e, trainer_e, model_f, trainer_f
        # )

        self.predictor = predictor

        self.energy=None
        self.forces=None

        self.nl = None

        self.supporting_calc = None
        self.forces_errors = []
        self.forces_errors_max = []
        self.forces_errors_plot_file = forces_errors_plot_file

        # self.results = None

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

# see example https://github.com/qsnake/ase/blob/master/ase/calculators/emt.py
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
        
        # self.nl.update(atoms)      
        # self.energy = 0.0
        # self.sigma1[:] = 0.0
        # self.forces[:] = 0.0


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

        # energy, energy_var = self.predictor.predict_energy_single(self.atoms)

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
            # print("Calculating FORCES with supporting calculator!")
            a_ = self.atoms.copy()
            a_.calc = self.supporting_calc
            supporting_forces = a_.get_forces()
            self.forces_errors.append(forces-supporting_forces)
            self.forces_errors_max.append( np.max(np.abs(forces-supporting_forces)) )


            if self.forces_errors_plot_file is not None and len(self.forces_errors)%self.forces_errors_loginterval==0:
                self.make_forces_errors_plot(plot_show=False, plot_file=self.forces_errors_plot_file)

        # print("FORCES calculated!")
        # natoms = len(self.atoms)
        # self.energies[:] = 0
        # self.sigma1[:] = 0.0
        # # self.forces[:] = 0.0
        # self.stress[:] = 0.0



        


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


    def make_forces_errors_plot(self, 
                                atomic_groups=None, 
                                titles=None,
                                steps_range=-1, 
                                plot_show=True, 
                                plot_file=None,
                                wandb_log=True, 
                                **kwargs):
        """
        Plot forces errors.
        """
        if atomic_groups is None:
            atomic_groups = self.atomic_groups
            if titles is None:
                titles = self.atomic_groups_titles
        
        if plot_file is None:
            plot_file = self.forces_errors_plot_file

        ngroups = len(atomic_groups)
        forces_errors = self.get_forces_errors()

        for n in range(ngroups):
            plt.figure(figsize=(20, 6), dpi=80)
            plt.plot(abs(forces_errors[0:steps_range, atomic_groups[n], :]).max(axis=-1), marker='o', label='x', linestyle='None')
            plt.legend(atomic_groups[n], title=titles[n], ncol=3, loc='center left', bbox_to_anchor=(1, 0.5),title_fontsize=18, fontsize=16)
            plt.xlabel("step", fontsize=16)
            plt.ylabel("forces max abs component error", fontsize=16)
            wandb.log({f"md-runs/forces_errors_group_{n}": float(np.abs(forces_errors[-1, atomic_groups[n], :]).max()) })
            if plot_file is not None and plot_show is False:
                plt.savefig(os.path.splitext(plot_file)[0] + "_group_{}.png".format(n), bbox_inches='tight' )
                plt.close()
                print("Plot saved to {}".format(plot_file) )
            if plot_show:
                plt.show()


        return 


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
        raise Warning("Please provide filename. Saving predictor requires humongous amount of memory! Spare some dozens of GBs!")
        torch.save(self.predictor, file_name)
        return




    # def get_xtb_energy(self, atoms=None):

    #     if atoms is not None:
    #         atoms_ = atoms.copy()
    #         atoms_.calc = self.xtb_calc
    #     else:
    #         atoms_ = self.atoms.copy()
    #         atoms_.calc = self.xtb_calc

    #     res_ = atoms_.get_potential_energy()

    #     del atoms_

    #     return res_

    # def get_xtb_forces(self, atoms=None):

    #     if atoms is not None:
    #         atoms_ = atoms.copy()
    #         atoms_.calc = self.xtb_calc
    #     else:
    #         atoms_ = self.atoms.copy()
    #         atoms_.calc = self.xtb_calc

    #     res_ = atoms_.get_forces()

    #     del atoms_

    #     return res_



################################################################
# from ase.build import molecule
# atoms = molecule("CH3CH2OCH3")
# atoms.calc = FandeCalc(hparams, hparams, model_e, trainer_e, model_f, trainer_f)
# print( atoms.get_potential_energy(), atoms.get_forces() )









# class FandeCalc2(Calculator):

#     implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
#     implemented_properties += ['stress', 'stresses']  # bulk properties
#     default_parameters = {
#         'epsilon': 1.0,
#         'sigma': 1.0,
#         'rc': None,
#         'ro': None,
#         'smooth': False,
#     }
#     nolabel = True

#     def __init__(
#             self, 
#             predictor: PredictorASE, 
#             **kwargs):

#         Calculator.__init__(self, **kwargs)

#         if self.parameters.rc is None:
#             self.parameters.rc = 3 * self.parameters.sigma

#         if self.parameters.ro is None:
#             self.parameters.ro = 0.66 * self.parameters.rc

#         self.nl = None

#         self.predictor = predictor

#     def calculate(
#         self,
#         atoms=None,
#         properties=None,
#         system_changes=all_changes,
#     ):
#         if properties is None:
#             properties = self.implemented_properties

#         Calculator.calculate(self, atoms, properties, system_changes)

#         natoms = len(self.atoms)

#         sigma = self.parameters.sigma
#         epsilon = self.parameters.epsilon
#         rc = self.parameters.rc
#         ro = self.parameters.ro
#         smooth = self.parameters.smooth

#         if self.nl is None or 'numbers' in system_changes:
#             self.nl = NeighborList(
#                 [rc / 2] * natoms, self_interaction=False, bothways=True
#             )

#         self.nl.update(self.atoms)

#         positions = self.atoms.positions
#         cell = self.atoms.cell

#         # potential value at rc
#         e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

#         energies = np.zeros(natoms)
#         forces = np.zeros((natoms, 3))
#         stresses = np.zeros((natoms, 3, 3))

#         forces = self.predictor.predict_forces_single_snapshot_r(self.atoms.copy())
#         self.forces = forces

#         # for ii in range(natoms):
#         #     neighbors, offsets = self.nl.get_neighbors(ii)
#         #     cells = np.dot(offsets, cell)

#         #     # pointing *towards* neighbours
#         #     distance_vectors = positions[neighbors] + cells - positions[ii]

#         #     r2 = (distance_vectors ** 2).sum(1)
#         #     c6 = (sigma ** 2 / r2) ** 3
#         #     c6[r2 > rc ** 2] = 0.0
#         #     c12 = c6 ** 2

#         #     if smooth:
#         #         cutoff_fn = cutoff_function(r2, rc**2, ro**2)
#         #         d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)

#         #     pairwise_energies = 4 * epsilon * (c12 - c6)
#         #     pairwise_forces = -24 * epsilon * (2 * c12 - c6) / r2  # du_ij

#         #     if smooth:
#         #         # order matters, otherwise the pairwise energy is already modified
#         #         pairwise_forces = (
#         #             cutoff_fn * pairwise_forces + 2 * d_cutoff_fn * pairwise_energies
#         #         )
#         #         pairwise_energies *= cutoff_fn
#         #     else:
#         #         pairwise_energies -= e0 * (c6 != 0.0)

#         #     pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

#         #     energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
#         #     forces[ii] += pairwise_forces.sum(axis=0)

#         #     stresses[ii] += 0.5 * np.dot(
#         #         pairwise_forces.T, distance_vectors
#         #     )  # equivalent to outer product

#         # no lattice, no stress
#         if self.atoms.cell.rank == 3:
#             stresses = full_3x3_to_voigt_6_stress(stresses)
#             self.results['stress'] = stresses.sum(axis=0) / self.atoms.get_volume()
#             self.results['stresses'] = stresses / self.atoms.get_volume()
            

#         energy = energies.sum()
#         self.results['energy'] = energy
#         self.results['energies'] = energies

#         self.results['free_energy'] = energy

#         self.results['forces'] = forces


# def cutoff_function(r, rc, ro):
#     """Smooth cutoff function.

#     Goes from 1 to 0 between ro and rc, ensuring
#     that u(r) = lj(r) * cutoff_function(r) is C^1.

#     Defined as 1 below ro, 0 above rc.

#     Note that r, rc, ro are all expected to be squared,
#     i.e. `r = r_ij^2`, etc.

#     Taken from https://github.com/google/jax-md.

#     """

#     return np.where(
#         r < ro,
#         1.0,
#         np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
#     )


# def d_cutoff_function(r, rc, ro):
#     """Derivative of smooth cutoff function wrt r.

#     Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
#     we need to multiply `2*r_ij`. This gives rise to the factor 2
#     above, the `r_ij` is cancelled out by the remaining derivative
#     `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
#     """

#     return np.where(
#         r < ro,
#         0.0,
#         np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
#     )




from ase import Atoms
from ase.io import write
import os
class FandeAtomsWrapper(Atoms):   
    def __init__(self, *args, **kwargs):
        super(FandeAtomsWrapper, self).__init__(*args, **kwargs)      
        self.calc_history_counter = 0
        self.request_variance = False
    
    def get_forces_variance(self):
        forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
        return forces_variance

    def get_forces(self):       
        forces = super(FandeAtomsWrapper, self).get_forces()
        if self.request_variance:
            forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
            self.arrays['forces_variance'] = forces_variance
        # energy = super(AtomsWrapped, self).get_potential_energy()
        os.makedirs("ase_calc_history" , exist_ok=True)
        write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        self.calc_history_counter += 1
        return forces