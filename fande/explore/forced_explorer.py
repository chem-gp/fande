# For theory and references see for example:
# https://arxiv.org/pdf/2202.13011.pdf


import numpy as np

import networkx as nx

from .forced_calc import ForcedCalc
from xtb.ase.calculator import XTB
from ase.calculators.emt import EMT

import ase
from ase import io
from ase import Atoms
from ase.optimize import BFGS

from ase.visualize import view

import matplotlib.pyplot as plt

import sys

from tqdm import tqdm

import matplotlib.cm as cm


def partition(collection):
    # https://stackoverflow.com/questions/19368375/set-partitions-in-python

    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


class ForcedExplorer:
    def __init__(self, atoms, force, constraints=None, logfile=None):
        self.atoms = atoms.copy()
        self.natoms = len(self.atoms)
        self.force = force

        self.traj_forced = None
        self.traj_opt = None

        self.logfile = logfile
        self.optimizer_logfile = "data/dump/ase/bfgs.log"

        self.G = nx.DiGraph()

        self.node_list = []

    def prepare_partition(self, atoms, species="all"):

        if species == "all":
            atoms_ind = list(range(self.natoms))
            full_partition = list(partition(atoms_ind))
            return full_partition

        all_symbols = atoms.get_chemical_symbols()
        selected_indices = []

        for i, s in enumerate(all_symbols):
            if s in species:
                selected_indices.append(i)

        part = list(partition(selected_indices))
        return part

    def reset(self):

        del self.node_list
        del self.G

        print("Nodes and graph G has been deleted!")

        self.node_list = []
        self.G = nx.DiGraph()

        return

    def pop_new_node(self):

        if len(self.node_list) == 0:
            self.node_list.append(0)
            return 0
        else:
            new_id = max(self.node_list) + 1
            self.node_list.append(new_id)
            return new_id

    def single_forced_run(self, atoms, atoms_partition, force):

        atoms_list_0 = atoms_partition[0]
        atoms_list_1 = atoms_partition[1]

        atoms.calc = ForcedCalc(atoms_list_0, atoms_list_1, force=force)

        dyn = BFGS(
            atoms,
            maxstep=0.05,
            trajectory="data/dump/ase/minimizer.traj",
            logfile=self.optimizer_logfile,
        )
        dyn.run(fmax=0.1)

        energy_pre_opt = atoms.get_potential_energy()

        traj1 = io.read("data/dump/ase/minimizer.traj", index=":")
        # io.write("data/dump/ase/minimizer.xyz", traj1, format="xyz")

        e_path = atoms.calc.e_path

        atoms2 = atoms.copy()
        atoms2.calc = XTB(method="GFN2-xTB")

        dyn = BFGS(
            atoms2,
            maxstep=0.05,
            trajectory="data/dump/ase/minimizer_post_opt.traj",
            logfile=self.optimizer_logfile,
        )

        try:
            dyn.run(fmax=0.05)
            # print('done')
        except XTBException:
            print("ITERATION DID NOT CONVERGE")

        # print("POST OPTIMIZATION OKAY")

        energy_opt = atoms2.get_potential_energy()

        atoms_opt = atoms2.copy()

        # atoms.positions
        traj2 = io.read("data/dump/ase/minimizer_post_opt.traj", index=":")
        # io.write("data/dump/ase/minimizer_changed.xyz", traj, format="xyz")

        traj = traj1 + traj2

        io.write("data/dump/ase/full_traj.xyz", traj, format="xyz")
        # print(f"External force steps: {len(traj1)}" )
        # print(f"Post optimization steps: {len(traj2)}" )

        return traj, e_path, atoms_opt, energy_opt, energy_pre_opt

    def single_explore(
        self,
        force=10.0, 
        max_exps=50, 
        n_parallel=1, 
        custom_partition=None, 
        species="all", 
        depth=0
    ):
        print("STARTING")
        # stdout_obj = sys.stdout
        # sys.stdout = open(self.logfile, "w")

        atoms = self.atoms

        if custom_partition is None:
            atoms_partitions_full = self.prepare_partition(atoms)
        else:
            atoms_partitions_full = custom_partition

        if species != "all":
            atoms_partitions_full = self.prepare_partition(atoms, species=species)

        energy_opt = self.get_opt_energy(atoms)

        parent_id = self.pop_new_node()
        self.G.add_node(parent_id, atoms=atoms, energy_opt=energy_opt)

        print("STARTING SEARCH")

        for it, p in tqdm(enumerate(atoms_partitions_full), desc="Progress: "):

            if len(p) == 1 or len(p) == 0:
                continue

            if it >= max_exps:
                break

            a_ = atoms.copy()

            try:
                (
                    traj,
                    e_path,
                    atoms_opt,
                    energy_opt,
                    energy_pre_opt,
                ) = self.single_forced_run(a_, p, force = force)
            except:
                print("ERROR DURING single_force_run(), please check")
                continue

            child_id = self.pop_new_node()
            self.G.add_node(
                child_id,
                atoms=atoms_opt,
                energy_opt=energy_opt,
                energy_pre_opt=energy_pre_opt,
            )

            self.G.add_edge(
                parent_id, child_id, e_path=e_path, path_max=max(e_path), traj=traj
            )

            # print("single_forced okay for it=", it)

        # sys.stdout = stdout_obj

        G = self.G

        return G


    def long_eploration(self):
        raise NotImplementedError



    def get_opt_energy(self, atoms):
        atoms.calc = XTB(method="GFN2-xTB")
        # energy_opt = atoms.get_potential_energy()
        dyn = BFGS(
            atoms,
            maxstep=0.05,
            trajectory="data/dump/ase/optimizer_run.traj",
            logfile=self.optimizer_logfile,
        )
        dyn.run(fmax=0.01)
        energy_opt = atoms.get_potential_energy()

        return energy_opt

    def draw_network(self):

        cmap = plt.cm.viridis

        energies_opt = np.array([self.G.nodes[n]["energy_opt"] for n in self.G.nodes])

        path_max = np.array([d["path_max"] for u, v, d in self.G.edges(data=True)])

        print(energies_opt)
        print(path_max)

        node_colors = cm.rainbow(energies_opt - min(energies_opt))
        edge_colors = cm.rainbow(path_max)  # - min(path_max))

        nx.draw(
            self.G,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=plt.cm.coolwarm,
            with_labels=True,
        )

        plt.draw()

        return

    def plot_energy_path(self, node1, node2):
        # still check
        e_path = self.G.edges[node1, node2]["e_path"]
        plt.plot(e_path)
        plt.ylabel("energy")
        plt.xlabel("step")
        plt.show()

        return
