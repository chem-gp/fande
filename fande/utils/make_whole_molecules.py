from ase import neighborlist

from ase import io

import numpy as np

def make_whole_molecules(atoms_orig, molecules_seeds, N_random_attempts=3):
        """
        Make the whole molecules by translating atoms across the boundary.

        Parameters:

        atoms_orig: Atoms object or path to xyz file
        molecules_seeds: list of indices of atoms that building of molecules is started from (they will not be translated)
        N_random_attempts: number of random attempts to translate atoms to the center of molecules (to limit execution time)

        """

        global R, good_boys, connmat, atoms

        if isinstance(atoms_orig, str):
                atoms = io.read(atoms_orig).copy()
        else:
                atoms = atoms_orig.copy()

        # Determine the indices of each molecule using neighborlists
        cutoff = neighborlist.natural_cutoffs(atoms)
        nl = neighborlist.build_neighbor_list(atoms, cutoffs=cutoff, bothways=True)
        connmat = nl.get_connectivity_matrix(False) # Connectivity matrix


        R = 0 # maximum number of recursive iterations (to limit execution time)
        for iteration in range(N_random_attempts):
                good_boys = []
                for k in molecules_seeds:
                        check_neighbors_and_translate(k)

        del R, good_boys, connmat

        return atoms




def check_neighbors_and_translate(a):
        """
        a: index of atom from where the search of neighbors starts
        NOTE: adjust the cutoff values of 3.0 to your needs based on the lattice size and types of bonds in your system
        """
        global R, good_boys, connmat

        good_boys.append(a)

        nb_list = np.where(connmat[a]==1)[0]
        print("Neighbors of the atom are: ", nb_list)

        for nb in nb_list:
                if nb in good_boys:
                        continue

                dist_vector = atoms.positions[a] - atoms.positions[nb] 
                if dist_vector[0] > 3.0 and nb not in good_boys:
                        print("translate +A")
                        atoms.positions[nb] = atoms.positions[nb] + atoms.cell[0]
                if dist_vector[0] < -3.0 and nb not in good_boys:
                        print("translating -A")
                        atoms.positions[nb] = atoms.positions[nb] - atoms.cell[0]
                
                if dist_vector[1] > 3.0 and nb not in good_boys:
                        print("translate +B")
                        atoms.positions[nb] = atoms.positions[nb] + atoms.cell[1]
                if dist_vector[1] < -3.0 and nb not in good_boys:
                        print("translating -B")
                        atoms.positions[nb] = atoms.positions[nb] - atoms.cell[1]

                if dist_vector[2] > 3.0 and nb not in good_boys:
                        print("translate +C")
                        atoms.positions[nb] = atoms.positions[nb] + atoms.cell[2]
                if dist_vector[2] < -3.0 and nb not in good_boys:
                        print("translate -C")
                        atoms.positions[nb] = atoms.positions[nb] - atoms.cell[2]

                if R < 10_000:
                        R += 1
                        # print("R is ", R)
                        check_neighbors_and_translate(nb)
                else:
                        return 0

        return