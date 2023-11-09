# Based on the number of random walks of lengths up to 10 returns the indices of atoms having similar connectivities patterns



import numpy as np
from numpy.linalg import matrix_power
from ase.io import read, write
from ase import Atoms, neighborlist
from scipy import sparse

from ase import io
import os

# from icecream import ic


def find_atomic_groups(snapshot):   
        """ 
        Uses method similar to described here: 
        https://mattermodeling.stackexchange.com/questions/4652/how-to-discard-molecules-at-the-boundary-in-the-atomic-simulation-environment-a
        """

        if isinstance(snapshot, str):
                supercell = io.read(snapshot, index="0")
                # supercell.pbc = [True, True, True] # Set periodic boundary conditions
        else:
                supercell = snapshot

        # Determine the indices of each molecule using neighborlists
        cutoff = neighborlist.natural_cutoffs(supercell)
        nl = neighborlist.build_neighbor_list(supercell, cutoffs=cutoff, bothways=True)
        connmat = nl.get_connectivity_matrix(False) # Connectivity matrix

        # n_components contains number of molecules
        # component_list contains molecule number of each atom in the system
        n_components, component_list = sparse.csgraph.connected_components(connmat)

        # Get symbols array of atoms
        origsymbols = np.array(supercell.get_chemical_symbols())

        # Excluding atoms in the boundary
        component_indices = [] # List to store selected indices
        ind_selected = [] # List to store selected indices
        for i in range(n_components): # For each molecule
                ind = np.where(component_list == i)[0] # Find indices of atoms in molecule
                component_indices.append(ind)
                # print(ind)
                # Based on MIC, if any atoms are in the boudary exclude the molecule
                if (supercell.get_distances(ind[0],ind) != supercell.get_distances(ind[0],ind, mic=True)).any():
                        continue
                ind_selected += ind.tolist() # Append selected indices to list
    

        # Positions and symbols of selected indices
        pos_selected = supercell.positions[ind_selected]
        symbols_selected = origsymbols[ind_selected]

        # Build Atoms object excluding atoms in boundaries
        finalsupercell = Atoms(
                positions=pos_selected,
                cell=supercell.cell,
                symbols=symbols_selected,
                pbc=supercell.pbc)     

        np.fill_diagonal(connmat, 0)

        connmat = np.array(connmat, dtype=np.int32)
        connmat2 = matrix_power(connmat,2)
        connmat3 = matrix_power(connmat,3)
        connmat4 = matrix_power(connmat,4)
        connmat5 = matrix_power(connmat,5)
        connmat6 = matrix_power(connmat,6)
        connmat7 = matrix_power(connmat,7)
        connmat8 = matrix_power(connmat,8)
        connmat9 = matrix_power(connmat,9)
        connmat10 = matrix_power(connmat,10)

        # feature_vector = connmat.sum(axis=-1) + 1.j* ( connmat4.sum(axis=-1) )

        feature_vector = np.array(
        list(zip( 
                supercell.get_chemical_symbols(),
                connmat.sum(axis=-1), 
                connmat2.sum(axis=-1), 
                connmat3.sum(axis=-1), 
                connmat4.sum(axis=-1), 
                connmat5.sum(axis=-1),
                connmat6.sum(axis=-1),
                connmat7.sum(axis=-1), 
                connmat8.sum(axis=-1),
                connmat9.sum(axis=-1), 
                connmat10.sum(axis=-1), 
                        )))#, dtype=np.int32)



        atomic_groups = []

        for i in range(0,len(supercell)):   #component_indices[0]:
                mask_i = set()
                # print(i)
                for j in range(0,len(supercell)):
                        # print(i)
                        if all(feature_vector[i] == feature_vector[j]):
                                mask_i.add(j)
                atomic_groups.append(mask_i)

        # print(mask)
        # ic(len(mask))
        # ic(supercell[mask])

        s = set()
        for item in atomic_groups:
                s.add(frozenset(item))


        # atomic_groups_list = list(s) ## THESE ARE THE AUTOMATIC ATOMIC GROUPS!!!
        atomic_groups_list = [sorted(list(item)) for item in s]



        print("Total number of found groups:", len(atomic_groups_list) )


        # check that all atoms are covered
        concat_list = []
        for ag in atomic_groups_list:
                concat_list += ag
        concat_list.sort()

        print("Checking if all atoms are covered: ", concat_list == list(range(0,len(supercell))))

        # atomic_groups_elements = supercell

        return atomic_groups_list