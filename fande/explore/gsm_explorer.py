# https://pubs.acs.org/doi/abs/10.1021/ct400319w#:~:text=The%20growing%20string%20method%20(GSM,for%20a%20local%20TS%20search.

#https://sites.lsa.umich.edu/zimmerman-lab/tutorial/surface-growing-string-method/growing-string-calculation/
# https://github.com/ZimmermanGroup/pyGSM
# https://zimmermangroup.github.io/pyGSM/examples/

#https://github.com/ZimmermanGroup/pyGSM/blob/master/examples/ase_api_example.py

## Running:
#SE-GSM
#gsm -xyzfile diels_alder.xyz -isomers isomers.txt -mode SE_GSM -package xTB_lot
#gsm -xyzfile diels_alder.xyz -isomers isomers.txt -mode SE_GSM -package xTB_lot > pygsm_logger.log

#DE-GSM
#gsm -xyzfile mols.xyz -mode DE_GSM -package xTB_lot


import ase.io
import numpy as np
from ase.calculators.morse import MorsePotential


from pygsm.level_of_theories.ase import ASELoT
from pygsm.optimizers import eigenvector_follow
from pygsm.potential_energy_surfaces import PES
from pygsm.utilities import elements, manage_xyz, nifty
from pygsm.wrappers import Molecule

from pygsm.coordinate_systems import DelocalizedInternalCoordinates

from xtb.ase.calculator import XTB




def main(geom):

    xtb_calc = XTB(method="GFN2-xTB")

    nifty.printcool(" Building the LOT")
    lot = ASELoT.from_options(xtb_calc, geom=geom)

    nifty.printcool(" Building the PES")
    pes = PES.from_options(
        lot=lot,
        ad_idx=0,
        multiplicity=1,
    )

    nifty.printcool("Building the topology")
    atom_symbols = manage_xyz.get_atoms(geom)
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    # top = Topology.build_topology(
    #     xyz,
    #     atoms,
    # )

    # nifty.printcool("Building Primitive Internal Coordinates")
    # p1 = PrimitiveInternalCoordinates.from_options(
    #     xyz=xyz,
    #     atoms=atoms,
    #     addtr=False,  # Add TRIC
    #     topology=top,
    # )

    nifty.printcool("Building Delocalized Internal Coordinates")
    coord_obj1 = DelocalizedInternalCoordinates.from_options(
        xyz=xyz,
        atoms=atoms,
        addtr=False,  # Add TRIC
    )

    nifty.printcool("Building Molecule")
    reactant = Molecule.from_options(
        geom=geom,
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=True,
    )

    nifty.printcool("Creating optimizer")
    optimizer = eigenvector_follow.from_options(Linesearch='backtrack', OPTTHRESH=0.0005, DMAX=0.5, abs_max_step=0.5,
                                                conv_Ediff=0.5)

    nifty.printcool("initial energy is {:5.4f} kcal/mol".format(reactant.energy))
    geoms, energies = optimizer.optimize(
        molecule=reactant,
        refE=reactant.energy,
        opt_steps=500,
        verbose=True,
    )

    nifty.printcool("Final energy is {:5.4f}".format(reactant.energy))
    manage_xyz.write_xyz('minimized.xyz', geoms[-1], energies[-1], scale=1.)


if __name__ == '__main__':
    diels_adler = ase.io.read("diels_alder.xyz", ":")
    xyz = diels_adler[0].positions

    # this is a hack
    geom = np.column_stack([diels_adler[0].symbols, xyz]).tolist()
    for i in range(len(geom)):
        for j in [1, 2, 3]:
            geom[i][j] = float(geom[i][j])
    # --------------------------

    main(geom)