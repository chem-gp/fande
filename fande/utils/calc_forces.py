# %% https://xtb-python.readthedocs.io/en/latest/general-api.html
from turtle import position
from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED
from xtb.interface import Calculator, Param
import numpy as np
from ase import io
from ase.units import Bohr, Hartree

from tqdm import tqdm

 
# %%
%%time
## Using XTB calculator from ASE:

from xtb.ase.calculator import XTB
# atoms = mol_traj[0] #molecule('H2O')
# atoms.calc = XTB(method="GFN2-xTB")
# print(atoms.get_potential_energy() * 0.0367493)
# print(atoms.get_forces())

mol_traj = io.read("../data/dump/mol_trj.xyz", index="5000:10000")

energies_np = np.zeros( len(mol_traj) )
forces_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
 
# energies_np = np.memmap("../data/dump/all_energies.npy", dtype='float32', mode='w+', shape=len(mol_traj))
# forces_np = np.memmap("../data/dump/all_forces.npy", dtype='float32', mode='w+', shape=(len(mol_traj), len(mol_traj[0]), 3))

print("okay...")

for i,mol in tqdm( enumerate(mol_traj) ):
      mol.calc=XTB(method="GFN2-xTB") 
      energies_np[i] = mol.get_potential_energy()
      forces_np[i,:,:] = mol.get_forces()

np.save("../data/dump/forces_np.npy", forces_np)
np.save("../data/dump/energies_np.npy", energies_np)

print(energies_np)
print(forces_np)

# positions_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
# energies_np = np.zeros( len(mol_traj) )
# forces_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
# numbers_np = mol_traj[0].numbers

# print(mol_traj[0].positions/Bohr)





# %%
%%time
## Using xtb-python methods:

mol_traj = io.read("../data/dump/mol_trj.xyz", index="5000:10000")

positions_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
energies_np = np.zeros( len(mol_traj) )
forces_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
numbers_np = mol_traj[0].numbers

print(mol_traj[0].positions/Bohr)

for i, mol in tqdm( enumerate(mol_traj) ):
      positions_np[i,:,:] =  mol.positions/Bohr

      calc = Calculator(Param.GFN2xTB, numbers_np, positions_np[i,:,:]/Bohr)
      calc.set_verbosity(VERBOSITY_MUTED)

      res = calc.singlepoint()
      energies_np[i] = res.get_energy()
      forces_np[i,:,:] = res.get_gradient()

np.save("../data/dump/forces_np.npy", forces_np)
np.save("../data/dump/energies_np.npy", energies_np)


print("Energies:")
print(energies_np[0])

print("Forces:")
print(forces_np[0])





# numbers = np.array([8, 1, 1])
# positions = np.array([
# [ 0.00000000000000, 0.00000000000000,-0.73578586109551],
# [ 1.44183152868459, 0.00000000000000, 0.36789293054775],
# [-1.44183152868459, 0.00000000000000, 0.36789293054775]])


