import subprocess
import os
import shutil

import ase.io
import numpy as np
from tqdm import tqdm
from xtb.ase.calculator import XTB

from colorama import init, Fore, Back, Style


class Compute_MD:
    def __init__(self, mol, hparams, engine="xTB", filename="computed_md.xyz"):
        self.filename = filename
        self.hparams = hparams
        self.mol = mol

    def prepare_xtb_md_input(self):
        temp=300
        duration = 100
        dump = 50
        step = 1.0

        inp_=f"""$md
temp= {temp}.0 # in K
time= {duration}.0  # in ps
dump= {dump}.0  # in fs
step= {step}  # in fs
velo=false
nvt =true
hmass=1
shake=0
sccacc=2.0
$end"""

        print("XTB input file:")
        print(Back.GREEN,inp_)
        print(Style.RESET_ALL)
        print(f"Writing input file INPUT.inp to {os.getcwd()}")

        with open("INPUT.inp","w+") as f:
            f.writelines(inp_)

        return




    def run_xTB_md(self):
        os.chdir(self.hparams['dump_dir'])
        os.makedirs('xtb_md', exist_ok=True)
        os.chdir('xtb_md')

        ase.io.write(self.mol, "mol.xyz", format="extxyz")

        print("Preparing xtb input file...")
        self.prepare_xtb_md_input()

        print("Current working dir: ", os.getcwd())
        print("Starting calculation of MD, see the xtb_md/OUTPUT.log file for progress...")
        subprocess.run('xtb --omd --gfnff --input INPUT.inp --coffee > OUTPUT.log', capture_output=True, shell=True)
        shutil.copyfile('xtb.trj', '../xtb_traj.xyz')
        os.chdir('..')
        print("MD calculation finished")
        print("Calculating energies and forces for trajectory...")
        print("Current working dir: ", os.getcwd())

        print("Calculating forces...")
        ase_md_traj, energies_np, forces_np = self.calc_energies_forces('xtb_traj.xyz')
        print("Forces and energies were calculated!")

        print("Saving energies and forces to files")
        np.save(os.getcwd() + "/forces_np.npy", forces_np)
        np.save(os.getcwd() + "/energies_np.npy", energies_np)


        return ase_md_traj, energies_np, forces_np

    def calc_energies_forces(self, file_name):
        """Calculates energy and forces for"""

        mol_traj = ase.io.read(file_name, index=':')

        energies_np = np.zeros( len(mol_traj) )
        forces_np = np.zeros( (len(mol_traj), len(mol_traj[0]), 3) )
        
        for i,mol in tqdm(enumerate(mol_traj)):
            mol.calc=XTB(method="GFN2-xTB")     
            # a = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])          
            # a.positions = a.positions + 0.01*np.random.rand(n_atoms,3 )
            # a.positions[7] = a.positions[7] + np.array([0.0, 0.0, 1.5*np.random.rand(1)] ) + np.array([1.5,1.5,1.5])
            # mol.euler_rotate(phi=i*1.0, theta=i*0.2, psi=i*0.3, center=(0, 0, 0))
            # new_traj.append(mol)
            energies_np[i] = mol.get_potential_energy()
            forces_np[i,:,:] = mol.get_forces()
            mol.calc=None

        # ase.io.write("data/dump/rotated_traj.xyz", mol_traj,  format="extxyz")

        return mol_traj, energies_np, forces_np






