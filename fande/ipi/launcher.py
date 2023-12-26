
import os
from ase import io

from joblib import Parallel, delayed

from .drivers import make_xtb_client, make_fande_client

IPI_PATH = os.path.expanduser("~/repos/i-pi/")

IPI_LAUNCH_SCRIPT = """
#!/bin/bash
source ~/repos/i-pi/env.sh
cd $1
nohup i-pi input.xml > OUTPUT.log 2>&1 &
"""

IPI_RESTART_SCRIPT = """
#!/bin/bash
source ~/repos/i-pi/env.sh
cd $1
rm EXIT
nohup i-pi RESTART >> OUTPUT.log 2>&1 &
"""


def launch_server(
        init_structure,
        calc_dir: str,
        input_xml_str: str,
        ):
    
        calc_dir = os.path.abspath(calc_dir)
        os.makedirs(calc_dir, exist_ok=True)

        # os.chdir(calc_dir)
        # print("Working directory: ", os.getcwd())

        io.write(calc_dir + "/init.xyz", init_structure, format="extxyz")
        # io.write(calc_dir + "/init.cif", init_structure, format="cif")
        io.write(calc_dir + "/POSCAR", init_structure, format="vasp")
        
        with open(calc_dir + "/input.xml","w+") as f:
                f.writelines(input_xml_str)

        with open(calc_dir + "/launch.sh","w+") as f:
                f.writelines(IPI_LAUNCH_SCRIPT)

        print("Launching i-pi...")
        print(f"bash {calc_dir}/launch.sh")
        os.system(f"bash {calc_dir}/launch.sh {calc_dir}")
       
        # Run multiple instances of fande_calc
        # with joblib
        # joblib.Parallel(n_jobs=num_instances)(
        #         joblib.delayed(fande_calc)(i) for i in range(num_instances)
        # )

        print("Launched! Please manually check if i-pi is running.")

        return 0


def launch_drivers(
        init_structure,
        calculator: str,
        num_instances: int,
        calc_dir: str,
        ipi_port: int,
        model_file=None,
        ):

        # gpu_id_list = []
        # gpu_id_list = [0, 1, 2, 3, 4, 5, 6, 7] 
        atoms = init_structure.copy()
        print("Starting clients with joblib...")

        status = Parallel(
                n_jobs=num_instances, prefer="processes")(delayed(make_xtb_client)(calc_dir, i, atoms, ipi_port) for i in range(0, num_instances)
                ) 
        
        return status

def exit_calculation(calc_dir):
        os.system(f"touch {calc_dir}EXIT")
        return 0

def kill_all():
        # os.system("pkill -f i-pi")
        os.system("kill -9 $(pgrep -f i-pi)")
        return 0

def restart(calc_dir):
        with open(calc_dir + "/restart.sh","w+") as f:
                f.writelines(IPI_RESTART_SCRIPT)
        os.system(f"bash {calc_dir}/restart.sh {calc_dir}")
        return 0