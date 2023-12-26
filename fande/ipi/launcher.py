
import os
from ase import io

import joblib

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
nohup i-pi RESTART > OUTPUT.log 2>&1 &
"""


def create_calculator(i):
        pass


def launch(
        init_structure,
        calculator: str,
        num_instances: int,
        calc_dir: str,
        input_xml_str: str
        ):
    
        calc_dir = os.path.abspath(calc_dir)
        os.makedirs(calc_dir, exist_ok=True)

        # os.chdir(calc_dir)
        # print("Working directory: ", os.getcwd())

        io.write(calc_dir + "/init.xyz", init_structure, format="extxyz")
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

def exit_calculation(calc_dir):
        os.system(f"touch {calc_dir}EXIT")
        return 0

def kill_all():
        os.system("pkill -f i-pi")
        return 0

def restart(calc_dir):
        with open(calc_dir + "/restart.sh","w+") as f:
                f.writelines(IPI_RESTART_SCRIPT)
        os.system(f"bash {calc_dir}/restart.sh {calc_dir}")
        return 0