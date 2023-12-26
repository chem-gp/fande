
import os
from ase import io

import joblib


def launch(
        init_structure,
        calculator: str,
        num_instances: int,
        calc_dir: str,
        input_xml_str: str
        ):
    
        calc_dir = os.path.abspath(calc_dir)
        os.makedirs(calc_dir, exist_ok=True)

        os.chdir(calc_dir)
        print("Working directory: ", os.getcwd())

        with open("input.xml","w+") as f:
                f.writelines(input_xml_str)

        io.write("init.xyz", init_structure, format="extxyz")

        # Run i-pi
        # os.system("source ~/repos/i-pi/env.sh; which i-pi >> OUTPUT.log")

        # Run multiple instances of fande_calc
        # with joblib
        # joblib.Parallel(n_jobs=num_instances)(
        #         joblib.delayed(fande_calc)(i) for i in range(num_instances)
        # )

        print("Launched! Please manually check if i-pi is running.")



        return 0


def kill_all():
        pass