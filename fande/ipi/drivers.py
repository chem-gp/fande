
from xtb.ase.calculator import XTB
import os

from ase.calculators.socketio import SocketClient, SocketIOCalculator

def make_xtb_client(calc_dir, idx, atoms, ipi_port):
        
        os.environ['OMP_NUM_THREADS'] = "1,1"
        
        temp_dir = calc_dir + "/calculator_" + str(ipi_port) + "_" + str(idx)
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)

        # for file in os.scandir(temp_dir):
        #     os.remove(file.path)

        atoms_copy = atoms.copy()

        # atoms_copy = FandeAtomsWrapper(atoms_copy)
        # atoms_copy.request_variance = True
        # fande_calc = prepare_fande_ase_calc(hparams, soap_params, gpu_id = gpu_id_list[i])
        # calc = fande_calc
        
        # https://dftb.org/parameters/download/3ob/3ob-3-1-cc
        # atoms_copy = FandeAtomsWrapper(atoms_copy)
        # atoms_copy.request_variance = False
        #     atoms_copy = RotationAtomsWrapper(atoms_copy)

        # atoms_copy.set_pbc(False)
        calc_xtb = XTB(method='GFN-FF')
        # atoms_copy.set_calculator(calc_xtb)

        atoms_copy.calc = calc_xtb

        # print( atoms_copy.get_stress() )

        print("Calculator is set up!")
        # print( atoms_copy.get_forces() )
        # Create Client
        # inet
        print(f"Launching client {idx} at port {ipi_port}")
        port = ipi_port
        host = "localhost"
        client = SocketClient(host=host, port=port)
        client.run(atoms_copy)#, use_stress=True) # for NPT set use_stress=True!

        return 0




