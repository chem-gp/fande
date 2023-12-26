
from xtb.ase.calculator import XTB
import os
import torch

from ase.calculators.socketio import SocketClient, SocketIOCalculator

from fande.predict import FandePredictor
from fande.ase import FandeCalc

def make_xtb_client(calc_dir, idx, atoms, ipi_port):
        
        os.environ['OMP_NUM_THREADS'] = "1,1"
        
        temp_dir = calc_dir + "/calculator_" + str(ipi_port) + "_" + str(idx)
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)

        for file in os.scandir(temp_dir):
            os.remove(file.path)

        atoms_copy = atoms.copy()
        atoms_copy.set_pbc(False)

        calc_xtb = XTB(method='GFN-FF')
        atoms_copy = FandeAtomsWrapper(atoms_copy)

        atoms_copy.calc = calc_xtb
        print("Calculator is set up!")
        print(f"Launching client {idx} at port {ipi_port}")
        port = ipi_port
        host = "localhost"
        client = SocketClient(host=host, port=port)
        client.run(atoms_copy)#, use_stress=True) # for NPT set use_stress=True!

        return 0


def make_dftb_client(calc_dir, idx, atoms, ipi_port):       
        ...
                         
def make_fande_client(calc_dir, idx, atoms, ipi_port, model_file):
        
        os.environ['OMP_NUM_THREADS'] = "1,1"
        
        temp_dir = calc_dir + "/calculator_" + str(ipi_port) + "_" + str(idx)
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)

        # for file in os.scandir(temp_dir):
        #     os.remove(file.path)

        atoms_copy = atoms.copy()

        # Load the predictor:
        predictor_loaded = torch.load(model_file)
        fande_calc_loaded = FandeCalc(predictor_loaded)
        # device = torch.device('cpu')
        # fande_calc_loaded.predictor.move_models_to_device(device)

        atoms_copy = FandeAtomsWrapper(atoms_copy)
        atoms_copy.request_variance = True
        atoms_copy.bookkeeping = True

        atoms_copy.calc = fande_calc_loaded

        print("Calculator is set up!")
        print(f"Launching client {idx} at port {ipi_port}")
        port = ipi_port
        host = "localhost"
        client = SocketClient(host=host, port=port)
        client.run(atoms_copy)#, use_stress=True) # for NPT set use_stress=True!

        return 0



from ase import Atoms
from ase import io

class FandeAtomsWrapper(Atoms):   
    def __init__(self, *args, **kwargs):
        super(FandeAtomsWrapper, self).__init__(*args, **kwargs)      
        self.calc_history_counter = 0
        self.request_variance = False
        self.bookkeeping = False
    
    def get_forces_variance(self):
        forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
        return forces_variance

    def get_forces(self):       
        forces = super(FandeAtomsWrapper, self).get_forces()
        if self.request_variance:
            forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
            self.arrays['forces_variance'] = forces_variance
        # energy = super(AtomsWrapped, self).get_potential_energy()
        if self.bookkeeping:
                os.makedirs("calc_history" , exist_ok=True)
                io.write( "calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        self.calc_history_counter += 1
        return forces