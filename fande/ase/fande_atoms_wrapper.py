
from ase import Atoms
from ase.io import write
import os


class FandeAtomsWrapper(Atoms):
        """
        Wrapper around ASE Atoms object to add the functionality writing force/energy uncertainties on the fly.

        """   
    def __init__(
        self, 
        *args, 
        **kwargs
        ):

        super(FandeAtomsWrapper, self).__init__(*args, **kwargs)      
        self.calc_history_counter = 0
        self.request_variance = False
    
    def get_forces_variance(self):
        forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
        return forces_variance

    def get_forces(self):       
        forces = super(FandeAtomsWrapper, self).get_forces()
        if self.request_variance:
            forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
            self.arrays['forces_variance'] = forces_variance
        # energy = super(AtomsWrapped, self).get_potential_energy()
        os.makedirs("ase_calc_history" , exist_ok=True)
        write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        self.calc_history_counter += 1
        return forces