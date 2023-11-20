# Make test plots for individual atoms on test trajectories


import numpy as np

import matplotlib.pyplot as plt

import os

from icecream import ic

import sys
from tqdm import tqdm



# def output_to_file(text):
#         """
#         Write text to file
#         """
#         with open("debug_log.log", "a") as f:
#                 f.write(text + "\n")


def make_test_plots(fande_calc, test_trajectory, save_dir):
        """
        Make test plots for individual atoms on test trajectories
        """

        # ic.configureOutput(outputFunction=output_to_file)

        # Make directory for test plots
        os.makedirs(save_dir, exist_ok=True) # remove manually to avoid overwriting

        traj = test_trajectory
        snap = traj[0].copy()

        f = []
        f_fande = []
        f_fande_uncertainty = []

        # Predict forces for test snapshots
        for i in tqdm(range(len(traj))):
                snap = traj[i]
                snap_copy = snap.copy()
                snap_copy.calc = fande_calc
                f_fande.append(snap_copy.get_forces())
                f_fande_uncertainty.append( snap_copy.calc.get_forces_variance(snap_copy) )
                f.append(snap.get_forces())

        
        symbols = snap.get_chemical_symbols()



        f = np.array(f)
        f_fande = np.array(f_fande)

        error = np.abs(f-f_fande)

        f_fande_uncertainty = np.array(f_fande_uncertainty)


        os.makedirs(save_dir, exist_ok=True)

        atomic_groups = fande_calc.predictor.fdm.atomic_groups

        # Make histogram plots for atomic groups
        for ag in range(len(atomic_groups)):
                errors_ag = error[:,ag,:]

                max_abs_ag_error = np.max(np.abs(errors_ag))
                mean_abs_ag_error = np.mean(np.abs(errors_ag))
                mse_ag_error = np.mean(errors_ag**2)

                ic("Atomic group: ", ag)
                ic( atomic_groups[ag])
                ic(max_abs_ag_error)
                ic(mean_abs_ag_error)
                ic(mse_ag_error)

                print("Atomic group " + str(ag) + ", max. abs. error: " + str(max_abs_ag_error))
                print("Atomic group " + str(ag) + ", mean abs. error: " + str(mean_abs_ag_error))
                print("Atomic group " + str(ag) + ", MSE: " + str(mse_ag_error))

                plt.title("Atomic group " + str(ag) + ", max. abs. error: " + str(max_abs_ag_error))

                

                plt.hist(errors_ag[:,0].flatten(), bins=100, label="x")
                plt.hist(errors_ag[:,1].flatten(), bins=100, label="y")
                plt.hist(errors_ag[:,2].flatten(), bins=100, label="z")
                plt.legend()
                plt.savefig(save_dir + "/histogram_ag_" + str(ag) + ".png")
                plt.close()

        # Get the atoms with largest errors:
        # todo!


        # Make test plots for individual atoms on test trajectories
        for ind in range(len(snap)):

                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

                plt.title("Atom " + str(ind) + ", " + symbols[ind])

                axs[0].plot(f[:,ind,0], label="Test")
                axs[0].plot(f_fande[:,ind,0], label="FANDE-ML", linestyle=":")
                
                axs[1].plot(f[:,ind,1], label="Test")
                axs[1].plot(f_fande[:,ind,1], label="FANDE-ML", linestyle=":")
                
                axs[2].plot(f[:,ind,2], label="Test")
                axs[2].plot(f_fande[:,ind,2], label="FANDE-ML", linestyle=":")

                axs[3].plot(error[:,ind,0], label="Error")
                axs[3].plot(f_fande_uncertainty[:,ind,0], label="Uncertainty", linestyle="--")
                plt.legend()
                plt.savefig(save_dir + "/forces_atom_" + str(ind) + ".png")
                plt.close()


        return f, f_fande, error, f_fande_uncertainty


        #         # Get atomic indices of atoms with largest errors
        #         forces_error_abs_max_ind = np.argmax(forces_error_abs, axis=1)
        #         energies_error_abs_max_ind = np.argmax(energies_error_abs, axis=1)

        #         # Get atomic indices of atoms with largest errors
        #         forces_error_abs_max_ind = np.argmax(forces_error_abs, axis=1)