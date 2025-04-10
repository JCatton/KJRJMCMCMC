import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sim.SimulateAndFlux import flux_data_from_params
from real_data_analysis import extend_params_for_stellar
from MCMC.mcmc import find_first_greater

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
plt.rcParams['text.usetex'] = False
from TransitAnalysis.TransitDetector import run_tls
from MCMC.mcmc import MCMC
from MCMC.main import gaussian_error_ln_likelihood








def plot_phase_curve(file_name, stellar_temp):

    # get the number of folders in the file
    num_folders = len(os.listdir(file_name))

    for i in range(num_folders):
        # get the folder name
        folder_name = f"{file_name}/run_{i}/"
        input_values_input = np.load(folder_name + "Input_values.npy")
        literature_values_input = np.load(folder_name + "Literature_values.npy")
        our_values_input = np.load(folder_name + "our_values.npy")
        times = np.load(folder_name + "times.npy")
        flux = np.load(folder_name + "flux.npy")

        mcmc_class = MCMC.load(file_name + f"/run_{i}")
        burn_in_index = mcmc_class.determine_burn_in_index()

        chain = mcmc_class.chain[burn_in_index:]

        stellar_radius = literature_values_input[0,0]

        num_planets = int(len(our_values_input) - 1)

        temp_estimate = np.zeros((num_planets, len(chain)))

        for j in range(num_planets):
            planet_index = j + 1

            boolean_map = np.zeros(len(our_values_input))

            boolean_map[planet_index] = 1
            boolean_map[0] = 1

            print(f"{our_values_input=}")
            print(f"{planet_index=}")

            chain_planet = chain[:, boolean_map.astype(bool)]
            print(f"{chain_planet.shape=}")
            input_params = input_values_input[boolean_map.astype(bool)]
            literature_params = literature_values_input[boolean_map.astype(bool)]
            our_params = our_values_input[boolean_map.astype(bool)]

            semi_major_axis = chain_planet[:, 1, 1]

            temp_to_the_four = stellar_temp*stellar_temp*stellar_temp*stellar_temp

            semi_major_axis_squared = semi_major_axis*semi_major_axis

            stellar_radius_squared = stellar_radius*stellar_radius

            temp = ((stellar_radius_squared*temp_to_the_four)/(4*semi_major_axis_squared))**(1/4)

            print(temp)

            plt.hist(temp, bins=100)
            plt.show()
            print(temp.mean())

            temp_estimate[j] = temp









if __name__ == "__main__":

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from sim.SimulateAndFlux import flux_data_from_params
    from real_data_analysis import extend_params_for_stellar

    import numpy as np
    import matplotlib.pyplot as plt
    import lightkurve as lk

    plot_phase_curve("For_Report_TOI-1130_fixed_i", stellar_temp=4350)
