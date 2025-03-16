import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sim.SimulateAndFlux import flux_data_from_params
from real_data_analysis import extend_params_for_stellar

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

from TransitAnalysis.TransitDetector import run_tls

def generate_flux_and_phase_fold(params, times, period):
    fluxes = flux_data_from_params(params, times, no_loading_bar=True, analytical_bool=True, batman_bool=True)
    folded_times, folded_fluxes = phase_fold(times, fluxes, period)

    return folded_times, folded_fluxes


def phase_fold(time, fluxes, period):
    folded_times = time % period - 1

    order = np.argsort(folded_times)
    # sort the true_fluxes the same way

    folded_times = folded_times[order]
    folded_fluxes = fluxes[order]
    return folded_times, folded_fluxes


def find_mid_transit(times, fluxes):
    times_output = np.where(fluxes <= 0.995)
    # print(f"{times_output=}")
    # find midpoint of these
    return np.mean(times[times_output])


def bin_data(times, fluxes, num_bins=10):
    if len(times) < num_bins:  # Avoid indexing beyond array size
        print("Warning: Not enough data points for binning. Returning original arrays.")
        return times, fluxes

    binned_fluxes = []
    binned_times = []

    for i in range(0, len(times), num_bins):
        flux_segment = fluxes[i:i + num_bins]
        time_segment = times[i:i + num_bins]

        if len(flux_segment) == 0 or len(time_segment) == 0:
            print(f"Warning: Empty slice detected at bin {i//num_bins}!")
            continue  # Skip empty bins

        binned_fluxes.append(np.mean(flux_segment))
        binned_times.append(np.mean(time_segment))

    return np.array(binned_times), np.array(binned_fluxes)




def plot_phase_curve(file_name, xlims:tuple = None, ylims:tuple = None, num_bins = 10, fit_x_shift=False):

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

        num_planets = int(len(our_values_input) - 1)

        for j in range(num_planets):
            planet_index = j + 1

            boolean_map = np.zeros(len(our_values_input))

            boolean_map[planet_index] = 1
            boolean_map[0] = 1

            print(f"{our_values_input=}")
            print(f"{planet_index=}")


            input_params = input_values_input[boolean_map.astype(bool)]
            literature_params = literature_values_input[boolean_map.astype(bool)]
            our_params = our_values_input[boolean_map.astype(bool)]


            period = our_params[1][2]


            # Phase Fold the data
            folded_raw_times, folded_raw_fluxes = phase_fold(times, flux, period)


            literature_times, literature_fluxes = generate_flux_and_phase_fold(literature_params, times, literature_params[1][2])
            our_times, our_fluxes = generate_flux_and_phase_fold(our_params, times, our_params[1][2])
            input_times, input_fluxes = generate_flux_and_phase_fold(input_params, times, input_params[1][2])   

            # Bin the data
            if fit_x_shift == False:
                fit_x_shift = np.array([0,0,0,0])
            binned_raw_times, binned_raw_fluxes = bin_data(folded_raw_times, folded_raw_fluxes, num_bins=num_bins) 
            binned_literature_times, binned_literature_fluxes = bin_data(literature_times, literature_fluxes, num_bins=num_bins) 
            binned_our_times, binned_our_fluxes = bin_data(our_times, our_fluxes, num_bins=num_bins) 
            binned_input_times, binned_input_fluxes = bin_data(input_times, input_fluxes, num_bins=num_bins) 
            

            # zero the times

            #Zero the raw time from tls output mid time -> Here we get a noise issue which messes things up

            binned_raw_times = binned_raw_times - find_mid_transit(binned_input_times, binned_input_fluxes) + fit_x_shift[0]

            # binned_raw_times = binned_raw_times - find_mid_transit(binned_raw_times, binned_raw_fluxes) + fit_x_shift[0]

            binned_literature_times = binned_literature_times - find_mid_transit(binned_literature_times, binned_literature_fluxes) + fit_x_shift[1]

            binned_our_times = binned_our_times - find_mid_transit(binned_our_times, binned_our_fluxes) + fit_x_shift[2]

            binned_input_times = binned_input_times - find_mid_transit(binned_input_times, binned_input_fluxes) + fit_x_shift[3]


            fig, ax = plt.subplots((1))

            #title
            ax.set_title(f"Planet {planet_index} Phase Curve")
            # raw data
            ax.plot(binned_raw_times, binned_raw_fluxes, "x", label="Raw Data", color="black")

            # literature data
            ax.plot(binned_literature_times, binned_literature_fluxes, label="Literature Data", color="red")
            # our data
            ax.plot(binned_our_times, binned_our_fluxes, label="Our Data", color="blue")
            # input data
            ax.plot(binned_input_times, binned_input_fluxes, label="Input Data", color="green")


            ax.set_xlabel("Phase")
            ax.set_ylabel("Relative Flux")
            if xlims:
                ax.set_xlim(xlims)
            if ylims:
                ax.set_ylim(ylims)
            ax.legend()


            plt.savefig(f"{file_name}/run_{i}/planet_{planet_index}_phase_curve.pdf")

            # Now snip out the last transit

            # def run_tls(
            #     data: np.ndarray,
            #     times_input: np.ndarray,
            #     limb_darkening_model: str,
            #     limb_darkening_coefficients: list,
            #     plot_bool=False,
            #     save_loc=None,
            #     index=None,
            #     duration_multiplier=4,
            #     period_min=None,
            #     period_max=None,
            # )

            plt.show()


            flux, times, output_dict = run_tls(data = flux, 
                                               times_input = times, 
                                               limb_darkening_model = "quadratic", 
                                               limb_darkening_coefficients = [literature_params[0,3],literature_params[0,4]], 
                                               plot_bool = True,
                                               duration_multiplier=4,
                                               period_min = period - 1,
                                               period_max = period + 1,
            )          
            plt.show()
            plt.plot(times, flux)
            plt.show()




if __name__ == "__main__":

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from sim.SimulateAndFlux import flux_data_from_params
    from real_data_analysis import extend_params_for_stellar

    import numpy as np
    import matplotlib.pyplot as plt
    import lightkurve as lk

    plot_phase_curve("Simulated_for_viva_modeled_after_TOI-1516", xlims=(-0.1, 0.1), ylims=(0.9825, 1.0075), num_bins=2, fit_x_shift=[+0.0033,-0.0,-0,-0])
