import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import time
import sim.FileCheck as fc
import os

from sim.PositionGenerator import N_Body_sim, n_body_sim_api, analytical_positions_api
from sim.FluxCalculation import combined_delta_flux
from sim.Decorators import TimeMeasure

@TimeMeasure
def simulate_and_interpolate_flux_vectorized(
    Stellar_params,
    planet_params,
    SamplesPerOrbit,
    numberMaxPeriod,
    times_input,
    show=False,
    save=False,
):
    """
    Optimized wrapper to perform N-body simulation, calculate flux, and interpolate flux values for given time arrays.
    Vectorized interpolation for improved efficiency.

    Parameters:
    - Stellar_params: List of stellar parameters [radius, mass]
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - SamplesPerOrbit: Number of samples per orbit for the shortest period planet
    - numberMaxPeriod: Number of periods to simulate
    - times_input: Array of arrays of times for which flux is required
    - show: Boolean to display plot
    - save: Boolean to save plot


    Returns:
    - interpolated_flux: List of arrays with interpolated flux values for each input time array
    - Plots and shows the flux values if show is True
    - Plots and saves the plot if save is True
    """
    # Run the N-body simulation
    x_pos, y_pos, x_orbit, y_orbit, simulation_times = N_Body_sim(
        StellarMass=Stellar_params[1],
        planet_params=planet_params,
        SamplesPerOrbit=SamplesPerOrbit,
        numberMaxPeriod=numberMaxPeriod,
    )

    # Run Flux Calucaltion
    z_pos = np.zeros_like(x_pos)  # z = 0 for now
    flux_values = combined_delta_flux(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        radius_star=Stellar_params[0],
        planet_params=planet_params,
        times=simulation_times,
    )

    all_times = np.concatenate(times_input)  # Combine Times for interpolation
    flux_interpolator = interp1d(
        simulation_times, flux_values, kind="linear", fill_value="extrapolate"
    )  # Interpolate Flux
    all_interpolated_flux = flux_interpolator(all_times)

    # Split interpolated flux back to the original structure of times_input -> from ChatGPT 4o
    split_indices = np.cumsum([len(times) for times in times_input[:-1]])
    interpolated_flux = np.split(all_interpolated_flux, split_indices)

    if show:  # Show the plot
        for i in range(0, len(times_input)):
            plt.plot(times_input[i], interpolated_flux[i])
        plt.xlabel("Time (days)")
        plt.ylabel("Relative Brightness")
        plt.show()

    if save:  # Save the plot
        current_directory = os.getcwd()
        Output_directory = os.path.join(current_directory, r"Outputs")
        final_directory = os.path.join(Output_directory, r"FluxPlots")
        fc.check_and_create_folder(Output_directory)
        fc.check_and_create_folder(final_directory)
        plot_path = os.path.join(
            final_directory, "FluxPlot @ " + time.strftime("%H:%M %d.%m") + ".pdf"
        )
        for i in range(0, len(times_input)):
            plt.plot(times_input[i], interpolated_flux[i])
        plt.xlabel("Time (days)")
        plt.ylabel("Relative Brightness")
        plt.savefig(plot_path)
    return interpolated_flux

def flux_data_from_params(stellar_params: np.ndarray,
    planet_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = True):

    positions = n_body_sim_api(
        stellar_mass=stellar_params[1],
        planet_params=planet_params,
        times=times,
        no_loading_bar=no_loading_bar
    )

    flux_values = combined_delta_flux(
        x=positions[:, :, 0].transpose(),
        y=positions[:, :, 1].transpose(),
        z=positions[:, :, 2].transpose(),
        radius_star=stellar_params[0],
        planet_params=planet_params,
        times=times,
    )

    return flux_values


def flux_data_from_params_Analytical(stellar_params: np.ndarray,
    planet_params: np.ndarray,
    times: np.ndarray):
    """
    Calculate flux values from analytical positions.

    Parameters:
    - stellar_params: List of stellar parameters [radius, mass]
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - times: Array of time values

    Returns:
    - flux_values: Array of flux values
    """


    positions = analytical_positions_api(
        planet_params=planet_params,
        times=times
    )

    flux_values = combined_delta_flux(
        x=positions[:, :, 0].transpose(),
        y=positions[:, :, 1].transpose(),
        z=positions[:, :, 2].transpose(),
        radius_star=stellar_params[0],
        planet_params=planet_params,
        times=times,
    )

    return flux_values

# Example Usage
if __name__ == "__main__":

    # Define Simulation Parameters
    # Stellar parameters: [radius, mass]
    Stellar_params = [100 * 4.2635e-5, 333000 * 1.12]

    # Planet parameters: [radiusRatios, mass, orbital radius, eccentricity, omega (phase)]
    planet_params = [
        [0.1, 90.26, 0.045, 0.000, 0]
        # ,[0.5 * 4.2635e-5, 66, 0.078, 0.021, 90],
        # [2 * 4.2635e-5, 70, 0.1, 0.000, 45],
    ]
    SamplesPerOrbit = 60000
    numberMaxPeriod = 4

    # Define some sample times for interpolation
    # times_input = [
    #     np.linspace(0, 12, 60000),
    #     np.linspace(15, 18, 60000),
    #     np.linspace(20, 30, 60000),
    # ]
    times_input = np.linspace(0, 30, 60000)

    # # Get Flux Values
    # interpolated_flux_output = simulate_and_interpolate_flux_vectorized(
    #     Stellar_params=Stellar_params,
    #     planet_params=planet_params,
    #     SamplesPerOrbit=SamplesPerOrbit,
    #     numberMaxPeriod=numberMaxPeriod,
    #     times_input=[times_input],
    #     show=True,
    #     save=True,
    # )

    output = flux_data_from_params(stellar_params=Stellar_params, planet_params=planet_params, times=times_input)
    # print(np.sum(output - interpolated_flux_output))

