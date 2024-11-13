# main.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import time
import sim.FileCheck as fc

from sim.PositionGenerator import n_body_sim, n_body_sim_api, analytical_positions_api
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
    x_pos, y_pos, x_orbit, y_orbit, simulation_times = n_body_sim(
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


"""
def flux_data_from_params(
    stellar_params: np.ndarray,
    planet_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = True,
):

    positions = n_body_sim_api(
        stellar_mass=stellar_params[1],
        planet_params=planet_params,
        times=times,
        no_loading_bar=no_loading_bar,
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
"""


def flux_data_from_params(
    stellar_params: np.ndarray,
    planet_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = False,
    analytical_bool: bool = False,
) -> np.ndarray:
    """
    Calculate flux values from analytical positions.

    Parameters:
    - stellar_params: List of stellar parameters [radius, mass]
    - planet_params: List of planet parameters [eta, p, a, e, inc, omega, big_ohm, phase_lag]
    - times: Array of time values
    - no_loading_bar: Boolean to disable loading bar
    - analytical_bool: Boolean to use analytical positions, default is False

    Returns:
    - flux_values: Array of flux values
    """
    if analytical_bool:
        positions = analytical_positions_api(planet_params=planet_params, times=times)
        flux_values = combined_delta_flux(
            x=positions[:, :, 0].transpose(),
            y=positions[:, :, 1].transpose(),
            z=positions[:, :, 2].transpose(),
            radius_star=stellar_params[0],
            eta_values=planet_params[:, 0],
            times=times,
        )

    else:
        positions = n_body_sim_api(
            stellar_mass=stellar_params[1],
            planet_params=planet_params,
            times=times,
            no_loading_bar=no_loading_bar,
        )
        # Get relative positions of the x,y,z coordinates from their star
        x = positions[:, :, 0].transpose()
        y = positions[:, :, 1].transpose()
        z = positions[:, :, 2].transpose()
        x_s, y_s, z_s = x[0], y[0], z[0]
        x_p_rel = x[1:] - x_s
        y_p_rel = y[1:] - y_s
        z_p_rel = z[1:] - z_s

        flux_values = combined_delta_flux(
            x=x_p_rel,
            y=y_p_rel,
            z=z_p_rel,
            radius_star=stellar_params[0],
            eta_values=planet_params[:, 0],
            times=times,
        )

    return flux_values


# Example Usage
if __name__ == "__main__":

    # Define Simulation Parameters
    # Stellar parameters: [radius, mass]
    # Stellar_params = [100 * 4.2635e-5, 333000 * 1.12]

    # Planet parameters: [radiusRatios, mass, orbital radius, eccentricity, omega (phase)]
    # planet_params = [
    #     [0.1, 90.26, 0.045, 0.000, 0]
    #     # ,[0.5 * 4.2635e-5, 66, 0.078, 0.021, 90],
    #     # [2 * 4.2635e-5, 70, 0.1, 0.000, 45],
    # ]

    """
    Use below params for analytical positions
    """
    # Stellar parameters: [radius, mass]
    radius_WASP148A = 0.912 * 696.34e6 / 1.496e11
    mass_WASP148A = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_WASP148A, mass_WASP148A]  # Based on WASP 148
    radius_WASP148_B = 8.47 * 6.4e6 / 1.496e11
    radius_WASP148_c = (
        9.4 * 6.4e6 / 1.496e11
    )  # assumed similar densities as no values for radius
    eta1 = radius_WASP148_B / radius_WASP148A
    eta2 = radius_WASP148_c / radius_WASP148A

    eta1 = 0.1
    eta2 = 0.3
    # planet_params =[ [ eta,   P,     a,   e,               inc, omega, OHM, phase_lag ] ]
    planet_params = np.array(
        [
            [eta1, 8.8, 0.08, 0.208, np.radians(90), 0, 0, 0]
            # [eta2, 34.5, 0.20, 0.1809, np.radians(90), 0, 0, np.pi / 4],
        ]
    )
    # True inclinations are 89.3 and 104.9 +- some

    SamplesPerOrbit = 60000
    numberMaxPeriod = 4

    # Define some sample times for interpolation
    # times_input = [
    #     np.linspace(0, 12, 60000),
    #     np.linspace(15, 18, 60000),
    #     np.linspace(20, 30, 60000),
    # ]
    times_input = np.linspace(0, 2 * 34.5, 60000)  # Three orbital periods for planet 1

    ouptut = flux_data_from_params(
        stellar_params=stellar_params, planet_params=planet_params, times=times_input
    )

    np.save("TestFluxes.npy", ouptut)
    np.save("TestTimes.npy", times_input)

    plt.plot(times_input, ouptut)
    # plt.ylim(0.999, 1.001)
    plt.show()

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

    # output = flux_data_from_params(
    #     stellar_params=Stellar_params, planet_params=planet_params, times=times_input
    # )
    # print(np.sum(output - interpolated_flux_output))
