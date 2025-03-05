# main.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import time
import sim.FileCheck as fc

from sim.PositionGenerator import n_body_sim_api, analytical_positions_api
from sim.FluxCalculation import combined_delta_flux, use_batman
from sim.Decorators import TimeMeasure


# @TimeMeasure
def flux_data_from_params(
    input_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = True,
    analytical_bool: bool = False,
    batman_bool: bool = False,
) -> np.ndarray:
    """
    Calculate flux values from analytical positions.

    Parameters:
    - stellar_params: List of stellar parameters [radius, mass, limb_darkening_model, limb_darkening_coefficients]
    - input_params: List of Stellar and planet params where each row represents
                        [
                        [radius, mass, limb_darkening_model, limb_darkening_coefficient_1, limb_darkening_coefficient_2, 0:],
                        [eta, a (only for analytical), p, e, inc, omega, big_ohm, phase_lag, mass (only for N body)]
    - times: Array of time values
    - no_loading_bar: Boolean to disable loading bar
    - analytical_bool: Boolean to use analytical positions, default is False

    Returns:
    - flux_values: Array of flux values
    """
    if len(input_params.shape) == 1:
        input_params = input_params.reshape(input_params.shape[0] // 8, 8)
    stellar_params = input_params[0]
    planet_params = input_params[1:]
    # print(f"{batman_bool=}")

    if analytical_bool:
        if batman_bool:
            stellar_params_to_input = stellar_params.copy()
            # print(stellar_params_to_input)
            stellar_params_to_input[0] = stellar_params[0] * 1.496e11  # Convert radius to meters
            flux_values = use_batman(
                stellar_params=stellar_params_to_input,
                planet_params=planet_params,
                times=times,
            )
        else:
            positions = analytical_positions_api(
                planet_params=planet_params[:, 1:], times=times
            )
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
            planet_params=planet_params[:, 1:],
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

    radius_toi_1181 = 1.26 * 696.34e6 / 1.496e11
    mass_toi_1181 = 1.46 * 2e30 / 6e24
    limb_darkening_model = 2
    limb_darkening_coefficients = [0.2192, 0.3127]

    stellar_params = [
        radius_toi_1181,
        mass_toi_1181,
        limb_darkening_model,
        limb_darkening_coefficients[0],
        limb_darkening_coefficients[1],
        0,
        0,
        0,
        0,
    ]  # Based on WASP 148




    eta1 = 0.3
    eta2 = 0.4
    # planet_params =[ [ eta,   a,     P,   e,               inc, omega, OHM, phase_lag ] ]


    planet_params = np.array(
        [
            [0.09716, 0.02087, 0.9414526, 0.0091, np.radians(84.88), 1.5484, 0, 1.51935416, 0],
            [0.04716, 0.02588, 1.3, 0.0091, np.radians(84.88), 1.5484, 0, 1.51935416, 0],
            # [eta2, 0.2044, 34.525, 0, np.radians(90), 0, 0, np.pi / 4, 0.392],
        ]
    )

    input_array = np.vstack((stellar_params, planet_params))
    # True inclinations are 89.3 and 104.9 +- some
    num_samples = 400
    number_max_period = 4
    times_input = np.linspace(0, 1/4 * 34, num_samples)  # Three orbital periods for planet 1

    input_array_analytical = input_array[:, :-1]
    output_analytical = flux_data_from_params(
        input_params=input_array_analytical,
        times=times_input,
        analytical_bool=True,
        batman_bool=True,
    )

    # n_body_mask = np.array([True, False, True, True, True, True, True, True, True])
    # planet_params_n_body = planet_params[:, n_body_mask]

    # output_n_body = flux_data_from_params(
    #     stellar_params=stellar_params, planet_params=planet_params_n_body, times=times_input, analytical_bool=False
    # )

    np.save("../TestFluxes.npy", output_analytical)
    np.save("TestTimes.npy", times_input)

    plt.plot(times_input, output_analytical, label="Analytical")
    # plt.plot(times_input, output_n_body, label="N Body")
    plt.legend()
    plt.show()
