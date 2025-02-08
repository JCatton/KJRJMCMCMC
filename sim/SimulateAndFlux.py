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
    stellar_params: np.ndarray,
    planet_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = True,
    analytical_bool: bool = False,
    batman_bool: bool = False
) -> np.ndarray:
    """
    Calculate flux values from analytical positions.

    Parameters:
    - stellar_params: List of stellar parameters [radius, mass, limb_darkening_model, limb_darkening_coefficients]
    - planet_params: List of planet parameters planet_params: 2D numpy array where each row represents
                     a planet's parameters as
                     [eta, a, p, e, inc, omega, big_ohm, phase_lag, mass (only for N body)]
    - times: Array of time values
    - no_loading_bar: Boolean to disable loading bar
    - analytical_bool: Boolean to use analytical positions, default is False

    Returns:
    - flux_values: Array of flux values
    """

    if analytical_bool:
        if batman_bool:
            flux_values = use_batman(
                stellar_params=stellar_params,
                planet_params=planet_params,
                times=times,
            )
        else:
            positions = analytical_positions_api(planet_params=planet_params[:,1:], times=times)
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

    radius_toi_1181 = 1.961 * 696.34e6
    mass_toi_1181 = 1.467 * 2e30 / 6e24
    limb_darkening_model = "quadratic"
    limb_darkening_coefficients = [0.295, 0.312]

    stellar_params = [radius_toi_1181, mass_toi_1181, limb_darkening_model, limb_darkening_coefficients]  # Based on WASP 148


    eta1 = 0.3
    eta2 = 0.4
    # planet_params =[ [ eta,   a,     P,   e,               inc, omega, OHM, phase_lag ] ]
    planet_params = np.array(
        [
            [eta1, 0.08215, 8.803809, 0, np.radians(90), 0, 0, -np.pi/2, 0.287],
            [eta2, 0.2044, 34.525, 0, np.radians(90), 0, 0, np.pi / 4, 0.392]
        ]
    )
    # True inclinations are 89.3 and 104.9 +- some
    num_samples = 60000
    number_max_period = 4
    times_input = np.linspace(0, 4 * 34, 60000)  # Three orbital periods for planet 1


    planet_params_analytical = planet_params[:, :-1]
    output_analytical = flux_data_from_params(
        stellar_params=stellar_params, planet_params=planet_params_analytical, times=times_input, analytical_bool=True, batman_bool=True
    )

    # n_body_mask = np.array([True, False, True, True, True, True, True, True, True])
    # planet_params_n_body = planet_params[:, n_body_mask]

    # output_n_body = flux_data_from_params(
    #     stellar_params=stellar_params, planet_params=planet_params_n_body, times=times_input, analytical_bool=False
    # )


    np.save("../TestFluxesMultiple.npy", output_analytical)
    np.save("../TestTimesMultiple.npy", times_input)

    plt.plot(times_input, output_analytical, label="Analytical")
    # plt.plot(times_input, output_n_body, label="N Body")
    plt.legend()
    plt.show()


