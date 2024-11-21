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
from sim.FluxCalculation import combined_delta_flux
from sim.Decorators import TimeMeasure

@TimeMeasure
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
        positions = analytical_positions_api(planet_params=planet_params[:,:-1], times=times)
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
            planet_params=planet_params[:, 2:],
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

    # Stellar parameters: [radius, mass]
    radius_WASP148A = 0.912 * 696.34e6 / 1.496e11
    mass_WASP148A = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_WASP148A, mass_WASP148A]  # Based on WASP 148
    radius_WASP148_B = 8.47 * 6.4e6 / 1.496e11
    radius_WASP148_c = (
        9.4 * 6.4e6 / 1.496e11
    )  # assumed similar densities as no values for radius

    eta1 = 0.1
    eta2 = 0.3
    # planet_params =[ [ eta,   a,     P,   e,               inc, omega, OHM, phase_lag ] ]
    planet_params = np.array(
        [
            [eta1, 0.08215, 8.8, 0.208, np.radians(90), 0, 0, 0, 0.208],
            [eta2, 0.2044, 34, 0.1809, np.radians(90), 0, 0, np.pi / 4, 0.1809]
        ]
    )
    # True inclinations are 89.3 and 104.9 +- some

    num_samples = 60000
    number_max_period = 4
    times_input = np.linspace(0, 4 * 34, 60000)  # Three orbital periods for planet 1


    ouptut = flux_data_from_params(
        stellar_params=stellar_params, planet_params=planet_params, times=times_input, analytical_bool=True
    )

    np.save("TestFluxesMultiple.npy", ouptut)
    np.save("TestTimesMultiple.npy", times_input)

    plt.plot(times_input, ouptut)
    plt.show()


