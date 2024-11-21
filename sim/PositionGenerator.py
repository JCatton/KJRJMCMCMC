# main.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rebound
import sim.FileCheck as fc
import numpy as np
from tqdm import tqdm
import os
from numba import jit


@jit(nopython=True)
def kepler_solver(mean_anomaly, p, e, a, tol=1e-9, max_iter=100):
    """
    Solve Kepler's equation for true anomaly and orbital radius.

    Parameters:
    - mean_anomaly: Mean anomaly of the planet
    - p: Orbital period of the planet
    - e: Eccentricity of the planet
    - a: Semi-major axis of the planet
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations

    Returns:
    - r: Orbital radius of the planet
    - f: True anomaly of the planet


    """
    # Initial guess for eccentric anomaly
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(max_iter):
        # Newton-Raphson iteration
        delta_e = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / \
                  (1 - e * np.cos(eccentric_anomaly))
        eccentric_anomaly -= delta_e

        # Check for convergence
        if np.all(np.abs(delta_e) < tol):
            break

    # true anomaly from Eccentric anomaly
    sin_f = np.sqrt(1 - e**2) * np.sin(eccentric_anomaly) / (1 - e * np.cos(eccentric_anomaly))
    cos_f = (np.cos(eccentric_anomaly) - e) / (1 - e * np.cos(eccentric_anomaly))
    f = np.arctan2(sin_f, cos_f)

    # Compute radius 
    r = a * (1 - e**2) / (1 + e * np.cos(f))

    return r, f


def n_body_sim_api(
    stellar_mass: float,
    planet_params: np.ndarray,
    times: np.ndarray,
    no_loading_bar: bool = True,
) -> np.ndarray:
    """
    Simulate the N-body system of a star and multiple planets over time.

    Parameters:
    - stellar_mass: Mass of the star
    - planet_params: 2D numpy array where each row represents
                     a planet's parameters as
                     [a, p, e, inc, omega, big_ohm, phase_lag, mass]

    Returns:
    - Array of body, positions of the star and planets across time. [time_idx, body_idx, coord]
    """
    sim = rebound.Simulation()
    sim.units = ["mearth", "day", "AU"]  # Set units to Earth masses, days, and AU

    planet_num = len(planet_params)
    sample_num = len(times)

    sim.add(m=stellar_mass)

    shortest_period = np.inf
    for params in planet_params:
        radius, mass, semi_major_axis, eccentricity, omega = params

        sim.add(m=mass, a=semi_major_axis, e=eccentricity, omega=omega)

        period = sim.particles[-1].P
        shortest_period = min(shortest_period, period)

    # +1 to include the star (particle 0)
    pos = np.empty((sample_num, planet_num + 1, 3), dtype=np.float64)

    for time_idx, time in enumerate(tqdm(times, disable=no_loading_bar)):
        sim.integrate(time)
        for j in range(planet_num + 1):  # Including the star (particle 0)
            pos[time_idx, :, :] = [[p.x, p.y, p.z] for p in sim.particles]

    return pos


@jit
def analytical_coordinate_generator(
    a: float,
    p: float,
    e: float,
    inc: float,
    omega: float,
    big_ohm: float,
    phase_lag: float,
    time_array: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generate the x, y, and z coordinates of a planet over time.

    Parameters:
    - eta, a, p, e, inc, omega, big_ohm, phase_lag: Planetary parameters
    - time_array: Array of time values in units of days

    Returns:
    - Arrays of x, y, and z coordinates of the planet over time
    """
    mean_anomaly = 2 * np.pi * (time_array / p) + phase_lag
    mean_anomaly = mean_anomaly % (2 * np.pi)  # Wrap to [0, 2π]

    # Solve Kepler's equation to get radius and true anomaly
    r, f = kepler_solver(
        mean_anomaly, p, e, a
    )

    x = r * (
        np.cos(big_ohm) * np.cos(omega + f)
        - np.sin(big_ohm) * np.sin(omega + f) * np.cos(inc)
    )
    y = r * (
        np.sin(big_ohm) * np.cos(omega + f)
        + np.cos(big_ohm) * np.sin(omega + f) * np.cos(inc)
    )
    z = -r * np.sin(omega + f) * np.sin(inc)

    return x, y, z


def analytical_positions_api(
    planet_params: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Simulate the N-body system of a star and multiple planets over time.

    Parameters:
    - planet_params: 2D numpy array where each row represents
                     a planet's parameters as
                     [a, p, e, inc, omega, big_ohm, phase_lag]
    - times: Array of time values

    Returns:
    - Array of positions of planets across time in shape [sample_num, planet_num, 3]
    """
    planet_num = planet_params.shape[0]
    sample_num = len(times)

    pos = np.empty((sample_num, planet_num, 3), dtype=np.float64)

    for i in range(planet_num):
        a, p, e, inc, omega, big_ohm, phase_lag = planet_params[i]
        x, y, z = analytical_coordinate_generator(
            a, p, e, inc, omega, big_ohm, phase_lag, times
        )
        pos[:, i, 0] = x
        pos[:, i, 1] = y
        pos[:, i, 2] = z

    return pos


if __name__ == "__main__":
    from sim.FluxCalculation import combined_delta_flux
    import matplotlib.pyplot as plt

    times = np.load("TestTimes.npy")
    radius_WASP148A = 0.912 * 696.34e6 / 1.496e11
    mass_WASP148A = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_WASP148A, mass_WASP148A]  # Based on WASP 148

    planet_params = np.array(
        [
            [
                0.1 + 0.001,
                0.08 + 0.001,
                8.8,
                0.208 - 0.0003,
                np.radians(88 + 0.002),
                0,
                0,
                0 + np.pi / 8,
            ],
            [0.1 + 0.1, 0.08 + 0.03, 8.8, 0.208 - 0.001, np.radians(89.5), 0, 0, 0],
        ]
    )
    positions = analytical_positions_api(planet_params=planet_params, times=times)
    print(positions.shape)
    flux_values = combined_delta_flux(
        x=positions[:, :, 0].transpose(),
        y=positions[:, :, 1].transpose(),
        z=positions[:, :, 2].transpose(),
        radius_star=stellar_params[0],
        eta_values=planet_params[:, 0],
        times=times,
    )

    plt.plot(times, flux_values)
    plt.show()
