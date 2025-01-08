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
def kepler_solver(mean_anomaly, p, e, a, tol=1e-9, max_iter=40):
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
                     [p, e, inc, omega, big_ohm, phase_lag, mass]

    Returns:
    - Array of body, positions of the star and planets across time. [time_idx, body_idx, coord]
    """
    sim = rebound.Simulation()
    num_steps_for_smallest_period = 50
    sim.units = ["mearth", "day", "AU"]  # Set units to Earth masses, days, and AU

    planet_num = len(planet_params)
    sample_num = len(times)

    sim.add(m=stellar_mass)

    shortest_period = np.min(planet_params[:,0])
    sim.dt = shortest_period / num_steps_for_smallest_period
    # Add planets to sim
    for params in planet_params:
        p, e, inc, omega, big_ohm, phase_lag, mass = params
        mean_anomaly = phase_lag % (2 * np.pi)
        sim.add(m=mass, P=p, e=e, inc = inc, omega=omega, Omega = big_ohm, M = mean_anomaly)

    sim.move_to_com()

    # +1 to include the star (particle 0)
    pos = np.empty((sample_num, planet_num + 1, 3), dtype=np.float64)
    for time_idx, time in enumerate(tqdm(times, disable=no_loading_bar)):
        sim.integrate(time)
        pos[time_idx, :, :] = [[p.x, p.y, -p.z] for p in sim.particles]
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

    # Test case with HD 23472 -> Barros et al., 2022
    times = np.load("TestTimes.npy")
    radius_HD23472 = 0.912 * 696.34e6 / 1.496e11
    mass_HD23472 = 0.67 * 2e30 / 6e24 # random radi for all vals

    stellar_params = [radius_HD23472, mass_HD23472]  # Based on WASP 148

    # planet_params =[ [ eta,   a,     P,   e,               inc, omega, OHM, phase_lag, mass ] ]
    planet_params = np.array(
        [
            [0.1, 0.04298, 3.97664, 0.0700, np.radians(90), 0, np.pi/2, 0, 0.55],
            [0.2, 0.0680, 7.90754, 0.0700, np.radians(90), 0, 0, np.pi/3, 0.72],
            [0.3, 0.0906, 12.1621839, 0.0700, np.radians(90), 0, 0, np.pi/2, 0.77],
            [0.4, 0.1162, 17.667087, 0.0720, np.radians(90), 0, 0, 2 * np.pi/3, 8.32],
            [0.5, 0.1646, 29.79749, 0.063, np.radians(90), 0, 0, 3/5 * np.pi, 3.41]
        ]
    )

    positions_analytical = analytical_positions_api(planet_params=planet_params[:, 1:-1], times=times)
    flux_values_analytical = combined_delta_flux(
        x=positions_analytical[:, :, 0].transpose(),
        y=positions_analytical[:, :, 1].transpose(),
        z=positions_analytical[:, :, 2].transpose(),
        radius_star=stellar_params[0],
        eta_values=planet_params[:, 0],
        times=times,
    )
    print(f"{positions_analytical.shape=}")
    # plt.show()

    positions_n_body = n_body_sim_api(
        stellar_mass=stellar_params[1],
        planet_params=planet_params[:, 2:],
        times=times,
    )
    x = positions_n_body[:, :, 0].transpose()
    y = positions_n_body[:, :, 1].transpose()
    z = positions_n_body[:, :, 2].transpose()
    print(f"{x.shape=}")
    print(f"{np.max(y[0])}")
    print(f"{np.max(y[1])}")
    # print(f"{np.max(y[2])}")
    x_s, y_s, z_s = x[0], y[0], z[0]
    x_p_rel = (x[1:] - x_s)
    y_p_rel = (y[1:] - y_s)
    z_p_rel = (z[1:] - z_s)
    print(f"{y_p_rel.shape=}")
    print(f"{np.max(y_p_rel[:,0])=}")
    # print(f"{np.max(y_p_rel[:,1])=}")
    # print(f"{np.max(y_p_rel[2])=}")

    flux_values_n_body = combined_delta_flux(
        x=x_p_rel,
        y=y_p_rel,
        z=z_p_rel,
        radius_star=stellar_params[0],
        eta_values=planet_params[:, 0],
        times=times,
    )
    print(f"{x_p_rel.shape=}")
    colors = ["r", "g", "b", "black", "purple"]
    for i in range(x_p_rel.shape[0]):
        plt.plot(x_p_rel[i, :], y_p_rel[i, :], label = f"Planet {i} n_body", color = colors[i], alpha = 0.5)
        plt.plot(x_p_rel[i, 0], y_p_rel[i, 0],  marker = "o", ms = 5, color = colors[i], alpha = 0.5)
        plt.plot(x_p_rel[i, 500], y_p_rel[i, 500],  marker = "x", ms = 5, color = colors[i], alpha = 0.5)
        plt.plot(x_p_rel[i, 1000], y_p_rel[i, 1000],  marker = "^", ms = 5, color = colors[i], alpha = 0.5)
    for i in range(positions_analytical.shape[1]):
        plt.plot(positions_analytical[:, i, 0], positions_analytical[:, i, 1], label = f"Planet {i} analytical, x,y", linestyle = "--", color = colors[i])
        plt.plot(positions_analytical[0, i, 0], positions_analytical[0, i, 1], marker = "o", ms = 5, color = colors[i])

        plt.plot(positions_analytical[500, i, 0], positions_analytical[500, i, 1], marker = "x", ms = 5, color = colors[i])
        plt.plot(positions_analytical[1000, i, 0], positions_analytical[1000, i, 1], marker = "^", ms = 5, color = colors[i])

    plt.title("Orbital Paths")
    plt.legend()
    plt.show()
    plt.plot(times, flux_values_analytical, label = "Analytical")
    plt.plot(times, flux_values_n_body, label = "N-Body", linestyle = "--")
    plt.legend()

    print(f"{np.max(a)=}, {np.min(a)=}")


    plt.show()
