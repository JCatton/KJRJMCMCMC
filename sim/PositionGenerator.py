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


def n_body_sim(
    stellar_mass: float,
    planet_params: np.ndarray,
    samples_per_orbit: int = 60,
    number_max_period: int = 4,
    save_loc: str = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Simulate the N-body system of a star and multiple planets over time.

    Parameters:
    - stellar_mass: Mass of the star
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - samples_per_orbit: Number of samples per orbit for the shortest period planet
    - number_max_period: Number of periods to simulate
    - save_loc: Path to save the N-body outputs

    Returns:
    - Arrays of x, y, and z positions of the star and planets over time
    - Unperturbed x and y positions of the planets
    - Array of time values

    If save_loc is provided, the N-body outputs are saved as a .npz file
    """
    sim = rebound.Simulation()
    sim.units = ["mearth", "day", "AU"]  # Set units to Earth masses, days, and AU

    N = len(planet_params)

    # Generate lists for unperturbed orbits for each planet
    theta = np.linspace(0, 2 * np.pi, 100)
    orbits_x = np.empty((N, len(theta)))
    orbits_y = np.empty((N, len(theta)))

    longest_period = 0
    shortest_period = None

    sim.add(m=stellar_mass)

    # Add each planet to the simulation
    for i, params in enumerate(planet_params):
        radius, mass, semi_major_axis, eccentricity, omega = params

        sim.add(
            m=mass, a=semi_major_axis, e=eccentricity, omega=omega
        )  # Add planet to simulation

        r = (semi_major_axis * (1 - eccentricity**2)) / (
            1 + eccentricity * np.cos(theta - omega)
        )  # Calculate unperturbed orbit
        x_orbit = r * np.cos(theta)
        y_orbit = r * np.sin(theta)
        orbits_x[i, :] = x_orbit
        orbits_y[i, :] = y_orbit

        period = sim.particles[-1].P  # Calculate period of planet
        longest_period = max(longest_period, period)  # Update longest period
        shortest_period = (
            min(shortest_period, period) if shortest_period is not None else period
        )  # Update shortest period

    # Ensure a the smallest orbit is sampled enough (30 times is suggested in Pearson 2019)
    time_step = 1 / samples_per_orbit * shortest_period
    total_time = longest_period * number_max_period
    number_steps = int(np.ceil(total_time / time_step))

    x_pos = np.empty((N + 1, number_steps))  # N+1 to include the star (particle 0)
    y_pos = np.empty((N + 1, number_steps))
    times = np.linspace(0, total_time, number_steps)

    # Integrate the simulation over time and save the positions
    for i, t in tqdm(enumerate(times), total=len(times)):
        sim.integrate(t)
        for j in range(N + 1):  # Including the star (particle 0)
            x_pos[j, i] = sim.particles[j].x
            y_pos[j, i] = sim.particles[j].y

    # Save the N-body outputs if save_loc is provided
    if save_loc:
        full_saveloc = fc.check_and_create_folder(save_loc)
        np.savez(
            os.path.join(full_saveloc, "N_body_outputs.npz"),
            x_pos=x_pos,
            y_pos=y_pos,
            orbits_x=orbits_x,
            orbits_y=orbits_y,
            times=times,
        )

    return x_pos, y_pos, orbits_x, orbits_y, times


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
    - planet_params: List of array planet parameters
                    [radius, mass, orbital radius, eccentricity, omega (phase)]

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
def GenerateCoordinates(
    eta: float,
    p: float,
    a: float,
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
    - eta, p, a, e, inc, omega, big_ohm, phase_lag: Planetary parameters
    - time_array: Array of time values

    Returns:
    - Arrays of x, y, and z coordinates of the planet over time
    """
    f = 2 * np.pi * ((time_array) / p + phase_lag)
    r = a * (1 - e**2) / (1 + e * np.cos(f))

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
                     [eta, p, a, e, inc, omega, big_ohm, phase_lag]
    - times: Array of time values

    Returns:
    - Array of positions of planets across time in shape [sample_num, planet_num, 3]
    """
    planet_num = planet_params.shape[0]
    sample_num = len(times)

    pos = np.empty((sample_num, planet_num, 3), dtype=np.float64)

    for i in range(planet_num):
        eta, p, a, e, inc, omega, big_ohm, phase_lag = planet_params[i]
        x, y, z = GenerateCoordinates(
            eta, p, a, e, inc, omega, big_ohm, phase_lag, times
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
                8.8,
                0.08 + 0.001,
                0.208 - 0.0003,
                np.radians(88 + 0.002),
                0,
                0,
                0 + np.pi / 8,
            ],
            [0.1 + 0.1, 8.8, 0.08 + 0.03, 0.208 - 0.001, np.radians(89.5), 0, 0, 0],
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
