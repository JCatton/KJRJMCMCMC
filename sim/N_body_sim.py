import rebound
import sim.FileCheck as fc
import numpy as np
from tqdm import tqdm
import os

def N_Body_sim(
    StellarMass, planet_params, SamplesPerOrbit=60, numberMaxPeriod=4, saveloc=None
):
    """
    Simulate the N-body system of a star and multiple planets over time.

    Parameters:
    - StellarMass: Mass of the star
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - SamplesPerOrbit: Number of samples per orbit for the shortest period planet
    - numberMaxPeriod: Number of periods to simulate
    - saveloc: Path to save the N-body outputs

    Returns:
    - Arrays of x, y, and z positions of the star and planets over time
    - Unperturbed x and y positions of the planets
    - Array of time values

    If saveloc is provided, the N-body outputs are saved as a .npz file
    """
    sim = rebound.Simulation()
    sim.units = ["mearth", "day", "AU"]  # Set units to Earth masses, days, and AU

    N = len(planet_params)

    # Generate lists for unperturbed orbits for each planet
    theta = np.linspace(0, 2 * np.pi, 100)
    orbits_x = np.empty((N, len(theta)))
    orbits_y = np.empty((N, len(theta)))

    LongestPeriod = 0
    ShortestPeriod = None

    sim.add(m=StellarMass)

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
        LongestPeriod = max(LongestPeriod, period)  # Update longest period
        ShortestPeriod = (
            min(ShortestPeriod, period) if ShortestPeriod is not None else period
        )  # Update shortest period

    # Ensure a the smallest orbit is sampled enough (30 times is suggested in Pearson 2019)
    timestep = 1 / SamplesPerOrbit * ShortestPeriod
    TotalTime = LongestPeriod * numberMaxPeriod
    Numbersteps = int(np.ceil(TotalTime / timestep))

    x_pos = np.empty((N + 1, Numbersteps))  # N+1 to include the star (particle 0)
    y_pos = np.empty((N + 1, Numbersteps))
    times = np.linspace(0, TotalTime, Numbersteps)

    # Integrate the simulation over time and save the positions
    for i, t in tqdm(enumerate(times), total=len(times)):
        sim.integrate(t)
        for j in range(N + 1):  # Including the star (particle 0)
            x_pos[j, i] = sim.particles[j].x
            y_pos[j, i] = sim.particles[j].y
        # Save the N-body outputs if saveloc is provided

    if saveloc:
        full_saveloc = fc.check_and_create_folder(saveloc)
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
    stellar_mass: float, planet_params: np.ndarray,
    times: np.ndarray
) -> (np.ndarray, np.ndarray):
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

        sim.add(
            m=mass, a=semi_major_axis, e=eccentricity, omega=omega
        )

        period = sim.particles[-1].P
        shortest_period = min(shortest_period, period)

    # +1 to include the star (particle 0)
    pos = np.empty((sample_num, planet_num + 1, 3), dtype=np.float64)

    for time_idx, time in enumerate(tqdm(times)):
        sim.integrate(time)
        for j in range(planet_num + 1):  # Including the star (particle 0)
            pos[time_idx, :, :] = [[p.x, p.y, p.z] for p in sim.particles]

    return pos
