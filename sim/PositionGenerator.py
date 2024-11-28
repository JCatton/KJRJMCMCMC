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
        omega = omega/np.pi
        print(f"{omega=} $\pi$")
        pos[:, i, 0] = x
        pos[:, i, 1] = y
        pos[:, i, 2] = z

    return pos


if __name__ == "__main__":
    from sim.FluxCalculation import combined_delta_flux
    import matplotlib.pyplot as plt

    # Test case with HD 23472 -> Barros et al., 2022
    times = np.load("TestTimes.npy")

        # Stellar parameters: [radius, mass]
    radius_wasp148a = 0.912 * 696.34e6 / 1.496e11
    mass_wasp148a = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_wasp148a, mass_wasp148a]  # Based on WASP 148
    radius_wasp148_b = 8.47 * 6.4e6 / 1.496e11
    radius_wasp148_c = (
        9.4 * 6.4e6 / 1.496e11
    )  # assumed similar densities as no values for radius

    eta = 0

    theta = np.linspace(0, 2 * np.pi, 500)  # Parameterize the circle

    # Compute the x and y coordinates for the circle
    x_star = radius_wasp148a * np.cos(theta)
    y_star = radius_wasp148a * np.sin(theta)

    x_pi_on_2 = radius_wasp148a * np.cos(np.pi / 2)
    y_pi_on_2 = radius_wasp148a * np.sin(np.pi / 2)

        # [eta, a, p, e, inc, omega, big_ohm, phase_lag, mass (only for N body)]
    big_omega_list = np.linspace(0, 2 * np.pi, 9)
    a_list = np.linspace(0.1, 0.5, 5)
    a = 0.02044
    e = 0.1809
    phase_lag = 0
    small_omega = 0
    
    ecc_list = [0.1, 0.4, 0.8]
    ecc_list = [0.4]
    small_omega = [0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4, 7*np.pi/4]
    # small_omega = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    # small_omega = [0]
    big_omega_list = [0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4, 7*np.pi/4]
    # big_omega_list = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    # big_omega_list = [0]

    phase_lag = [0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4, 7*np.pi/4]
    # phase_lag = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    # phase_lag = [0]


    max = len(ecc_list) * len(small_omega) * len(big_omega_list) * len(phase_lag)
    
    points = (np.linspace(0, max, 50,dtype=int))


    count = 0 
    fig_count = 0
    for i in range(len(ecc_list)):
        for j in range(len(small_omega)):
            for k in range(len(big_omega_list)):
                for l in range(len(phase_lag)):
                    e = ecc_list[i]
                    r_pi_2 = a * (1 - e**2)
                    r_min = a * (1 - e**2)/(1+ +e* np.cos(3*np.pi/2 - small_omega[j]))
                    inc = np.arccos(radius_wasp148a * (1 + eta) / r_min)
                    # inc = np.radians(75)
                    print(inc/np.pi * 180)
                    planet_params = np.array(
                        [
                            [eta, a, 34.525, e, inc, small_omega[j], big_omega_list[k], phase_lag[l], 0.392]
                        ]
                    )
                    analyitical_positions = analytical_positions_api(planet_params[:, 1:-1], times)

                    rotated_x_pi_on_2 = x_pi_on_2 * np.cos(big_omega_list[k]) - y_pi_on_2 * np.sin(big_omega_list[k])
                    rotated_y_pi_on_2 = x_pi_on_2 * np.sin(big_omega_list[k]) + y_pi_on_2 * np.cos(big_omega_list[k])


                    r = a*(1-e**2)/(1+e*np.cos(np.pi/2))
                    x_at_point = r * np.cos(big_omega_list[k]) * np.cos(small_omega[j] + np.pi/2) - r * np.sin(big_omega_list[k]) * np.sin(small_omega[j] + np.pi/2) * np.cos(inc)
                    y_at_point = r * np.sin(big_omega_list[k]) * np.cos(small_omega[j] + np.pi/2) + r * np.cos(big_omega_list[k]) * np.sin(small_omega[j] + np.pi/2) * np.cos(inc)

                    r_little_after = a*(1-e**2)/(1+e*np.cos(np.pi/8))
                    x_little_after = r_little_after * np.cos(big_omega_list[k]) * np.cos(small_omega[j] + np.pi/8) - r_little_after * np.sin(big_omega_list[k]) * np.sin(small_omega[j] + np.pi/8) * np.cos(inc)
                    y_little_after = r_little_after * np.sin(big_omega_list[k]) * np.cos(small_omega[j] + np.pi/8) + r_little_after * np.cos(big_omega_list[k]) * np.sin(small_omega[j] + np.pi/8) * np.cos(inc)
                    


                    count += 1
                    x_start = analyitical_positions[0, 0, 0]
                    y_start = analyitical_positions[0, 0, 1]
                    if count in points:
                        plt.figure(fig_count)
                        plt.plot(analyitical_positions[:, 0, 0], analyitical_positions[:, 0, 1])
                        plt.plot(x_start, y_start, "ro", label="Start")
                        plt.plot(x_star, y_star, color="orange", label="Star")
                        plt.scatter(rotated_x_pi_on_2, rotated_y_pi_on_2, label="$\pi/4$", color="black")
                        plt.scatter(x_at_point, y_at_point, label="varied thingy", color="green")
                        plt.scatter(x_little_after, y_little_after, label="little After", color="blue")
                        plt.title(f"e: {e}, $\omega$: {small_omega[j]/np.pi} $\pi$, $\Omega$: {big_omega_list[k]/np.pi} $\pi$, $\phi$: {phase_lag[l]/np.pi} $\pi$")
                        plt.legend()
                        plt.xlim(-0.025, 0.025)
                        plt.ylim(-0.025, 0.025)
                        plt.show()
                        fig_count +=1 

                    print(count)
    print(count)
        
