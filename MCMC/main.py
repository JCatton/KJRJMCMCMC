# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from MCMC.mcmc import Statistics



from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

# from mcmc import MCMC
from sim.SimulateAndFlux import flux_data_from_params

# Global Configuration


def add_gaussian_error(input_arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return input_arr + np.random.normal(mu, sigma, input_arr.shape)


def gaussian_error_ln_likelihood(
    observed: np.array,
    prior_funcs: list[Callable[..., float]],
    analytic_func: Callable[..., float],
    params: np.array,
    sigma_n: float,
) -> float:
    if prior_funcs is not None:
        log_prior = np.sum(
            np.log([prior_funcs[i](params[i]) for i in range(len(params))])
        )
    else:
        log_prior = 0
    deviation_lh = 1 / 2 * np.log(sigma_n)
    observed_lh = np.power(observed - analytic_func(params), 2) / (2 * sigma_n**2)
    ln_likelihood = log_prior - deviation_lh - np.sum(observed_lh)
    return ln_likelihood


def extract_timeseries_data(file_location: str) -> (np.ndarray, np.ndarray):
    """
    Returns the times and the values at those times of the data.
    :param file_location: str
    :return: Tuple(np.ndarray, np.ndarray)
    """
    timeseries = np.load(file_location, allow_pickle=True)
    return timeseries[0], timeseries[1]


def main():

    # Generate synthetic data
    # times, inp_fluxes = extract_timeseries_data(r"C:\Users\jonte\PycharmProjects\KJRJMCMCMC\sim\Outputs\Example\timeseries_flux.npy")

    times = np.load("TestTimesMultiple.npy")
    inp_fluxes = np.load("TestFluxesMultiple.npy")

    # #Stellar params = [radius (in AU), mass]
    # stellar_params = np.array([100 * 4.2635e-5, 333000 * 1.12])
    # Initial parameter guesses
    # param_names = np.array([r"\eta radius", "mass", "orbital radius", "eccentricity", r"\omega (phase)"])
    # true_vals = np.array([0.1, 90.26, 0.045, 0.000, 0])

    # initial_params = np.array([[0.1 + 0.001, 90.26, 0.045, 0, 0-0.0001]])
    # proposal_std = np.array([3*1e-4, 0, 5*1e-7, 1e-5, 0])
    # param_bounds = [(0,1), (0, 1e15), (1e-6, 1e5), (0, 0.99), (-np.pi, np.pi)]

    # planet_params =[ [ eta,   P,     a,   e,               inc, omega, OHM, phase_lag ] ]
    # planet_params =  np.array([[  eta1, 8.8, 0.08, 0.208, np.radians(90),   0, 0,  0]
    param_names = np.array([
        [r"\eta_1", "P_1", "a_1", "e_1", "inc_1", "omega_1", "OHM_1", "phase_lag_1"],
        [r"\eta_2", "P_2", "a_2", "e_2", "inc_2", "omega_2", "OHM_2", "phase_lag_2"]
    ])

    true_vals = np.array([
        [0.1, 8.8, 0.08, 0.208, np.radians(90), 0, 0, 0],
        [0.3, 12, 0.101, 0.1809, np.radians(90), 0, 0, np.pi / 4]
    ])
    initial_params = np.array([
        [0.1+0.025, 8.8, 0.08, 0.208, np.radians(90), 0, 0, 0],
        [0.3+0.025, 12, 0.101, 0.1809, np.radians(90), 0, 0, np.pi / 4]
    ])

    proposal_std = np.array([
        [3e-5, 5e-4, 5e-6, 1e-6, 0, 4e-5, 0, 4e-6],  # Planet 1
        [3e-5, 5e-4, 5e-6, 1e-6, 0, 4e-5, 0, 4e-6],   # Planet 2
    ])

    param_bounds = [
        [(0.05, 0.25), (0, 1e1000), (0.04, 0.2), (0, 0.3), (np.radians(86.8), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8)],
        [(0.2, 0.4), (0, 1e1000), (0.08, 0.18), (0, 0.3), (np.radians(86.8), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (0, np.pi/2)]
    ]


    sigma_n = 6 * 1e-4
    fluxes = add_gaussian_error(inp_fluxes, 0, sigma_n)
    num_iterations = int(1_000_000)

    radius_WASP148A = 0.912 * 696.34e6 / 1.496e11
    mass_WASP148A = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_WASP148A, mass_WASP148A]  # Based on WASP 148

    # Plot to check
    plt.subplot(2, 1, 1)
    plt.plot(times, inp_fluxes)
    plt.title("Original Data")
    plt.subplot(2, 1, 2)
    fluxes = add_gaussian_error(fluxes, 0, sigma_n)
    plt.plot(times, fluxes)
    plt.title("Data with Gaussian Noise")
    plt.show()

    def likelihood_fn(params):
        return gaussian_error_ln_likelihood(
            fluxes,
            None,
            lambda params: flux_data_from_params(
                stellar_params, params, times, analytical_bool=True
            ),
            params,
            sigma_n,
        )

    from mcmc import MCMC

    mcmc = MCMC(
        fluxes,
        initial_params,
        param_bounds,
        proposal_std,
        param_names=param_names,
        likelihood_func=likelihood_fn,
        max_cpu_nodes=1,
    )

    mcmc.metropolis_hastings(num_iterations)
    mcmc.chain_to_plot_and_estimate(true_vals)
    mcmc.corner_plot(true_vals, burn_in_index=350_000)

    plt.title("Difference between true and estimated fluxes")
    plt.xlabel("Time")
    plt.ylabel("Difference in Fluxes")
    plt.plot(
        times,
        flux_data_from_params(
            stellar_params, mcmc.chain[-1], times, analytical_bool=True
        )
        - flux_data_from_params(
            stellar_params, true_vals, times, analytical_bool=True
        ),
    )
    plt.show()


    plt.title("True and estimated fluxes")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.plot(
        times,
        flux_data_from_params(
            stellar_params, mcmc.chain[-1], times, analytical_bool=True
        ),
        label="Estimated",
    )
    plt.plot(
        times,
        flux_data_from_params(
            stellar_params, true_vals, times, analytical_bool=True
        ),
        label="True",
    )
    plt.legend()
    plt.show()
    print(mcmc.acceptance_num)
    # mcmc = MCMC(fluxes, initial_params, param_bounds, proposal_std,
    #             param_names=param_names, likelihood_func=likelihood_fn, max_cpu_nodes=4)
    # mcmc.metropolis_hastings(50_000)
    # mcmc.chain_to_plot_and_estimate(true_vals)
    # mcmc.corner_plot(true_vals)
    folder_names = [
        "2024-11-06_run1",
        "2024-11-06_run2",
        "2024-11-06_run3",
        "2024-11-06_run4",
        "2024-11-06_run5",
    ]
    stats = Statistics(folder_names)
    for mcmc in stats.loaded_mcmcs:
        # mcmc.corner_plot(true_vals=true_vals, burn_in_index=1000)
        mcmc.chain_to_plot_and_estimate(true_vals=true_vals)
    # gr = stats.calc_gelman_rubin()
    # print(f"The Gelman Rubin Statistic is {gr}")


if __name__ == "__main__":
    main()
