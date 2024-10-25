# main.py

from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from data_generator import generate_linear_data
from metropolis_hastings import determine_burn_in_index, metropolis_hastings
from sim.FluxCalculation import delta_flux_from_cartesian

# Global Configuration
# Linear
m_true = 2.5
c_true = 1.0
m_noise_true = 3.0
c_noise_true = 1.0
# Sin
a_true = 1.5
a_noise_true = 0.5
b_true = 10
b_noise_true = 0.5


def linear_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
) -> float:
    """
    Computes the log-likelihood for the linear model y = m x + c,
    where m and c have their own Gaussian uncertainties.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        params (np.ndarray): Model parameters [m_true, c_true,
                                                log_m_noise, log_c_noise].

    Returns:
        float: Log-likelihood value.
    """
    m_true, c_true, log_m_noise, log_c_noise = params
    m_noise = np.exp(log_m_noise)
    c_noise = np.exp(log_c_noise)
    mu = m_true * x + c_true  # Expected mean of y_i
    sigma2 = x**2 * m_noise**2 + c_noise**2  # Variance of y_i
    residuals = y - mu
    # Avoid division by zero or negative variances
    if np.any(sigma2 <= 0):
        return -np.inf
    log_likelihood = -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))
    return log_likelihood

def gaussian_error_ln_likelihood(observed: np.array, prior_funcs: list[Callable[..., float]],
                                 analytic_func: Callable[..., float],  params: np.array,
                                 sigma_n: float) -> float:
    if prior_funcs is not None:
        log_prior = np.sum(np.log([prior_funcs[i](params[i]) for i in range(len(params))]))
    else:
        log_prior = 0
    deviation_lh = 1/2 * np.log(sigma_n)
    observed_lh = np.power(observed - analytic_func(*params), 2) / (2 * sigma_n ** 2)
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

def chain_to_plot_and_estimate(chain: np.ndarray, likelihoods: np.ndarray):
    m_samples = chain[:, 0]
    c_samples = chain[:, 1]
    m_noise_samples = np.exp(chain[:, 2])
    c_noise_samples = np.exp(chain[:, 3])

    # Print summary
    print("MCMC sampling completed.")
    print(f"Estimated m: {np.mean(m_samples):.4f}, true m: {m_true}")
    print(f"Estimated c: {np.mean(c_samples):.4f}, true c: {c_true}")
    print(
        f"Estimated m_noise: {np.mean(m_noise_samples):.4f},"
        f" true m_noise: {m_noise_true}"
    )
    print(
        f"Estimated c_noise: {np.mean(c_noise_samples):.4f},"
        f" true c_noise: {c_noise_true}"
    )

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 8))
    x = np.arange(len(chain))
    fig.suptitle("Parameter Iterations")
    axs[0].plot(x, m_samples)
    axs[0].set_ylabel("eta")
    axs[1].plot(x, c_samples)
    axs[1].set_ylabel("a")
    axs[2].plot(x, m_noise_samples)
    axs[2].set_ylabel(r"$\omega$")
    axs[3].plot(x, c_noise_samples)
    axs[3].set_ylabel(r"$\phi$")
    axs[4].plot(x, likelihoods)
    axs[4].set_ylabel(r"Log Likelihoods")
    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.show()


def synthetic_data(
    linear: bool, num_data_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    x_data = np.linspace(0, 10, num_data_points)
    if linear:
        linear_params = np.array([m_true, c_true])
        linear_noise = np.array([m_noise_true, c_noise_true])
        x_data, y_data = generate_linear_data(x_data, linear_params, linear_noise)
    else:
        pass
    return x_data, y_data


def mcmc_initialisation(bounded: bool = False):
    initial_params = np.array(
        [
            0.0,  # Initial guess for m_true
            0.0,  # Initial guess for c_true
            np.log(1.0),  # Initial guess for log_m_noise
            np.log(1.0),  # Initial guess for log_c_noise
        ]
    )
    # Parameter bounds [(lower, upper), ...]
    if bounded:
        param_bounds = [
            (-10, 10),  # Bounds for m_true
            (-10, 10),  # Bounds for c_true
            (np.log(1e-3), np.log(1e3)),  # Bounds for log_m_noise
            (np.log(1e-3), np.log(1e3)),  # Bounds for log_c_noise
        ]
    else:
        param_bounds = [(-np.inf, np.inf) for _ in initial_params]
    proposal_std = np.array([0.5, 0.5, 0.1, 0.1])
    num_iterations = 20000
    return initial_params, num_iterations, param_bounds, proposal_std

def circular_delta_flux_function(x, eta, a, omega, phi):
    x = a * np.sin(omega * x + phi)
    y = a * np.cos(omega * x + phi)
    z = np.zeros(len(x))
    return delta_flux_from_cartesian(x, y, z, 1, eta)


def main():

    # Generate synthetic data
    times, fluxes = extract_timeseries_data(r"C:\Users\jonte\Documents\Coding\Python\KJRJMCMCMC\sim\Outputs\Example\timeseries_flux.npy")

    # Initial parameter guesses
    initial_params = [1e-2, 0.01, 1, 1] # eta, A, omega, phi
    sigma_n = 1e-2
    num_iterations = 20000
    param_bounds = [(0,1), (1, 1e8), (0, 1), (-np.pi, np.pi)]
    proposal_std = np.array([1e-5, 1e-3, 1e-3, 1e-2])


    def likelihood_fn(x, y, params):
        return gaussian_error_ln_likelihood(fluxes, None,
                                            lambda eta, a,o,p: circular_delta_flux_function(times, eta, a, o, p),
                                            params, sigma_n)

    # Run MCMC
    chain, likelihoods = metropolis_hastings(
        x=times,
        y=fluxes,
        initial_params=initial_params,
        param_bounds=param_bounds,
        likelihood_fn=likelihood_fn,
        proposal_std=proposal_std,
        num_iterations=num_iterations,
    )

    # Save chain
    np.save("mcmc_chain.npy", chain)

    # Extract samples
    chain_to_plot_and_estimate(chain, likelihoods)

    burn_in_index = determine_burn_in_index(chain)
    print(
        f"\n\nBurn-in burn_in_index: {burn_in_index} "
        f"or {burn_in_index/chain.shape[0] * 100:.2f}% of the chain"
    )
    fig = corner(
        chain[burn_in_index:],
        labels=[
            r"$eta$",
            r"$a$",
            r"$omega$",
            r"$phi$",
        ],
        truths=[m_true, c_true, m_noise_true, c_noise_true],
        show_titles=True,
        title_kwargs={"fontsize": 18},
    )
    fig.suptitle(
        f"A plot of y=mx + c,\nFor m~N({m_true}, {m_noise_true}),"
        f" and c~N({c_true}, {c_noise_true})"
    )
    plt.show()

    print("After Burn-in")
    chain_to_plot_and_estimate(chain[burn_in_index:])


# def main():
#
#     # Generate synthetic data
#     x_data, y_data = synthetic_data()
#
#     # Initial parameter guesses
#     initial_params, num_iterations, param_bounds, proposal_std = mcmc_initialisation(
#         bounded=True
#     )
#
#     def likelihood_fn(x, y, params):
#         return linear_likelihood(x, y, params)
#
#     # Run MCMC
#     chain = metropolis_hastings(
#         x=x_data,
#         y=y_data,
#         initial_params=initial_params,
#         param_bounds=param_bounds,
#         likelihood_fn=likelihood_fn,
#         proposal_std=proposal_std,
#         num_iterations=num_iterations,
#     )
#
#     # Save chain
#     np.save("mcmc_chain.npy", chain)
#
#     # Extract samples
#     chain_to_plot_and_estimate(chain)
#
#     burn_in_index = determine_burn_in_index(chain)
#     print(
#         f"\n\nBurn-in burn_in_index: {burn_in_index} "
#         f"or {burn_in_index/chain.shape[0] * 100:.2f}% of the chain"
#     )
#     fig = corner(
#         chain[burn_in_index:],
#         labels=[
#             r"$m$",
#             r"$c$",
#             r"$\sigma_m$",
#             r"$\sigma_c$",
#         ],
#         truths=[m_true, c_true, m_noise_true, c_noise_true],
#         show_titles=True,
#         title_kwargs={"fontsize": 18},
#     )
#     fig.suptitle(
#         f"A plot of y=mx + c,\nFor m~N({m_true}, {m_noise_true}),"
#         f" and c~N({c_true}, {c_noise_true})"
#     )
#     plt.show()
#
#     print("After Burn-in")
#     chain_to_plot_and_estimate(chain[burn_in_index:])


if __name__ == "__main__":
    main()
