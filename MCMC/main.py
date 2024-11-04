# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from data_generator import generate_linear_data
from sim.FluxCalculation import delta_flux_from_cartesian
from sim.SimulateAndFlux import flux_data_from_params

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

def add_gaussian_error(input_arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return input_arr + np.random.normal(mu, sigma, input_arr.shape)


def gaussian_error_ln_likelihood(observed: np.array, prior_funcs: list[Callable[..., float]],
                                 analytic_func: Callable[..., float],  params: np.array,
                                 sigma_n: float) -> float:
    if prior_funcs is not None:
        log_prior = np.sum(np.log([prior_funcs[i](params[i]) for i in range(len(params))]))
    else:
        log_prior = 0
    deviation_lh = 1/2 * np.log(sigma_n)
    observed_lh = np.power(observed - analytic_func(params), 2) / (2 * sigma_n ** 2)
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

def chain_to_plot_and_estimate(chain: np.ndarray, likelihoods: np.ndarray,
                               param_names: np.ndarray[str], true_vals: Optional[np.ndarray[float]] = None):
    print("MCMC sampling completed.")

    plt.figure(figsize=(10, 8))
    plt.xlabel("Iteration #")
    x = np.arange(len(chain))
    plt.plot(x, likelihoods)
    plt.ylabel(r"Log Likelihoods")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(nrows=chain.shape[2], ncols=chain.shape[1], figsize=(10, 8))
    axs = axs.reshape(chain[0].shape)
    fig.suptitle("Parameter Iterations")
    plt.xlabel("Iteration #")
    x = np.arange(len(chain))

    for body in range(chain.shape[1]):
        for i, name in enumerate(param_names):
            param_samples = chain[:, body, i]
            print(f"Estimated {name}: {np.mean(param_samples):.3e}",
                  f", true {name}: {true_vals[i]}" if true_vals is not None else None)
            axs[body, i].plot(x, param_samples, label=name)
            axs[body, i].set_ylabel(f"{name}")

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
    num_iterations = 1000
    return initial_params, num_iterations, param_bounds, proposal_std

def circular_delta_flux_function(x, eta, a, omega, phi):
    x = a * np.sin(omega * x + phi)
    y = a * np.cos(omega * x + phi)
    z = np.zeros(len(x))
    return delta_flux_from_cartesian(x, y, z, 1, eta)


def Kai_chain_to_plot_and_estimate_new(chain: np.ndarray, likelihoods: np.ndarray, params_labels: list[str]):
    """
    Extracts samples from an MCMC chain and plots the estimated parameters and log-likelihoods.

    Args:
        chain (np.ndarray): The MCMC chain of sampled parameters with shape (num_samples, num_params).
        likelihoods (np.ndarray): The log-likelihood values for each sample.
        params_labels (list[str]): List of parameter names.
    """
    num_params = len(params_labels)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
    fig.suptitle("Parameter Iterations")

    x = np.arange(len(chain))

    # Plot each parameter on its own subplot
    for i in range(3):
        param_samples = chain[:, i]
        axs[i,0].plot(x, param_samples)
        axs[i,0].set_ylabel(params_labels[i])
        axs[i,0].set_xlabel("Iteration")
        axs[i,0].set_title(f"Iteration for Parameter: {params_labels[i]}")

    for i in range(2):
        param_samples = chain[:, i + 3]
        axs[i,1].plot(x, param_samples)
        print(i, i + 3)
        axs[i,1].set_ylabel(params_labels[i + 3])
        axs[i,1].set_xlabel("Iteration")
        axs[i,1].set_title(f"Iteration for Parameter: {params_labels[i + 3]}")

    # Plotting the likelihoods on a separate subplot
    axs[-1,1].plot(x, likelihoods)
    axs[-1,1].set_ylabel(r"Log Likelihoods")
    axs[-1,1].set_xlabel("Iteration")
    axs[-1,1].set_title("Log-Likelihood Iterations")

    plt.tight_layout()
    plt.show()

    # Print summary for parameters
    print("MCMC sampling completed.")
    for i in range(num_params):
        if i < chain.shape[1]:
            print(f"Estimated {params_labels[i]}: {np.mean(chain[:, i]):.4f}")
        else:
            print(f"Estimated {params_labels[i]}: Constant value")



def main():

    # Generate synthetic data
    # times, inp_fluxes = extract_timeseries_data(r"C:\Users\jonte\PycharmProjects\KJRJMCMCMC\sim\Outputs\Example\timeseries_flux.npy")

    times = np.load("../Times.npy")
    inp_fluxes = np.load("../Fluxes.npy")

    # Initial parameter guesses
    param_names = np.array([r"\eta radius", "mass", "orbital radius", "eccentricity", r"\omega (phase)"])
    true_vals = np.array([0.1, 90.26, 0.045, 0.000, 0])

    initial_params = np.array([[0.1 + 0.001, 90.26, 0.045, 0, 0-0.0001]])
    proposal_std = np.array([3*1e-4, 0, 5*1e-7, 1e-5, 0])
    param_bounds = [(0,1), (0, 1e15), (1e-6, 1e5), (0, 0.99), (-np.pi, np.pi)]
    sigma_n = 2*1e-3
    fluxes = add_gaussian_error(inp_fluxes, 0, sigma_n)
    num_iterations = 2000

    # #Stellar params = [radius (in AU), mass]
    stellar_params = np.array([100 * 4.2635e-5, 333000 * 1.12])



    #Plot to check
    plt.subplot(2, 1, 1)
    plt.plot(times, inp_fluxes)
    plt.title("Original Data")
    plt.subplot(2, 1, 2)
    fluxes = add_gaussian_error(fluxes, 0, sigma_n)
    plt.plot(times, fluxes)
    plt.title("Data with Gaussian Noise")
    plt.show()



    def likelihood_fn(x, y, params):
        return gaussian_error_ln_likelihood(fluxes, None,
                                            lambda params: flux_data_from_params(stellar_params,
                                                                                 params,
                                                                                 times),
                                            params, sigma_n)

    from mcmc import MCMC
    mcmc = MCMC(fluxes, initial_params, param_bounds, proposal_std, likelihood_func=likelihood_fn)
    mcmc.metropolis_hastings(100)


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
    np.save("mcmc_likelihoods.npy", likelihoods)

    # Extract samples of parameters that aren't fixed
    non_fixed_indexes = np.array(proposal_std, dtype=bool)
    chain = chain[:, :, non_fixed_indexes]
    param_names = param_names[non_fixed_indexes]
    true_vals = true_vals[non_fixed_indexes]

    chain_to_plot_and_estimate(chain, likelihoods, param_names, true_vals=true_vals)

    burn_in_index = determine_burn_in_index(chain)
    print(
        f"\n\nBurn-in burn_in_index: {burn_in_index} "
        f"or {burn_in_index/chain.shape[0] * 100:.2f}% of the chain"
    )

    fig = corner(
        chain[burn_in_index:, 0],
        labels=param_names,
        truths=true_vals,
        show_titles=True,
        title_kwargs={"fontsize": 18},
        title_fmt=".2e"
    )
    plt.show()

    print("After Burn-in")
    chain_to_plot_and_estimate(chain[burn_in_index:], likelihoods[burn_in_index:],
                               param_names, true_vals=true_vals)


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
