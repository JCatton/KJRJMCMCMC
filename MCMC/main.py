# main.py
import sys
import os

import scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from MCMC.mcmc import Statistics
from MCMC.priors import Priors



from typing import Callable, List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from mcmc import MCMC
from sim.SimulateAndFlux import flux_data_from_params

# Global Configuration


def add_gaussian_error(input_arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return input_arr + np.random.normal(mu, sigma, input_arr.shape)


def gaussian_error_ln_likelihood(
    observed: np.array,
    prior_funcs: List[List[Callable[..., float]]],
    analytic_func: Callable[..., float],
    params: np.array,
    sigma_n: float,
) -> float:
    log_prior = 0

    if prior_funcs is not None:
        params = np.reshape(params, prior_funcs.shape)
        for p_fns, body_params in zip(prior_funcs, params):
                log_prior += np.sum(
                    np.log([p_fn(param) for p_fn, param in zip(p_fns, body_params) if p_fn])
                )
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

def prepare_arrays_for_mcmc(param_names, true_vals, initial_params, proposal_std, param_bounds,
                            analytical_bool, priors, prior_transform_funcs):
    if analytical_bool is None:
        raise ValueError("analytical_bool must be set to True or False")

    # Mask gets rid of mass or semi-major axis depending on analytical_bool
    n_body_mask = np.array([True, False, True, True, True, True, True, True, True])
    slc = (slice(None), slice(None, -1)) if analytical_bool else (slice(None), n_body_mask)

    param_names = param_names[slc]
    true_vals = true_vals[slc]
    initial_params = initial_params[slc]
    proposal_std = proposal_std[slc]
    param_bounds = param_bounds[slc]
    priors = priors[slc]
    prior_transform_funcs = prior_transform_funcs[slc]
    return param_names, true_vals, initial_params, proposal_std, param_bounds, priors, prior_transform_funcs


def inclination_checker(proposals: np.ndarray, r_star: float, indices: tuple[int, int, int, int, int] = (0, 1, 3, 5, 4)) -> bool:
    """
    Check if the inclinations of the planets are above the critical value.

    Parameters:
    - proposals: Array of proposals
    - indices: Tuple of indices (a_idx, e_idx, omega_idx, inc_idx)
    - r_star: Radius of the star

    Returns:
    - Boolean indicating if all inclinations are above the critical value
    """

    eta_idx, a_idx, e_idx, omega_idx, inc_idx = indices
    eta = proposals[0, :, eta_idx]
    a = proposals[0, :, a_idx]
    e = proposals[0, :, e_idx]
    omega = proposals[0, :, omega_idx]
    inc = proposals[0, :, inc_idx]

    # Calculate the critical inclination
    r = a * (1 - e**2) / (1 + e * np.cos(3* np.pi / 2 - omega))
    critical_inc = np.arccos((r_star*(1+eta)) / r)

    return np.all(inc >= critical_inc) # Return True if all inclinations are above the critical value




def prior_transform_calcs(priors: List[List[Optional[Dict]]],
                          param_bounds: List[List[Tuple]],
                          proposal_stds: List[List[float]],
                          initial_params: List[List[float]]) -> Tuple[np.ndarray[Callable],np.ndarray[Callable]]:
    prior_transforms = []
    prior_densities = []
    for body_idx, body_bounds in enumerate(param_bounds):
        prior_transforms.append([])
        prior_densities.append([])
        body = prior_transforms[body_idx]

        for param_idx, p_bounds in enumerate(body_bounds):
            body.append([None])
            prior_densities[body_idx].append([None])
            if proposal_stds[body_idx][param_idx] == 0:
                initial_param = initial_params[body_idx][param_idx]
                prior = Priors.get_dirac_prior(initial_param)
                # body[param_idx] = lambda x, initial_param=initial_param: dirac_delta_transform(initial_param, x)
            elif isinstance(priors[body_idx][param_idx], dict):
                prior = Priors.get_prior_by_config(priors[body_idx][param_idx])
            elif priors[body_idx][param_idx] is None:
                prior = Priors.get_uniform_prior(p_bounds[0], p_bounds[1])
                # body[param_idx] = lambda x, p0=p0, p1=p1: uniform_transform(p0, p1, x)
            body[param_idx]= prior.transform_func
            prior_densities[body_idx][param_idx] = prior.prior_func
    return np.array(prior_densities), np.array(prior_transforms)



def main():
    # Generate synthetic data
    # times, inp_fluxes = extract_timeseries_data(r"C:\Users\jonte\PycharmProjects\KJRJMCMCMC\sim\Outputs\Example\timeseries_flux.npy")

    times = np.load("../TestTimesMultiple.npy")
    inp_fluxes = np.load("../TestFluxesMultiple.npy")

    param_names = np.array([
        [r"\eta_1", "a_1", "P_1", "e_1", "inc_1", "omega_1", "big_ohm_1", "phase_lag_1", "mass_1"],
        [r"\eta_2", "a_2", "P_2", "e_2", "inc_2", "omega_2", "big_ohm_2", "phase_lag_2", "mass_2"]
    ])

    true_vals = np.array([
        [0.1, 0.08215, 8.803809, 0.208, np.radians(90), 0, 0, 0, 0.287],
        [0.3, 0.2044, 34.525, 0.1809, np.radians(90), 0, 0, np.pi / 4, 0.392]
    ])
    initial_params = np.array([
        [0.1+0.05, 0.08215-0.003, 8.803809 - 0.02, 0.208- 0.03, np.radians(90), 0, 0, 0, 0.287],
        [0.3+0.1, 0.2044 + 0.003, 34.525+0.002, 0.1809 + 0.007, np.radians(90), 0, 0, np.pi / 4 + np.pi/100, 0.392]
    ])

    proposal_std = np.array([
        [1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 0, 0],  # Planet 1
        [1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 0, 0],   # Planet 2
    ])


    analytical_bool = True
    param_bounds = np.array([
        [(0.05, 0.15), (0.04, 0.2), (0, 1e10), (0, 0.3), (np.radians(86.8), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (0, 6000)],
        [(0.2, 0.4), (0.08, 0.3), (0, 1e10), (0, 0.3), (np.radians(86.8), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (0, np.pi/2), (0, 6000)]
    ])
    priors = [
        [{"distribution": "gaussian", "lower_bound": 0.05, "upper_bound":0.15, "mean": 0.1, "std":5*1e-2}, None, None, None, None, None, None, None, None],
        [{"distribution": "gaussian", "lower_bound": 0.1, "upper_bound":0.5, "mean": 0.3, "std":5*1e-2}, None, None, None, None, None, None, None, None]
    ]

    priors, prior_transform_funcs = prior_transform_calcs(priors, param_bounds, proposal_std, initial_params)

    (param_names, true_vals, initial_params, proposal_std,
     param_bounds, priors, prior_transform_funcs) = prepare_arrays_for_mcmc(param_names,
                                                                             true_vals,
                                                                             initial_params,
                                                                             proposal_std,
                                                                             param_bounds,
                                                                             analytical_bool,
                                                                             priors,
                                                                             prior_transform_funcs)


    print(param_names.shape, true_vals.shape, initial_params.shape, proposal_std.shape, param_bounds.shape)
    sigma_n = 1e-3
    fluxes = add_gaussian_error(inp_fluxes, 0, sigma_n)
    num_iterations = int(50_000)

    radius_wasp148_a = 0.912 * 696.34e6 / 1.496e11
    mass_wasp_a = 0.9540 * 2e30 / 6e24

    stellar_params = [radius_wasp148_a, mass_wasp_a]  # Based on WASP 148

    # Plot to check
    plt.subplot(2, 1, 1)
    plt.plot(times, inp_fluxes)
    plt.title("Original Data")
    plt.subplot(2, 1, 2)
    fluxes = add_gaussian_error(fluxes, 0, sigma_n)
    plt.plot(times, fluxes)
    plt.title("Data with Gaussian Noise")
    plt.show()


    r_star = stellar_params[0]  # Stellar radius


    def likelihood_fn(params):
        return gaussian_error_ln_likelihood(
            fluxes,
            priors,
            lambda params: flux_data_from_params(
                stellar_params, params, times, analytical_bool=analytical_bool
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
        inclination_rejection_func=lambda proposals: inclination_checker(proposals, r_star),
        max_cpu_nodes=8,
        prior_transforms=prior_transform_funcs,
    )

    # mcmc.nested_sampling()
    mcmc.metropolis_hastings(num_iterations)
    mcmc.chain_to_plot_and_estimate(true_vals)
    mcmc.corner_plot(true_vals)

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
    # folder_names = [
    #     "2024-11-06_run1",
    #     "2024-11-06_run2",
    #     "2024-11-06_run3",
    #     "2024-11-06_run4",
    #     "2024-11-06_run5",
    # ]
    # stats = Statistics(folder_names)
    # for mcmc in stats.loaded_mcmcs:
        # mcmc.corner_plot(true_vals=true_vals, burn_in_index=1000)
        # mcmc.chain_to_plot_and_estimate(true_vals=true_vals)
    # gr = stats.calc_gelman_rubin()
    # print(f"The Gelman Rubin Statistic is {gr}")


if __name__ == "__main__":
    main()
