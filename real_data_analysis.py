from MCMC.mcmc import MCMC
from MCMC.main import inclination_checker
from sim.SimulateAndFlux import flux_data_from_params
from pathlib import Path
from typing import Callable
import numpy as np
import asyncio
import shutil

from sim.ExampleSimulation import stellar_params

# Type Aliases
Params = list[list[float]]
Bounds = list[list[tuple[float, float]]]
Proposal = list[list[float]]


async def download_data() -> list[Path]:
    """
    Downloads data from api
    :return: list of file paths to downloaded data
    """
    pass


def process_data(data: list[Path]) -> list[Path]:
    """
    Processes data into the form of timeseries and flux_curve like that made by SimulateAndFlux.py
    :return: list of file paths to processed data
    """
    file_paths = []
    data_to_process = []
    for curve_data in data:
        if not curve_data.exists():
            print(f"Error: {curve_data} not found. Skipping")
        dirname = Path(curve_data.parent ,curve_data.stem)
        file_paths.append(dirname)# Removes file extension
        if not dirname.is_dir():
            dirname.mkdir(parents=True)
            curve_data.rename(dirname / curve_data.name)
            data_to_process.append(dirname)
        else:
            print(f"Error: {dirname} already exists. Skipping creation")

    # Data processing Todo


    return file_paths

def get_stellar_params(file: Path) -> tuple[float, float]:
    """
    Gets the stellar parameters stored somewhere in the file
    """
    pass

def estimate_parameters(times: np.ndarray, flux: np.ndarray) -> Params:
    pass

def estimate_proposal(times: np.ndarray, flux: np.ndarray) -> Proposal:
    pass

def estimate_noise(times: np.ndarray, flux: np.ndarray) -> float:
    pass

def estimate_bounds(times: np.ndarray, flux: np.ndarray) -> Bounds:
    pass

def initial_param_fuzzer(initial_params: Params, proposal_std: Proposal, param_bounds: Bounds) -> Params:
    return initial_params


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

def run_mcmc_code(file: Path, iteration_num: int=50_000, run_number: int=3, analytic_sim:bool = True):
    times = np.load(file / 'times.npy')
    flux = np.load(file / 'flux.npy')

    stellar_params = get_stellar_params(file) # Todo
    initial_params = estimate_parameters(times, flux) # Todo
    proposal_std = estimate_proposal(times, flux) # Todo
    param_bounds = estimate_bounds(times, flux) # Todo
    noise = estimate_noise(times, flux) # Todo

    def likelihood_fn(params):
        return gaussian_error_ln_likelihood(
            flux,
            None,
            lambda params: flux_data_from_params(
                stellar_params, params, times, analytical_bool=analytic_sim
            ),
            params,
            noise,
        )

    for i in range(run_number):
        mcmc = MCMC(
            flux,
            initial_param_fuzzer(initial_params, proposal_std, param_bounds),
            param_bounds,
            proposal_std,
            param_names=None,
            likelihood_func=likelihood_fn,
            inclination_rejection_func=lambda proposals: inclination_checker(proposals, stellar_params[0]),
            specified_folder_name = file / "run_{i}",
            max_cpu_nodes=4,
        )
        mcmc.metropolis_hastings(iteration_num)

def main():
    data = download_data()  # Todo
    files = process_data(data)  # Todo

    [run_mcmc_code(file) for file in files]  # Todo



if __name__ == "__main__":
    main()