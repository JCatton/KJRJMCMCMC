from MCMC.mcmc import MCMC
from MCMC.main import inclination_checker, prepare_arrays_for_mcmc
from sim.SimulateAndFlux import flux_data_from_params
from pathlib import Path
from typing import Callable
from TransitAnalysis.TransitDetector import search_for_transits
from TransitAnalysis.TransitDataExtractor import download_data
import numpy as np
import asyncio
import shutil

# from sim.ExampleSimulation import stellar_paramss

# Type Aliases
Params = list[list[float]]
Bounds = np.ndarray
Proposal = np.ndarray

def download_data_api(target_name: str, 
                  exptime: int = 120,
                  mission: str = "Tess",
                  sector: int = None,
                  author = None,
                  cadence = None,
                  max_number_downloads: int = 20,
                  use_regression_model = True):
    """
    Downloads data from the target_name

    Parameters:
    - target_name: String representing the target name
    - use_regression_model: Boolean representing whether to use the regression model

    Returns:
    - times: Array of time values
    - flux: Array of flux values
    """

    times, fluxes = download_data(target_name = target_name,
                                  exptime = exptime,
                                  mission = mission,
                                  sector = sector,
                                  author = author, 
                                  cadence = cadence,
                                  max_number_downloads = max_number_downloads)


    return times, fluxes


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
    # Stellar parameters: [radius, mass]
    radius_wasp148a = 0.912 * 696.34e6 / 1.496e11
    mass_wasp148a = 0.9540 * 2e30 / 6e24

    limb_darkening_model = "linear"
    limb_darkening_coefficients = [0]

    stellar_params = [radius_wasp148a, mass_wasp148a, limb_darkening_model, limb_darkening_coefficients]  # Based on WASP 148
    return stellar_params

def estimate_parameters(times: np.ndarray,
                        flux: np.ndarray,
                        stellar_params,
                        signal_detection_efficiency = 10,
                        period_min = None,
                        period_max = None
                        ) -> Params:
    """
    Estimates the parameters of the planets from the times and flux data

    Parameters:
    - times: np.ndarray of the times
    - flux: np.ndarray of the flux values
    - stellar_params: tuple of the stellar parameters

    Returns:
    - estimated_params: List of the estimated parameters [eta, a, P, e, inc, omega, big_ohm, phase_lag, mass]
    """
    stellar_radius = stellar_params[0]
    stellar_mass = stellar_params[1]
    limb_darkening_model = stellar_params[2]
    limb_darkening_coefficients = stellar_params[3]

    estimated_params = search_for_transits(times_input=times, 
                                           data=flux, 
                                           stellar_params=[stellar_radius, stellar_mass], 
                                           limb_darkening_model=limb_darkening_model, 
                                           limb_darkening_coefficients=limb_darkening_coefficients,
                                           signal_detection_efficiency=signal_detection_efficiency,
                                           plot_bool=True,
                                           save_loc=None,
                                           duration_multiplier=4,
                                           period_min = period_min,
                                           period_max = period_max)
    
    output_array = np.zeros((len(estimated_params), 9))

    for i in range(len(estimated_params)):
        output_array[i,0] = estimated_params[i]["eta"]
        output_array[i,1] = estimated_params[i]["a"]
        output_array[i,2] = estimated_params[i]["P"]
        output_array[i,3] = estimated_params[i]["e"]
        output_array[i,4] = estimated_params[i]["inc"]
        output_array[i,5] = estimated_params[i]["omega"]
        output_array[i,6] = estimated_params[i]["OHM"]
        output_array[i,7] = estimated_params[i]["phase_lag"]
        output_array[i,8] = 0  # Mass currently irrelevant
    
    return output_array

def estimate_proposal(times: np.ndarray, flux: np.ndarray) -> Proposal:
    return np.atleast_2d([
                    [5*1e-4, 2*1e-4, 2*1e-4, 0, 5*1e-4, 0, 0, 0, 0],  # Planet 1
                    # [1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 0, 0],   # Planet 2
                    ])


def estimate_noise(times: np.ndarray, flux: np.ndarray) -> float:
    return np.std(flux)

def estimate_bounds(times: np.ndarray, flux: np.ndarray) -> Bounds:
    return np.atleast_3d([
                    [(1e-5, 0.4), (1e-3, 0.5), (0, 1e4), (0, 0.3), (np.radians(80), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (-6,6), (0, 6000)],
                    # [(1e-5, 0.4), (1e-3, 0.5), (0, 1e10), (0, 0.3), (np.radians(86.8), np.pi), (-np.pi/8, np.pi/8), (-np.pi/8, np.pi/8), (-6, 6), (0, 6000)]
                    ])

def initial_param_fuzzer(initial_params: Params, proposal_std: Proposal, param_bounds: Bounds) -> Params:
    return initial_params

def generate_param_names(initial_parameters: Params) -> np.ndarray:
    depth, _ = initial_parameters.shape
    base_names = [r"\eta", "a", "P", "e", "inc", "omega", "big_ohm", "phase_lag", "mass"]
    return np.array([[name + f"_{obj_num}" for name in base_names] for obj_num in range(1, depth + 1)])


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

def run_mcmc_code(file: Path, target_search_params:list, target_stellar_params, iteration_num: int=50_000, run_number: int=3, analytic_sim:bool = True):
    """
    Run the MCMC code on the data

    Parameters:
    - file: Path to the file
    - target_search_params: List of parameters to search for the target [target_name, exptime, mission, sector, author, max_number_downloads, use_regression_model]
    - target_stellar_params: Tuple of the stellar parameters[stellar_radius, limb_darkening_model, limb_darkening_model,limb_darkening_coefficients]
    - iteration_num: Number of iterations to run
    - run_number: Number of runs to do
    - analytic_sim: Boolean representing whether to use the analytical

    Returns:
    - None


    """
    # times = np.load(file / 'times.npy')
    # flux = np.load(file / 'flux.npy')
    times, flux = download_data_api(*target_search_params)
    # stellar_params = get_stellar_params(file, target_name) # Todo -> Currently just give the regular stellar params
    stellar_params = target_stellar_params  # [radius, mas, limb_darkening_model, limb_darkening_coefficients] 
    #initial_params = estimate_parameters(times, flux, stellar_params, signal_detection_efficiency = 60, period_min=1, period_max=6)
    initial_params = np.atleast_2d([ 0.095751,  0.07806046,  3.5224991,  0.          ,np.radians(84),  0.,
   0.,         -3.6653389, 0])

    true_vals = np.array([
        [0.0875, 0.06608, 5.0037775, 0.05, np.radians(82.54), 0, 0, 0, 0],
    ])
    print(f"{initial_params=}, {initial_params.shape=}")
    proposal_std = estimate_proposal(times, flux) # Todo
    param_bounds = estimate_bounds(times, flux) # Todo
    noise = estimate_noise(times, flux) # Todo
    param_names = generate_param_names(initial_params)

    param_names, true_vals, initial_params, proposal_std, param_bounds = prepare_arrays_for_mcmc(param_names,
                                                                                                 true_vals,
                                                                                                 initial_params,
                                                                                                 proposal_std,
                                                                                                 param_bounds,
                                                                                                 analytic_sim)

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

    plt.figure()
    plt.title("Estimated Parameters Initial Fit")
    plt.plot(times, flux, label="Data")
    plt.plot(times, flux_data_from_params(stellar_params, initial_params, times, analytical_bool=True), label="Estimated", ls=":")
    plt.plot(times, flux_data_from_params(stellar_params, true_vals, times, analytical_bool=True), label="True", ls="--")
    plt.legend()
    plt.show()

    for i in range(run_number):
        mcmc = MCMC(
            flux,
            initial_param_fuzzer(initial_params, proposal_std, param_bounds),
            param_bounds,
            proposal_std,
            param_names=param_names,
            likelihood_func=likelihood_fn,
            inclination_rejection_func=lambda proposals: inclination_checker(proposals, stellar_params[0]),
            specified_folder_name = Path(file) / f"run_{i}",
            max_cpu_nodes=4,
        )
        mcmc.metropolis_hastings(iteration_num)
        mcmc.chain_to_plot_and_estimate()
        mcmc.corner_plot()

def main():
    data = download_data_api()  # Todo
    files = process_data(data)  # Todo

    [run_mcmc_code(file) for file in files]  # Todo



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # main()
    taget_name = 'TIC 147977348'
    exptime = None
    mission = None
    sector = None
    author = 'Kepler'
    cadence = 'long'
    max_number_downloads = 4
    use_regression_model = True
    target_search_params = [taget_name, exptime, mission, sector, author, cadence, max_number_downloads, use_regression_model]


    times, flux = download_data_api(*target_search_params)

    # plt.plot(times, flux)/
    print(f"Shapes of times and flux: {times.shape}, {flux.shape}")
    plt.plot(times, flux)
    plt.show()

    radius_tic_147977348 = 2.082 * 696.34e6 / 1.496e11
    mass_tic_147977348 = 1.536 * 2e30 / 6e24
    limb_darkening_model = "quadratic"
    limb_darkening_coefficients = [0.295, 0.312]

    stellar_params = [radius_tic_147977348, mass_tic_147977348, limb_darkening_model, limb_darkening_coefficients]  # Based on WASP 148

    period_min = 1
    period_max = 6
    signal_detection_efficiency = 60
    # estimated_params = estimate_parameters(times, flux, stellar_params, signal_detection_efficiency, period_min, period_max)
    # print(estimated_params)

    run_mcmc_code(file="test", target_search_params=target_search_params, target_stellar_params=stellar_params, iteration_num=150_000, run_number=1, analytic_sim=True)
