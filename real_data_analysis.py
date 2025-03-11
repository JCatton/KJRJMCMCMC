from MCMC.mcmc import MCMC
from MCMC.main import inclination_checker, prepare_arrays_for_mcmc, add_gaussian_error, prior_transform_calcs
from sim.SimulateAndFlux import flux_data_from_params
from pathlib import Path
from typing import Callable
from TransitAnalysis.TransitDetector import search_for_transits
from TransitAnalysis.TransitDataExtractor import download_data
import numpy as np
import asyncio
import shutil
from sim.Decorators import TimeMeasure

# from sim.ExampleSimulation import stellar_paramss

# Type Aliases
Params = list[list[float]]
Bounds = np.ndarray
Proposal = np.ndarray


def download_data_api(
    target_name: str,
    exptime: int = 120,
    mission: str = "Tess",
    sector: int = None,
    author=None,
    cadence=None,
    indicies_requested=None,
    max_number_downloads: int = 20,
    use_regression_model=True,
    use_lightcurve_direct=False,
):
    """
    Downloads data from the target_name

    Parameters:
    - target_name: String representing the target name
    - use_regression_model: Boolean representing whether to use the regression model

    Returns:
    - times: Array of time values
    - flux: Array of flux values
    """

    times, fluxes = download_data(
        target_name=target_name,
        exptime=exptime,
        mission=mission,
        sector=sector,
        author=author,
        cadence=cadence,
        indicies_requested=indicies_requested,
        max_number_downloads=max_number_downloads,
        apply_regressor_bool=use_regression_model,
        use_lightcurve_direct=use_lightcurve_direct,
    )

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
        dirname = Path(curve_data.parent, curve_data.stem)
        file_paths.append(dirname)  # Removes file extension
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

    stellar_params = [
        radius_wasp148a,
        mass_wasp148a,
        limb_darkening_model,
        limb_darkening_coefficients,
    ]  # Based on WASP 148
    return stellar_params


def estimate_parameters(
    times: np.ndarray,
    flux: np.ndarray,
    stellar_params,
    signal_detection_efficiency=10,
    period_min=None,
    period_max=None,
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
    if limb_darkening_model == 2:
        limb_darkening_model = "quadratic"
    elif limb_darkening_model == 1:
        limb_darkening_model = "linear"
    else:
        print("Jonte tf have you done")

    limb_darkening_coefficients = [stellar_params[3], stellar_params[4]]

    estimated_params = search_for_transits(
        times_input=times,
        data=flux,
        stellar_params=[stellar_radius, stellar_mass],
        limb_darkening_model=limb_darkening_model,
        limb_darkening_coefficients=limb_darkening_coefficients,
        signal_detection_efficiency=signal_detection_efficiency,
        plot_bool=True,
        save_loc=None,
        duration_multiplier=4,
        period_min=period_min,
        period_max=period_max,
    )

    output_array = np.zeros((len(estimated_params), 9))

    for i in range(len(estimated_params)):
        output_array[i, 0] = estimated_params[i]["eta"]
        output_array[i, 1] = estimated_params[i]["a"]
        output_array[i, 2] = estimated_params[i]["P"]
        output_array[i, 3] = estimated_params[i]["e"]
        output_array[i, 4] = estimated_params[i]["inc"]
        output_array[i, 5] = estimated_params[i]["omega"]
        output_array[i, 6] = estimated_params[i]["OHM"]
        output_array[i, 7] = estimated_params[i]["phase_lag"]
        output_array[i, 8] = 0  # Mass currently irrelevant

    return output_array


def extend_params_for_stellar(planet_params: Params, stellar_params: list[float]) -> Params:
    """
    Merge the planet and stellar parameters into one array for the MCMC code

    Parameters:
    - planet_params: List of the planet parameters [eta, a, P, e, inc, omega, big_ohm, phase_lag, mass]
    - stellar_params: List of the stellar parameters [radius, mass, limb_darkening_model, limb_darkening_coefficients]

    Returns:
    - output_array: List of the merged parameters
    """
    output_array = np.zeros((planet_params.shape[0] + 1, 8))
    output_array[0, 0] = stellar_params[0]
    output_array[0, 1] = stellar_params[1]
    output_array[0, 2] = stellar_params[2]
    output_array[0, 3] = stellar_params[3]
    output_array[0, 4] = stellar_params[4]

    for i in range(1, planet_params.shape[0] + 1):
        output_array[i, 0] = planet_params[i-1, 0]
        output_array[i, 1] = planet_params[i-1, 1]
        output_array[i, 2] = planet_params[i-1, 2]
        output_array[i, 3] = planet_params[i-1, 3]
        output_array[i, 4] = planet_params[i-1, 4]
        output_array[i, 5] = planet_params[i-1, 5]
        output_array[i, 6] = planet_params[i-1, 6]
        output_array[i, 7] = planet_params[i-1, 7]
        # output_array[i, 8] = 0  # Mass currently irrelevant

    return output_array


def estimate_proposal(times: np.ndarray, flux: np.ndarray) -> Proposal:
    return np.atleast_2d(
        [
            [6*1e-4, 1e-3, 0, 1e-4, 0, 0, 0, 0, 0],  # Planet 1
            [6*1e-4, 1e-3, 0, 1e-4, 0, 0, 0, 0, 0],   # Planet 2
        ]
    )


def estimate_noise(times: np.ndarray, flux: np.ndarray) -> float:
    return np.std(flux)


def estimate_bounds(times: np.ndarray, flux: np.ndarray) -> Bounds:
    return np.atleast_3d(
        [
            [
                (0.07, 0.4),
                (1e-3, 0.5),
                (0, 1e4),
                (0, 0.3),
                (np.radians(70), np.radians(110)),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-6, 6),
                (0, 6000),
            ],
            [
                (0.01, 0.07),
                (1e-3, 0.5),
                (0, 1e4),
                (0, 0.3),
                (np.radians(70), np.radians(110)),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-6, 6),
                (0, 6000),
            ],
        ]
    )


def initial_param_fuzzer(
    initial_params: Params, proposal_std: Proposal, param_bounds: Bounds
) -> Params:
    return initial_params


def generate_param_names(initial_parameters: Params) -> np.ndarray:
    depth, _ = initial_parameters.shape
    base_names = [
        r"\eta",
        "a",
        "P",
        "e",
        "inc",
        "omega",
        "big_ohm",
        "phase_lag",
        "mass",
    ]
    names = np.array(
        [
            [name + f"_{obj_num}" for name in base_names]
            for obj_num in range(1, depth + 1)
        ]
    )

    return names

def extend_names_for_stellar(names: np.ndarray) -> np.ndarray:
    new_names = np.zeros((names.shape[0] + 1, names.shape[1]), dtype=object)
    new_names[0,0] = "stellar_radius"
    new_names[0,1] = "stellar_mass"
    new_names[0,2] = "limb_darkening_model"
    new_names[0,3] = "limb_darkening_coefficient_1"
    new_names[0,4] = "limb_darkening_coefficient_2"
    new_names[0,5:] = "Empty"
    new_names[1:] = names

    return new_names

def extend_proposal_for_stellar(proposal: Proposal) -> Proposal:

    new_proposal = np.zeros((proposal.shape[0] + 1, proposal.shape[1]))
    new_proposal[0,0] = 0
    new_proposal[0,1] = 0
    new_proposal[0,2] = 0
    new_proposal[0,3] = 5e-4  # 5*1e-4
    new_proposal[0,4] = 5e-4  # 5*1e-4
    new_proposal[0,5:] = 0
    new_proposal[1:] = proposal
    

    return new_proposal

def extend_param_bounds_for_stellar(param_bounds: Bounds) -> Bounds:

    new_param_bounds = np.zeros((param_bounds.shape[0] + 1, param_bounds.shape[1], param_bounds.shape[2]))

    new_param_bounds[0,0] = (1e-4, 2)
    new_param_bounds[0,1] = (1e5, 10e7)
    new_param_bounds[0,2] = (0, 1e3)
    new_param_bounds[0,3] = (-1, 1)
    new_param_bounds[0,4] = (-1, 1)
    new_param_bounds[0,5:] = (0, 5)

    new_param_bounds[1:] = param_bounds

    return new_param_bounds





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

@TimeMeasure
def run_mcmc_code(
    file: Path,
    target_search_params: list,
    target_stellar_params,
    iteration_num: int = 50_000,
    run_number: int = 3,
    analytic_sim: bool = True,
    batman_bool: bool = False,
    real_data_bool: bool = True,
    do_nested_sampling: bool = False,
):
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
    if real_data_bool:
        times, flux = download_data_api(*target_search_params)
    else:
        times = np.load("TestTimes.npy")
        flux = np.load("TestFluxesNoise.npy")
        # flux = add_gaussian_error(flux, 0, 5e-4)
        plt.plot(times, flux)
        plt.show()


    #Save the params for easier testing
    # np.save("Test-Params/times", times)
    # np.save("Test-Params/flux", flux)

    # times = np.load("Test-Params/times.npy")
    # flux = np.load("Test-Params/flux.npy")



    # stellar_params = get_stellar_params(file, target_name) # Todo -> Currently just give the regular stellar params
    stellar_params = target_stellar_params  # [radius, mas, limb_darkening_model, limb_darkening_coefficients]
    # estimated_params = estimate_parameters(
    #             times,
    #             flux,
    #             stellar_params,
    #             signal_detection_efficiency=10,
    #             period_min=3,
    #             period_max=9,
    #         )
    estimated_params = np.array([[0.11946044, 0.07106616, 8.34280155, 0.        , 1.57079633,
        0.        , 0.        , 2.73798487, 0.        ],
       [0.04668363, 0.04405248, 4.07167491, 0.        , 1.57079633,
        0.        , 0.        , 0.97389479, 0.        ]])
    initial_params = np.atleast_2d(
        np.vstack([estimated_params,
                   # np.array([0, 0, 0, 0, 0, 0, 0, np.pi / 4, 0.392])
                   ])
    )
    # np.save("Test-Params/initial_params", initial_params)



    # initial_params = np.load("Test-Params/initial_params.npy")



    #     initial_params = np.atleast_2d([ 0.095751,  0.07806046,  3.5224991,  0.          ,np.radians(84),  0.,
    #    0.,         -3.6653389, 0])



    true_vals = np.atleast_2d(
        np.array(
            [
                [
                    0.171,  #  +- 0.005 eta
                    0.0731,    # a
                    8.3501898, # P
                    0.0398, # e
                    np.radians(87.61), # inc
                    np.radians(182.5), # omega
                    0, # big_ohm
                    2.73763007, # phase_lag
                    0, # mass
                ],
                [
                    0.0480,
                    0.0453,
                    4.074554,
                    0.052162,
                    np.radians(87.49),
                    np.radians(141.11),
                    0,
                    0.97367272,
                    0,
                ],
            ]
        )
    )
    print(f"{initial_params=}, {initial_params.shape=}")
    proposal_std = np.atleast_2d(estimate_proposal(times, flux))  # Todo
    param_bounds = np.atleast_2d(estimate_bounds(times, flux))  # Todo
    noise = estimate_noise(times, flux)  # Todo
    param_names = np.atleast_2d(generate_param_names(initial_params))

    param_names, true_vals, initial_params, proposal_std, param_bounds, priors, prior_transform_funcs = (
        prepare_arrays_for_mcmc(
            param_names,
            true_vals,
            initial_params,
            proposal_std,
            param_bounds,
            analytic_sim,
        )
    )

    print(f"After {initial_params.shape=}")
    print(f"After {proposal_std.shape=}")

    input_params = extend_params_for_stellar(initial_params, stellar_params)
    print(f"After {input_params.shape=}")
    param_names = extend_names_for_stellar(param_names)
    proposal_std = extend_proposal_for_stellar(proposal_std)
    param_bounds = extend_param_bounds_for_stellar(param_bounds)
    true_vals = extend_params_for_stellar(true_vals, stellar_params)

    priors = np.full(proposal_std.shape, None)
    # priors[1,0] = {"distribution": "gaussian", "lower_bound": 0.01, "upper_bound":0.2, "mean": input_params[1,0], "std":5*1e-2}
    priors, prior_transform_funcs = prior_transform_calcs(priors, param_bounds, proposal_std, input_params)

    print(f"After {proposal_std.shape=}")
    

    def likelihood_fn(params):
        return gaussian_error_ln_likelihood(
            flux,
            None,
            lambda input_params: flux_data_from_params(
                input_params, times, analytical_bool=analytic_sim, batman_bool=batman_bool #############
            ),
            params,
            noise,
        )

    plt.figure()
    plt.title(f"Estimated Parameters Initial Fit\n{file}")
    plt.plot(times, flux, label="Data")
    plt.plot(
        times,
        flux_data_from_params(
            input_params, times, analytical_bool=True, batman_bool=batman_bool
        ),
        label="Estimated",
        ls=":",
        alpha=0.5
    )
    plt.plot(
        times,
        flux_data_from_params(true_vals, times, analytical_bool=True, batman_bool=batman_bool),
        label="Literature-reported Value",
        ls="--",
        alpha=0.5
    )
    plt.legend()
    plt.savefig("inferred_flux_plot_before.pdf", dpi=500)
    plt.show()

    # print(f"{input_params.shape=}")
    # print(f"{input_params[0]=}")
    # print(f"{input_params[0,0]=}")

    r_star = input_params[0,0]  

    # print(f"{r_star=}")

    for i in range(run_number):
        # print(f"{input_params.shape=}")
        mcmc = MCMC(
            flux,
            initial_param_fuzzer(input_params, proposal_std, param_bounds),
            param_bounds,
            proposal_std,
            true_vals = true_vals,
            param_names=param_names,
            likelihood_func=likelihood_fn,
            inclination_rejection_func=lambda input_params: inclination_checker(
                proposals = input_params, r_star = r_star
            ),
            priors=priors,
            prior_transforms=prior_transform_funcs,
            specified_folder_name=Path(file) / f"run_{i}",
            max_cpu_nodes=4,
        )

        if do_nested_sampling:
            mcmc.nested_sampling()
            do_nested_sampling = False
            return
        mcmc.rj_mh(iteration_num)
        marginalised = mcmc.marginalize_by_model()

        mcmc.chain_to_plot_and_estimate(true_vals)
        mcmc.corner_plot()

        for model in marginalised:
            print(f"{model=:=^30}")
            mcmc.chain_to_plot_and_estimate(true_vals=true_vals,
                                            chain=marginalised[model]["parameters"],
                                            likelihood_chain=marginalised[model]["likelihoods"],
                                            planet_number=model)
            try:
                mcmc.corner_plot(true_vals=true_vals,
                                 chain=marginalised[model]["parameters"],
                                 likelihood_chain=marginalised[model]["likelihoods"],
                                 planet_number=model)
            except AssertionError:
                pass
        # mcmc.metropolis_hastings(iteration_num)
        # mcmc.gaussian_hmc(iteration_num)

        plt.figure()
        plt.title(f"Inferred Parameters vs Literature-reported Fit\n{file}")
        plt.plot(times, flux, label="Data")
        plt.plot(
            times,
            flux_data_from_params(true_vals, times, analytical_bool=True, batman_bool=batman_bool),
            label="Literature-reported Value",
            ls="--",
            alpha=0.5
        )
        plt.plot(
            times,
            flux_data_from_params(mcmc.chain[np.argmax(mcmc.likelihood_chain)],
                                  times, analytical_bool=True, batman_bool=batman_bool),
            label="Max-Likelihood",
            ls=":",
            alpha=0.5
        )
        plt.legend()
        plt.savefig(Path(file) / f"run_{i}" / "inferred_flux_plot_after.pdf", dpi=500)
        plt.show()

        our_values = mcmc.chain[np.argmax(mcmc.likelihood_chain)]

        np.save(Path(file) / f"run_{i}" /"Input_values.npy", input_params)
        np.save(Path(file) / f"run_{i}" /"Literature_values.npy", true_vals)
        np.save(Path(file) / f"run_{i}" /"our_values.npy", our_values)
        np.save(Path(file) / f"run_{i}" /"times.npy", times)
        np.save(Path(file) / f"run_{i}" /"flux.npy", flux)


def main():
    data = download_data_api()  # Todo
    files = process_data(data)  # Todo

    [run_mcmc_code(file) for file in files]  # Todo


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # main()
    taget_name = "Toi-1130"
    exptime = None
    mission = "TESS"
    sector = None
    author = "SPOC"
    cadence = None
    indicies_requested = (0, 2)
    max_number_downloads = 31
    use_regression_model = False
    use_lightcurve_direct = True
    target_search_params = [
        taget_name,
        exptime,
        mission,
        sector,
        author,
        cadence,
        indicies_requested,
        max_number_downloads,
        use_regression_model,
        use_lightcurve_direct
    ]

    # times, flux = download_data_api(*target_search_params)

    # # plt.plot(times, flux)/
    # print(f"Shapes of times and flux: {times.shape}, {flux.shape}")
    # plt.plot(times, flux)
    # plt.show()

    radius_toi_1811 =   0.687 * 696.34e6 / 1.496e11
    mass_toi_1811 = 	0.684 * 2e30 / 6e24
    limb_darkening_model = 2
    limb_darkening_coefficients = [0.50, 0.27]

    stellar_params = [
        radius_toi_1811,
        mass_toi_1811,
        limb_darkening_model,
        limb_darkening_coefficients[0],
        limb_darkening_coefficients[1],
    ]  # Based on WASP 148

    run_mcmc_code(
        file="TOI-1130_test_3",
        target_search_params=target_search_params,
        target_stellar_params=stellar_params,
        iteration_num=4_000_000,
        run_number=1,
        analytic_sim=True,
        batman_bool=True,
        real_data_bool=True,
        do_nested_sampling=False,
    )
