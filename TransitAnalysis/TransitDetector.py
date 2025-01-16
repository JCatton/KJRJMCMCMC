from transitleastsquares import transitleastsquares, transit_mask, cleaned_array
import numpy as np
import matplotlib.pyplot as plt

def run_tls(data: np.ndarray, times_input: np.ndarray, limb_darkening_model: str, limb_darkening_coefficients: list, plot_bool = False, save_loc = None, index = None):
    """
    """
    from transitleastsquares import transitleastsquares, transit_mask, cleaned_array
    model = transitleastsquares(times_input, data)
    results = model.power(limb_dark = limb_darkening_model, u = limb_darkening_coefficients)

    period = results.period
    transit_times = results.transit_times
    transit_depth = results.depth
    duration = results.duration
    SDE = results.SDE
    t_0 = results.T0

    output_dict = {"Period": period, "Transit_times": transit_times, "Transit_depth": transit_depth, "Duration": duration, "SDE": SDE, "t_0": t_0}

    intransit = transit_mask(times_input, results.period, 2*results.duration, results.T0)
    y_second_run = data[~intransit]
    t_second_run = times_input[~intransit]
    t_second_run, y_second_run = cleaned_array(t_second_run, y_second_run)

    if plot_bool:
        plot_tls_stuff(results, times_input, data, save_loc, index)
                
    return y_second_run, t_second_run, output_dict

def plot_tls_stuff(results, times_input, data, save_loc = None, index = None):
    """
    Plot useful information from the TLS results

    Parameters:
    - results: TLS results object
    - times_input: Array of time values
    - data: Array of flux values

    Returns:
    - Displays a plot of the data with the TLS model overlayed
    """
    from transitleastsquares import transit_mask
    in_transit = transit_mask(
        times_input,
        results.period,
        results.duration,
        results.T0)
    plt.scatter(
        times_input[in_transit],
        data[in_transit],
        color='red',
        s=2,
        zorder=0)
    plt.scatter(
        times_input[~in_transit],
        data[~in_transit],
        color='blue',
        alpha=0.5,
        s=2,
        zorder=0)
    plt.plot(
        results.model_lightcurve_time,
        results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
    plt.xlim(min(times_input), max(times_input))
    plt.ylim(min(data*0.98), max(data*1.02))
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux')
    if save_loc is not None:
        plt.savefig(f"{save_loc}/TLS_Model_overlay_{index}.pdf")
    else:
        plt.show()

    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.title(f"Power spectrum of the data with period {results.period}")
    plt.xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    if save_loc is not None:
        plt.savefig(f"{save_loc}/TLS_Power_spectrum_{index}.pdf")
    else:
        plt.show()


def search_for_transits(data: np.ndarray, times_input: np.ndarray, limb_darkening_model: str, limb_darkening_coefficients: list, signal_detection_efficiency: float = 10.0, plot_bool = False, save_loc = None):
    """
    """
    # list of dictionaries: [dict([Period, transit_times, transit_depth, duration, SDE])]
    results_list = [] 
    # Run TLS on the data
    current_signal_detection_efficiency = 1000
    while current_signal_detection_efficiency > signal_detection_efficiency:
        data, times_input, dictionary_entry = run_tls(data, times_input, limb_darkening_model, limb_darkening_coefficients, plot_bool, save_loc, len(results_list))
        current_signal_detection_efficiency = dictionary_entry["SDE"]

        if current_signal_detection_efficiency < signal_detection_efficiency:
            print(f"{len(results_list)} transits have been found with a signal detection efficiency of {signal_detection_efficiency}")
            break
        print(f"Found a planet with SDE of {current_signal_detection_efficiency} and period of {dictionary_entry["Period"]}")
        results_list.append(dictionary_entry)
    
    background_mean = np.mean(data)
    background_std = np.std(data)
    
    return results_list, background_mean, background_std

if __name__ == "__main__":
    from sim.SimulateAndFlux import flux_data_from_params
    from MCMC.main import add_gaussian_error

    # Stellar parameters: [radius, mass]
    radius_wasp148a = 0.912 * 696.34e6 / 1.496e11
    mass_wasp148a = 0.9540 * 2e30 / 6e24

    eta1 = 0.3
    eta2 = 0.4
    # planet_params =[ [ eta,   a,     P,   e,               inc, omega, OHM, phase_lag ] ]
    planet_params = np.array(
        [
            [eta1, 0.08215, 8.803809, 0.208, np.radians(90), 0, 0, 0, 0.287],
            [eta2, 0.2044, 34.525, 0.1809, np.radians(90), 0, 0, np.pi / 4, 0.392]
        ]
    )
    # True inclinations are 89.3 and 104.9 +- some

    times_input = np.linspace(0, 4 * 34, 60000)  # Three orbital periods for planet 1

    planet_params_analytical = planet_params[:, :-1]
    output_analytical = flux_data_from_params(
        stellar_params=[radius_wasp148a, mass_wasp148a], planet_params=planet_params_analytical, times=times_input, analytical_bool=True
    )

    sigma_n = 1e-3
    fluxes = add_gaussian_error(output_analytical, 0, sigma_n)

    results = search_for_transits(fluxes, times_input, signal_detection_efficiency=10.0, plot_bool=True)

    print(f"{results=}")
    print(f"shape of results: {np.array(results).shape}")


