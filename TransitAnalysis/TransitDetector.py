from transitleastsquares import transitleastsquares, transit_mask, cleaned_array
import numpy as np
import matplotlib.pyplot as plt
def search_for_transits(data: np.ndarray, times_input: np.ndarray, signal_detection_efficiency: float = 10.0, plot_bool = False):
    """
    """
    # [[Period, transit_times, transit_depth, duration, SDE]]
    results_list = [] 
    # Run TLS on the data
    current_signal_detection_efficiency = 1000
    while current_signal_detection_efficiency > signal_detection_efficiency:

        data, times_input, iteration_list = run_tls(data, times_input, plot_bool)

        current_signal_detection_efficiency = iteration_list[-1]

        if current_signal_detection_efficiency < signal_detection_efficiency:
            print(f"{len(results_list)} transits have been found with a signal detection efficiency of {signal_detection_efficiency}")
            break
        print(f"Found a planet with SDE of {current_signal_detection_efficiency} and period of {iteration_list[0]}")
        results_list.append(iteration_list)
    
    return results_list
