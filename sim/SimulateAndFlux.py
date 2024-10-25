import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from N_body_sim import N_Body_sim
from FluxCalculation import combined_delta_flux


def simulate_and_interpolate_flux_vectorized(Stellar_params, planet_params, SamplesPerOrbit, numberMaxPeriod, times_input, PlotBool=False, saveloc=None):
    """
    Optimized wrapper to perform N-body simulation, calculate flux, and interpolate flux values for given time arrays.
    Vectorized interpolation for improved efficiency.
    
    Parameters:
    - Stellar_params: List of stellar parameters [radius, mass]
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - SamplesPerOrbit: Number of samples per orbit for the shortest period planet
    - numberMaxPeriod: Number of periods to simulate
    - times_input: Array of arrays of times for which flux is required
    - saveloc: Optional path to save results
    
    Returns:
    - interpolated_flux: List of arrays with interpolated flux values for each input time array
    """
    # Run the N-body simulation
    x_pos, y_pos, x_orbit, y_orbit, simulation_times = N_Body_sim(
        StellarMass=Stellar_params[1],
        planet_params=planet_params,
        SamplesPerOrbit=SamplesPerOrbit,
        numberMaxPeriod=numberMaxPeriod,
        saveloc=saveloc,
    )

    # Run Flux Calucaltion
    z_pos = np.zeros_like(x_pos)  # z = 0 for now
    flux_values = combined_delta_flux(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        radius_star=Stellar_params[0],
        planet_params=planet_params,
        times=simulation_times,
        saveloc=saveloc,
        plot=PlotBool
    )
    
    all_times = np.concatenate(times_input)  # Combine Times for interpolation
    flux_interpolator = interp1d(simulation_times, flux_values, kind='linear', fill_value="extrapolate") # Interpolate Flux
    all_interpolated_flux = flux_interpolator(all_times) 

    # Split interpolated flux back to the original structure of times_input -> from ChatGPT 4o
    split_indices = np.cumsum([len(times) for times in times_input[:-1]])
    interpolated_flux = np.split(all_interpolated_flux, split_indices)
    
    return interpolated_flux


# Example Usage 
if __name__ == "__main__":
        
    # Define Simulation Parameters
    # Stellar parameters: [radius, mass]
    Stellar_params = [100 * 4.2635e-5, 333000 * 1.12]

    # Planet parameters: [radius, mass, orbital radius, eccentricity, omega (phase)]
    planet_params = [
        [1 * 4.2635e-5, 90.26, 0.045, 0.000, 0],
        [0.5 * 4.2635e-5, 66, 0.078, 0.021, 90],
        [2 * 4.2635e-5, 70, 0.1, 0.000, 45],
    ]
    SamplesPerOrbit = 60000
    numberMaxPeriod = 4

    # Define some sample times for interpolation
    times_input = [
        np.linspace(0,12,60000),
        np.linspace(15,18,60000),
        np.linspace(20,30,60000),
    ]

    # Get Flux Values
    interpolated_flux_output = simulate_and_interpolate_flux_vectorized(
        Stellar_params=Stellar_params,
        planet_params=planet_params,
        SamplesPerOrbit=SamplesPerOrbit,
        numberMaxPeriod=numberMaxPeriod,
        times_input=times_input)


    print(interpolated_flux_output)
    for i in range(0, len(times_input)):
        plt.plot(times_input[i], interpolated_flux_output[i])
    plt.xlabel("Time (days)")
    plt.ylabel("Relative Brightness")
    plt.show()

