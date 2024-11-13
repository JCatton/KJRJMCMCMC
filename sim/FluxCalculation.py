import numpy as np
import sim.FileCheck as fc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numba import jit


@jit
def delta_flux_from_mandel_and_agol(
    x: float, 
    y: float, 
    z: float, 
    radius_star: float, 
    eta: float
) -> np.ndarray:
    """
    Calculate the delta flux for a planet using the Mandel and Agol model.

    Parameters:
    - x : x sky coordinate
    - y : y sky coordinate
    - z : towards the observer
    - eta : planet radius / star radius
    - radius_star : radius of the star

    Returns:
    - delta_flux : Array of delta flux values as fractions of the total stellar flux
    """
    d = np.sqrt(x**2 + y**2)
    r = d / radius_star
    delta_flux = np.ones_like(d)

    complete_overlap = np.nonzero((z > 0) & (r <= 1 - eta))
    partial_overlap = np.nonzero((z > 0) & (r > abs(1 - eta)) & (r < 1 + eta))

    delta_flux[complete_overlap] = 1 - eta * eta
    delta_flux[partial_overlap] -= overlap_calc(r, eta, radius_star, partial_overlap)
    return delta_flux


@jit
def overlap_calc(
    r: np.ndarray, 
    eta: float, 
    radius_star: float, 
    slice_indices: np.ndarray
) -> np.ndarray:
    """
    Calculate the overlap area between the star and planet.

    Parameters:
    - r : distance from the center of the star
    - eta : planet radius / star radius
    - radius_star : radius of the star
    - slice_indices : indices of the overlap area

    Returns:
    - blocked_flux : Array of blocked flux values as fractions of the total stellar flux
    """
    phi = np.arccos((r[slice_indices] ** 2 + 1 - eta**2) / (2 * r[slice_indices]))
    psi = np.arccos((r[slice_indices] ** 2 + eta**2 - 1) / (2 * r[slice_indices] * eta))

    radius_planet = eta * radius_star
    area1 = radius_planet**2 * psi
    area2 = radius_star**2 * phi
    area3 = r[slice_indices] * radius_star**2 * np.sin(phi)
    overlap_area = area1 + area2 - area3
    star_area = np.pi * radius_star**2

    blocked_flux = overlap_area / star_area
    return blocked_flux


@jit
def combined_delta_flux(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray,
    radius_star: float, 
    planet_params: np.ndarray, 
    times: np.ndarray
    ) -> np.ndarray:
    """
    Treating each transits individually, calculate the combined delta flux for all planets.

    Parameters:
    - x : Arrays of x positions of the star (If using Rebound) and planets over time
    - y : Arrays of y positions of the star (If using Rebound) and planets over time
    - z : Arrays of z positions of the star (If using Rebound) and planets over time
    - radius_star : Radius of the star
    - planet_params : List of list of the planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)] for each planet
    - times : Array of time values

    Returns:
    - combined_flux : Array of combined delta flux values as fractions of the total stellar flux
    """

    N = len(planet_params)


    # Initialize the combined delta flux as an array of ones
    combined_flux = np.ones(len(times))

    # Calculate the delta flux for each planet and subtract it from the combined flux
    for i in range(N):
        eta = planet_params[i][0]
        delta_flux = delta_flux_from_mandel_and_agol(
            x[i], y[i], z[i], radius_star, eta
        )
        combined_flux += delta_flux - 1
    return combined_flux


def plot_flux(
    times: np.ndarray, 
    flux_values: np.ndarray, 
    save_bool: bool =False
    ) -> None:
    """
    Plot the flux values against time.

    Parameters:
    - times: Array of time values
    - flux_values: Array of flux values
    - save: Boolean to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, flux_values, color="black", lw=1)
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Flux vs Time")
    if save_bool:
        plt.savefig("Flux_vs_Time.png")
    plt.show()
