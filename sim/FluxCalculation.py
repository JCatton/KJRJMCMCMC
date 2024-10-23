import numpy as np
import FileCheck as fc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def delta_flux_from_cartesian(x, y, z, radius_star, radius_planet):
    """
    Calculate the delta flux for a single planet based on its position relative to the star.
    Parameters:
    - x: Array of horizontal positions of the planet relative to the star's center
    - y: Array of y-positions (direction to star from observer) of the planet relative to the star's center
    - z: Array of z-coordinates of the planet relative to the star's center (perpendicular to both x and y)
    - radius_star: Radius of the star
    - radius_planet: Radius of the planet

    Returns:
    - Array of delta flux values as fractions of the total stellar flux
    """
    # Center to center distance
    d = np.sqrt(x**2 + z**2)

    # Initialize delta flux array with all values set to 1 (no blocking)
    delta_flux = np.ones_like(d)

    # Iterate through each point and calculate the delta flux
    for i in tqdm(range(0, len(x)), total=len(x)):
        if d[i] >= (radius_star + radius_planet):  # No overlap
            continue
        elif y[i] >= 0:  # If the planet is behind the star, no overlap
            continue
        elif (
            d[i] <= radius_star - radius_planet
        ):  # If the planet is completely infront of the star, full overlap
            delta_flux[i] = (
                1 - radius_planet / radius_star * radius_planet / radius_star
            )
            continue
        elif (
            abs(radius_star - radius_planet) < d[i]
            and d[i] < radius_star + radius_planet
        ):  # Partial overlap

            # Derivation of the overlap area based on Mulcock (2024, Imperial College London, unpublished Literature Review)
            r1, r2 = radius_star, radius_planet
            phi = np.arccos((d[i] ** 2 + r1**2 - r2**2) / (2 * d[i] * r1))
            psi = np.arccos((d[i] ** 2 + r2**2 - r1**2) / (2 * d[i] * r2))
            area1 = r1**2 * phi
            area2 = r2**2 * psi
            area3 = d[i] * r1 * np.sin(phi)

            overlap_area = area1 + area2 - area3
            star_area = np.pi * r1**2
            delta_flux[i] = 1 - overlap_area / (star_area)

    return delta_flux


def combined_delta_flux(
    x, y, z, radius_star, planet_params, times, saveloc=None, plot=False
):
    """
    Treating each transits individually, calculate the combined delta flux for all planets.

    Parameters:
    - x, y, z: Arrays of x, y, and z positions of the star and planets over time
    - radius_star: Radius of the star
    - planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    - times: Array of time values
    - saveloc: Path to save the combined flux array
    - plot: Boolean to determine if a plot of the combined flux should be saved

    Returns:
    - Array of combined delta flux values as fractions of the total stellar flux
    - If saveloc is provided, the combined flux array is saved as a .npy file
    - If plot is True, a plot of the combined flux is saved as a .pdf file
    """

    N = len(planet_params)

    x_s, y_s, z_s = x[0], y[0], z[0]
    x_p_rel, y_p_rel, z_p_rel = x[1:] - x_s, y[1:] - y_s, z[1:] - z_s

    # Initialize the combined delta flux as an array of ones
    combined_flux = np.ones(len(times))

    # Calculate the delta flux for each planet and subtract it from the combined flux
    for i in range(N):
        radius_planet = planet_params[i][0]
        x, y, z = x_p_rel[i], y_p_rel[i], z_p_rel[i]
        delta_flux = delta_flux_from_cartesian(x, y, z, radius_star, radius_planet)
        combined_flux -= 1 - delta_flux

    if saveloc:
        full_saveloc = fc.check_and_create_folder(saveloc)
        timeseries_flux = np.array([times, combined_flux], dtype=np.float64)
        np.save(os.path.join(full_saveloc, "timeseries_flux.npy"), timeseries_flux)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.xlabel("Time (days)")
        plt.ylabel("Relative Brightness")
        plt.title("Light Curve")
        plt.plot(
            np.linspace(0, len(combined_flux) - 1, len(combined_flux)), combined_flux
        )
        if saveloc:
            plt.savefig(os.path.join(full_saveloc, "combined_flux_plot.pdf"))
        plt.close()
    return combined_flux
