import numpy as np
import sim.FileCheck as fc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numba import jit



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
    eta_sq = radius_planet / radius_star * radius_planet / radius_star
    # Initialize delta flux array with all values set to 1 (no blocking)
    delta_flux = np.ones_like(d)
    behind_star = np.nonzero(y >= 0)
    near_star = np.nonzero((d <= radius_star - radius_planet))
    mid_and_near_star = np.nonzero((d <= radius_star + radius_planet))
    mid_star = np.setdiff1d(mid_and_near_star, near_star)
    in_front_near = np.setdiff1d(near_star, behind_star)
    infront_mid = np.setdiff1d(mid_star, behind_star)
    delta_flux[in_front_near] = 1 - eta_sq
    delta_flux[infront_mid] = overlap_calc(d, radius_planet, radius_star, infront_mid)
    return delta_flux

@jit
def overlap_calc(r, eta, radius_star, slice_indices):
    phi = np.arccos((r[slice_indices] ** 2 + 1 - eta ** 2) / (2 * r[slice_indices]))
    psi = np.arccos((r[slice_indices] ** 2 + eta ** 2 - 1) / (2 * r[slice_indices] * eta))
    
    radius_planet = eta * radius_star
    area1 = radius_planet ** 2 * psi
    area2 = radius_star ** 2 * phi
    area3 = r[slice_indices] * radius_star ** 2 * np.sin(phi)
    overlap_area = area1 + area2 - area3
    star_area = np.pi * radius_star ** 2
    
    blocked_flux = overlap_area / star_area
    return blocked_flux


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
        radius_planet = planet_params[i][0] * radius_star
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
            times, combined_flux
        )
        if saveloc:
            plt.savefig(os.path.join(full_saveloc, "combined_flux_plot.pdf"))
        plt.close()
    return combined_flux



def delta_flux_from_Mandel_and_Agol(a, i, P, phase, radius_star, eta, times):

    omega = 2*np.pi/P

    r = (a/radius_star) * np.sqrt(1-(np.sin(i)**2)*np.cos(omega * times + phase)**2)

    phi = np.arccos((r**2 + 1 - eta**2)/(2*r))

    psi = np.arccos((r**2 + eta**2 - 1)/(2*r*eta))

    # r(t) >= 1 + eta
    delta_flux = np.zeros_like(r)
    mask_case1 = (r >= 1 + eta)
    delta_flux[mask_case1] = 1

    # |1 - eta| < r(t) <= 1 + eta and CurrPhase is between -pi/2 and pi/2
    mask_case2 = ((np.abs(1 - eta) < r) & (r <= 1 + eta) &
                  ((omega * times + phase) >= -np.pi / 2) & ((omega * times + phase) <= np.pi / 2))
    delta_flux[mask_case2] = (1 - 1 / np.pi * (phi[mask_case2] + eta**2 * psi[mask_case2] -
                                               0.5 * np.sqrt(4 * r[mask_case2]**2 - (1 + r[mask_case2]**2 - eta**2)**2)))

    # r(t) <= 1 - eta and CurrPhase is between -pi/2 and pi/2
    mask_case3 = ((r <= 1 - eta) &
                  ((omega * times + phase) >= -np.pi / 2) & ((omega * times + phase) <= np.pi / 2))
    delta_flux[mask_case3] = 1 - eta**2

    return delta_flux


