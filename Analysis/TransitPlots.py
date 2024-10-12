import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm


# Define a function to calculate intensity using the quadratic limb darkening model with Kipping 2013 reparameterization
def calculate_intensity(r, q1, q2):
    # Calculate the limb darkening coefficients
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)

    # Calculate mu (cosine of the angle between line of sight and surface normal)
    mu = np.sqrt(1 - r**2)

    # Calculate the intensity based on the quadratic limb darkening law
    I = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

    return I


# Function to generate the 2D grid for the star and calculate the normalized intensity
def generate_heatmap(q1, q2, resolution=500):
    # Create a 2D grid for the star
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate the radial distance from the center of the star
    r = np.sqrt(X**2 + Y**2)

    # Mask points outside the star's disk (r > 1)
    mask = r <= 1

    # Calculate intensity only within the star's radius
    intensity = np.zeros_like(r)
    intensity[mask] = calculate_intensity(r[mask], q1, q2)

    # Normalize intensity so that the average intensity over the stellar disk is 1
    normalization_factor = np.mean(intensity[mask])
    normalized_intensity = intensity / normalization_factor

    return X, Y, normalized_intensity, resolution


# Funciton to Plot the Heatmap
def plot_heatmap(q1, q2):
    # Generate the intensity map
    X, Y, intensity_map = generate_heatmap(q1, q2)

    # Plotting the heatmap
    plt.figure(figsize=(8, 8))
    # Create an array for the full plot with white background
    full_map = (
        np.ones_like(intensity_map) * np.nan
    )  # Use NaN for areas outside the star
    full_map[intensity_map > 0] = intensity_map[
        intensity_map > 0
    ]  # Fill with intensity values

    plt.imshow(
        full_map, extent=(-1, 1, -1, 1), origin="lower", cmap="inferno", aspect="equal"
    )
    plt.colorbar(label="Normalized Intensity")
    plt.xlabel("x (Normalized Distance)")
    plt.ylabel("y (Normalized Distance)")
    plt.title(f"Stellar Intensity Heatmap (q1={q1}, q2={q2})")

    # Set the color of NaN values to white
    plt.set_cmap("inferno")  # Set the colormap to inferno
    plt.gca().set_facecolor("white")  # Set the face color to white
    plt.show()


# Function to obtain indicies of transit events
def GetTransitIndicies(a, thetas, R_star, buffer=2):
    # Find the indices of the points where the planet is in transit
    transit_indices = np.where(
        (thetas > np.pi)
        & (thetas < 2 * np.pi)
        & (np.abs(a * np.cos(thetas)) < buffer * R_star)
    )[0]
    count = 1
    print(transit_indices.shape)
    # Count the number of transits
    for i in range(0, len(transit_indices) - 1):
        # if i == (len(transit_indices)):
        #     break
        if (transit_indices[i] + 1) != transit_indices[i + 1]:
            count += 1

    # Calculate the x and y positions of the planet
    x = a * np.cos(thetas)
    y = a * np.sin(thetas)

    print(f" num transits = {count}")
    return transit_indices, x, y


# Function to generate the flux lightcurve
def FluxCalculator(Positions, R_planets, R_star, LimbDarkeningMap, buffer=2):
    LimbInt = LimbDarkeningMap[2]
    LimbRes = LimbDarkeningMap[3]
    total_flux = np.nansum(LimbInt)

    x_grid = np.linspace(-R_star, R_star, LimbRes)
    y_grid = np.linspace(-R_star, R_star, LimbRes)

    delta_flux = np.ones(len(Positions[0, 1]))
    aArray = Positions[:, 0]
    ThetaArray = Positions[:, 1]

    # Loop over the planets and calculate the flux at each time step
    for i in tqdm(range(0, len(R_planets))):
        a = aArray[i]
        thetas = ThetaArray[i]
        transit_indicies, All_planet_x, All_planet_y = GetTransitIndicies(
            a, thetas, R_star
        )
        planet_radius = R_planets[i]

        # Loop over the transit events and calculate the flux
        for j in tqdm(transit_indicies):
            # Calculate indices of the blocked area for the planet
            planet_x = All_planet_x[j]
            planet_y = All_planet_x[j]

            x_min = np.searchsorted(x_grid, planet_x - planet_radius)
            x_max = np.searchsorted(x_grid, planet_x + planet_radius)
            y_min = np.searchsorted(y_grid, planet_y - planet_radius)
            y_max = np.searchsorted(y_grid, planet_y + planet_radius)
            x_min = max(x_min, 0)
            x_max = min(x_max, LimbRes)
            y_min = max(y_min, 0)
            y_max = min(y_max, LimbRes)

            # Sum the intensity over the blocked area
            blocked_flux = np.nansum(LimbInt[y_min:y_max, x_min:x_max])

            # print("Before", delta_flux[j])
            delta_flux[j] = 1 - (blocked_flux / total_flux)
            # print("After", delta_flux[j])
    return delta_flux


# Function to plot the flux lightcurve
def PlotFlux(q1, q2, resolution, Data, planetRadius, stellarRadius, saveLocation):
    LimbDarkeningMap = generate_heatmap(q1, q2, resolution)
    Flux = FluxCalculator(Data, planetRadius, stellarRadius, LimbDarkeningMap, buffer=2)
    plt.plot(range(len(Flux)), Flux)
    plt.savefig(saveLocation)
