import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm

from N_body_sim import N_Body_sim
from FluxCalculation import combined_delta_flux
from Animation import animate_orbits


# Define the parameters for the star and planets
# make sure to define parameters in terms of AU, Earth masses, and days

# Stellar parameters: [radius, mass]
Stellar_params = [100 * 4.2635e-5, 333000 * 1.12]

# Planet parameters: [radius, mass, orbital radius, eccentricity, omega (phase)]
planet_params = [
    [1 * 4.2635e-5, 90.26, 0.045, 0.000, 0],
    # [0.5 * 4.2635e-5, 66, 0.078, 0.021, 90],
    # [2 * 4.2635e-5, 70, 0.1, 0.000, 45],
]

# Run N-body simulation
x_pos, y_pos, x_orbit, y_orbit, times = N_Body_sim(
    StellarMass=Stellar_params[1],
    planet_params=planet_params,
    SamplesPerOrbit=20000,
    numberMaxPeriod=4,
    saveloc="Example",
)

z_pos = np.zeros(
    (x_pos.shape)
)  # Currently does not output z positions so set to zero for now

# Calculate the combined delta flux for all planets
delta_flux_combined = combined_delta_flux(
    x=x_pos,
    y=y_pos,
    z=z_pos,
    radius_star=Stellar_params[0],
    planet_params=planet_params,
    times=times,
    saveloc="Example",
    plot=True,
)

# Animate the orbits and flux
ani = animate_orbits(
    x_pos=x_pos,
    y_pos=y_pos,
    x_orbit=x_orbit,
    y_orbit=y_orbit,
    times=times,
    planet_params=planet_params,
    flux=delta_flux_combined,
    saveloc="Example",
)

# animate_orbits currently outpts a gif file to the saveloc folder, use online gif to video converter to convert to mp4 e.g. https://ezgif.com/gif-to-mp4 then speed up the video
