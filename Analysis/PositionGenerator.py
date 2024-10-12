import numpy as np
import matplotlib.pyplot as plt
import os


def PosMapGenerator(
    a,
    StartAngle,
    massStar,
    NumMaxOrbits,
    save_path="Analysis/Data/Positions.npy",
):
    """
    Generates the positions of planets over their orbits and saves the data to a specified file.

    Parameters:
    a : array-like
        Semi-major axes of the planets in meters.
    StartAngle : array-like
        Initial angular positions of the planets in radians.
    massStar : float
        Mass of the central star in kilograms.
    NumMaxOrbits : int
        The number of orbits to simulate.
    save_path : str, optional
        The file path where the output data will be saved (default is 'Working/TestPlanetData_EarthToNeptune.npy').

    Returns:
    finalOutput : ndarray
        A numpy array of shape (n, 3) where n is the number of planets.
        The array contains:
        - Semi-major axes in the first column
        - Angular positions over time in the second column
        - Placeholder for future use in the third column (currently not used)

    Example:
    --------
    >>> startTheta = np.pi * np.array([1, 0.3, 1.5, 1, 0.7, 1.4])
    >>> PlanetData = PosMapGenerator(
    ...     1.1e11 * np.array([1, 1.5, 5.2, 9.5, 19.2, 30]), startTheta, 2e30, 1
    ... )
    >>> # Data saved to 'Working/TestPlanetData_EarthToNeptune.npy'
    """
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Calculate orbital parameters and generate positions
    w = np.sqrt(6.67e-11 * massStar / (a**3))  # Angular velocity
    t = 2 * np.pi / w  # Orbital period
    maxTime = np.max(t)
    tSpace = np.linspace(0, maxTime * NumMaxOrbits, NumMaxOrbits * 20000000)

    finalOutput = np.empty((len(a), 3), dtype=object)
    finalOutput[:, 0] = a

    for i in range(len(a)):
        theta = (tSpace * 2 * np.pi / t[i] + StartAngle[i]) % (2 * np.pi)
        finalOutput[i, 1] = theta
        x = a[i] * np.cos(theta)
        y = a[i] * np.sin(theta)
        plt.plot(x, y, ".", ms=0.1)

    # Show the plot of the orbits
    plt.show()

    # Save the output to the specified file path
    np.save(save_path, finalOutput)
    print(f"Data saved to {save_path}")

    return 
