from typing import Tuple

import numpy as np
from numpy.random import normal


def generate_linear_data(
    x: np.ndarray,
    params: np.ndarray,
    noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for a linear model y = m x + c with Gaussian noise.

    Args:
        x (np.ndarray): Independent variable values.
        params (np.ndarray): m (true slope), c (true intercept)
        noise (np.ndarray): Gaussian noise for respective parameters.
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays.
    """
    m = params[0], noise[0]
    c = params[1], noise[1]
    m_sim = normal(m[0], m[1], len(x))
    c_sim = normal(c[0], c[1], len(x))
    y = m_sim * x + c_sim
    return x, y


def generate_sinusoidal_data(
    x: np.ndarray,
    params: np.ndarray,
    noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for a sinusoidal model y = A sin(b x + c) + d with Gaussian parameter uncertainties.

    Args:
        x (np.ndarray): Independent variable values.
        params (np.ndarray): [A_true, b_true, c_true, d_true]
        noise (np.ndarray): Gaussian noise for respective parameters [A_noise, b_noise, c_noise, d_noise].

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays.
    """
    A_true, b_true, c_true, d_true = params
    A_noise, b_noise, c_noise, d_noise = noise

    # Simulate parameters with uncertainties
    A_sim = normal(A_true, A_noise, len(x))
    b_sim = normal(b_true, b_noise, len(x))
    c_sim = normal(c_true, c_noise, len(x))
    d_sim = normal(d_true, d_noise, len(x))

    # Generate synthetic data
    y = A_sim * np.sin(b_sim * x + c_sim) + d_sim
    return x, y
