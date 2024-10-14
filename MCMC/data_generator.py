import numpy as np
from numpy.random import normal
from typing import Tuple

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
    Generate synthetic data for a sinusoidal model y = A sin(b x + c) with Gaussian noise.

    Args:
        x (np.ndarray): Independent variable values.
        params (np.ndarray): amplitude, angular frequency, phase shift, vertical_shift
        noise (np.ndarray): Gaussian noise for respective parameters.
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays.
    """
    a = params[0], noise[0]
    b = params[1], noise[1]
    c = params[2], noise[2]
    d = params[3], noise[3]
    a_sim = normal(a[0], a[1], len(x))
    b_sim = normal(b[0], b[1], len(x))
    c_sim = normal(c[0], c[1], len(x))
    d_sim = normal(d[0], d[1], len(x))
    y = a_sim * np.sin(b_sim * x + c_sim) + d_sim
    return x, y

