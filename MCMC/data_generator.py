import numpy as np
from numpy.random import normal
from typing import Tuple

def generate_linear_data(
    x: np.ndarray,
    m: Tuple[float, float],
    c: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for a linear model y = m x + c with Gaussian noise.

    Args:
        x (np.ndarray): Independent variable values.
        m (tuple): True slope. Standard deviation of Gaussian noise.
        c (tuple): True intercept. Standard deviation of Gaussian noise.
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays.
    """
    m_sim = normal(m[0], m[1], len(x))
    c_sim = normal(c[0], c[1], len(x))
    y = m_sim * x + c_sim
    return x, y

def generate_sinusoidal_data(
    x: np.ndarray,
    a_true: Tuple[float, float],
    b_true: Tuple[float, float],
    c_true: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for a sinusoidal model y = A sin(b x + c) with Gaussian noise.

    Args:
        x (np.ndarray): Independent variable values.
        a_true (tuple): True amplitude. Standard deviation of Gaussian noise.
        b_true (tuple): True frequency. Standard deviation of Gaussian noise.
        c_true (tuple): True phase shift. Standard deviation of Gaussian noise.
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays.
    """
    a_sim = normal(a_true[0], a_true[1], len(x))
    b_sim = normal(b_true[0], b_true[1], len(x))
    c_sim = normal(c_true[0], c_true[1], len(x))
    y = a_sim * np.sin(b_sim * x + c_sim)
    return x, y

