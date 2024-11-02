from typing import Callable, List, Tuple, Any

from numpy import ndarray, dtype, floating
from tqdm import trange
import numpy as np
from numpy.random import normal


def metropolis_hastings(
    x: np.ndarray,
    y: np.ndarray,
    initial_params: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    proposal_std: np.ndarray,
    num_iterations: int,
) -> tuple[ndarray[Any, dtype[np.float64]], ndarray[Any, dtype[np.float64]]]:
    """
    Performs Metropolis-Hastings MCMC sampling.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        initial_params (np.ndarray): Initial guess for parameters.
        param_bounds (List[Tuple[float, float]]): Bounds for parameters.
        likelihood_fn (Callable): Function to compute the likelihood.
        proposal_std (np.ndarray): Standard deviation for proposal distributions.
        num_iterations (int): Number of iterations.

    Returns:
        np.ndarray: Chain of sampled parameters.
    """
    chain = np.zeros((num_iterations, *initial_params.shape))
    acceptance = 0

    current_params = initial_params.copy()
    current_likelihood = likelihood_fn(x, y, current_params)
    likelihoods = np.full(num_iterations, current_likelihood)

    for i in trange(num_iterations):
        proposal = current_params + normal(0, proposal_std, size=initial_params.shape)

        for j, (lower, upper) in enumerate(param_bounds):
            proposal[:, j] = np.clip(proposal[:, j], lower, upper)

        # Compute likelihood of proposed parameters
        proposal_likelihood = likelihood_fn(x, y, proposal)

        # Acceptance probability
        acceptance_prob = min(1, np.exp(proposal_likelihood - current_likelihood))

        if np.random.rand() < acceptance_prob:
            acceptance += 1
            current_params = proposal
            current_likelihood = proposal_likelihood
        acceptance_rate = acceptance / i

        chain[i] = current_params
        likelihoods[i] = current_likelihood

    return chain, likelihoods


def determine_burn_in_index(chain: np.ndarray) -> int:
    """
    Determines the burn-in cutoff index for an MCMC chain.

    Args:
        chain (np.ndarray): The MCMC chain with shape (num_samples, num_params).

    Returns:
        int: The burn-in cutoff index.
    """
    num_params = chain.shape[1]
    indices = []

    for i in range(num_params):
        samples = chain[:, i]
        mean = np.mean(samples)
        std = np.std(samples)
        indices_param = np.where(np.abs(samples - mean) < std)[0]
        if len(indices_param) > 0:
            index_param = indices_param[0]
        else:
            index_param = len(samples)
        indices.append(index_param)

    burn_in_index = max(indices)
    return burn_in_index
