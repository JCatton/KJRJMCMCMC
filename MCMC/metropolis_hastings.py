import numpy as np
from typing import Callable, Tuple, List

def metropolis_hastings(
    x: np.ndarray,
    y: np.ndarray,
    initial_params: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    proposal_std: np.ndarray,
    num_iterations: int
) -> np.ndarray:
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
    num_params = len(initial_params)
    chain = np.zeros((num_iterations, num_params))
    current_params = initial_params.copy()
    current_likelihood = likelihood_fn(x, y, current_params)

    for i in range(num_iterations):
        # Propose new parameters
        proposal = current_params + np.random.normal(0, proposal_std, size=num_params)

        # Enforce parameter bounds
        for j, (lower, upper) in enumerate(param_bounds):
            proposal[j] = np.clip(proposal[j], lower, upper)

        # Compute likelihood of proposed parameters
        proposal_likelihood = likelihood_fn(x, y, proposal)

        # Acceptance probability
        acceptance_prob = min(1, np.exp(proposal_likelihood - current_likelihood))

        # Accept or reject
        if np.random.rand() < acceptance_prob:
            current_params = proposal
            current_likelihood = proposal_likelihood

        chain[i] = current_params

    return chain
