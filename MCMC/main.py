import numpy as np
from data_generator import generate_linear_data
from metropolis_hastings import metropolis_hastings

def linear_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    sigma: float = 1.0
) -> float:
    """
    Computes the log-likelihood for the linear model y = m x + c.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        params (np.ndarray): Model parameters [m, c].
        sigma (float): Standard deviation of the noise.

    Returns:
        float: Log-likelihood value.
    """
    m, c = params
    model = m * x + c
    residuals = y - model
    log_likelihood = -0.5 * np.sum((residuals / sigma) ** 2)
    return log_likelihood

def main():
    # True parameters
    m_true = 2.5
    c_true = 1.0
    noise_std = 1.0

    # Generate synthetic data
    x_data = np.linspace(0, 10, 50)
    x_data, y_data = generate_linear_data(m_true, c_true, x_data, noise_std)

    # Initial parameter guess
    initial_params = np.array([0.0, 0.0])

    # Parameter bounds [(lower, upper), ...]
    param_bounds = [(-10, 10), (-10, 10)]

    # Proposal distribution standard deviations
    proposal_std = np.array([0.5, 0.5])

    # Number of MCMC iterations
    num_iterations = 10000

    # Run MCMC
    chain = metropolis_hastings(
        x=x_data,
        y=y_data,
        initial_params=initial_params,
        param_bounds=param_bounds,
        likelihood_fn=linear_likelihood,
        proposal_std=proposal_std,
        num_iterations=num_iterations
    )

    # Save chain
    np.save('mcmc_chain.npy', chain)

    # Print summary
    print("MCMC sampling completed.")
    print(f"Estimated m: {np.mean(chain[:, 0])}, true m: {m_true}")
    print(f"Estimated c: {np.mean(chain[:, 1])}, true c: {c_true}")

if __name__ == "__main__":
    main()
