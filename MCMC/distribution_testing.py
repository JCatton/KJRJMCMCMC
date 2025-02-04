from scipy.stats import multivariate_normal
from random import random#
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from corner import corner

def gaussian_log_likelihood(pos: np.ndarray, covariance_mat: np.mat, mean: np.ndarray):
    delta = pos - mean
    n = len(pos)
    # return -0.5 * n * np.log(np.linalg.det(covariance_mat)) - 0.5 * delta.transpose() @ np.linalg.inv(covariance_mat) @ delta
    return -0.5 * n * np.log(np.linalg.det(covariance_mat)) - 0.5 * delta.transpose() @ covariance_mat @ delta

def potential_energy(position_ln_likelihood: float) -> float:
    """
    Betancourt Introduction to HMC
    """
    return -position_ln_likelihood

def kinetic_energy(mom_generating_func: multivariate_normal, momentum: np.ndarray) -> float:
    """
    Betancourt Introduction to HMC
    """
    return - np.log(mom_generating_func.pdf(momentum))

def gaussian_hmc(num_of_new_iterations: int, timestep: float, cov_mat_est: np.matrix, cov_mat_est_interval: int=1):

    acceptance_number = 0
    rejection_number = 0
    iter_num = 1
    chain = np.empty(shape=(num_of_new_iterations, cov_mat_est.shape[0]),  dtype=np.float64)
    likelihoods = np.empty(num_of_new_iterations, dtype=np.float64)

    pbar = tqdm(
        initial=1, total=num_of_new_iterations, desc="MCMC Run "
    )
    prev_iter = iter_num - 1
    chain[prev_iter] = np.array([100,20,-40,560,10])
    # chain[prev_iter] = np.array([1, 1, 1, 1, 1])
    current_position = chain[prev_iter]
    current_ln_likelihood = ln_likelihood_func(current_position)

    for i in range(num_of_new_iterations - 1):
        accept, new_likelihood, new_pos = do_gaussian_hmc_step(current_ln_likelihood, current_position,
                                                               cov_mat_est, timestep)
        if accept:
            current_position = new_pos
            current_ln_likelihood = new_likelihood
            acceptance_number += 1
        else:
            rejection_number += 1

        chain[iter_num] = current_position
        likelihoods[iter_num] = current_ln_likelihood
        iter_num += 1
        prev_iter += 1
        if iter_num % cov_mat_est_interval == 0:
            cov_mat_est = update_covariance(chain, prev_iter, cov_mat_est_interval)
        pbar.update(1)
    print(cov_mat_est)


    acceptance_rate = acceptance_number / (rejection_number + acceptance_number)
    autoc = autocorrelation(chain[1000:])
    print(f"{acceptance_rate=}")
    print(f"ESF={1/(1+2*np.sum(autoc))}")
    domain = np.arange(iter_num)
    fig, axs = plt.subplots(
            nrows=chain.shape[1], ncols=1, figsize=(10, 8)
        )
    fig.suptitle("Chains")
    for i in range(cov_mat_est.shape[0]):
        axs[i].plot(domain, chain[:, i])
    plt.show()
    fig, axs = plt.subplots(
        nrows=chain.shape[1], ncols=1, figsize=(10, 8)
    )
    fig.suptitle("Burn-in chains")
    for i in range(cov_mat_est.shape[0]):
        axs[i].plot(domain[:50], chain[:50, i])
    plt.show()

    fig, axs = plt.subplots(
        nrows=chain.shape[1], ncols=1, figsize=(10, 8)
    )
    fig.suptitle("Burn-in chains")
    for i in range(cov_mat_est.shape[0]):
        axs[i].plot(domain[50:], chain[50:, i])
    plt.show()

    plt.title("Likelihoods")
    plt.plot(domain, likelihoods)
    plt.show()
    corner(chain[1000:])
    plt.show()

# def update_gradient(current_position, hessian):
#     """
#     Currently define gradient for multivariate Gaussian. WIP
#     """
#     gradient = np.linalg.inv(hessian) @ (current_position)
#     return gradient

def update_covariance(chain: np.ndarray, max_iter, interval: int):
    return np.corrcoef(chain[:max_iter, :], rowvar=0)

def hamiltonian(cov, pos, mean, mom):
    delta = pos - mean
    # return 0.5 * delta.transpose() @ cov @ delta - mean.transpose() @ pos + 0.5 * mom.transpose() @ np.linalg.inv(cov) @ mom
    # return 0.5 * delta.transpose() @ cov @ delta + 0.5 * mom.transpose() @ np.linalg.inv(cov) @ mom
    return -ln_likelihood_func(pos) + 0.5 * mom.transpose() @ np.linalg.inv(cov) @ mom
    # return 0.5 * pos.transpose() @ pos + 0.5 * mom.transpose() @ mom

def do_gaussian_hmc_step(current_ln_likelihood, current_pos, covariance_mat, timestep):

    expected_mean = mean
    current_normal = multivariate_normal(np.zeros(len(current_pos)), covariance_mat)

    current_mom = current_normal.rvs() # Velocity sample
    a_i = np.linalg.inv(covariance_mat) @ current_mom
    b_i = current_pos - expected_mean

    new_pos = expected_mean + a_i * np.sin(timestep) + b_i * np.cos(timestep)
    new_mom = covariance_mat @ (a_i * np.cos(timestep) - b_i * np.sin(timestep))

    # current_potential = potential_energy(current_ln_likelihood)
    # current_energy = current_potential + kinetic_energy(current_normal, current_mom)

    new_ln_likelihood = ln_likelihood_func(new_pos)
    # new_potential = potential_energy(new_ln_likelihood)
    # new_energy = new_potential + kinetic_energy(current_normal, new_mom)

    old_h = hamiltonian(covariance_mat, current_pos, expected_mean, current_mom)
    new_h = hamiltonian(covariance_mat, new_pos, expected_mean, new_mom)

    acceptance_prob = np.exp(-new_h + old_h)

    accept = random() < acceptance_prob
    return accept, new_ln_likelihood, new_pos

def autocorrelation (x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size//2]/np.sum(xp**2)

def main():
    global ln_likelihood_func, mean
    mean = np.zeros(5, dtype=np.float64)
    mean[0] = 1
    mean[2] = 10
    mean[3] = 60
    covariance = np.diag(np.ones(5, dtype=np.float64))
    covariance[0,1] = 0.3
    covariance[1,0] = 0.3
    covariance[2,0] = -0.25
    covariance[0,2] = -0.25
    def ln_likelihood_func(x):
        return gaussian_log_likelihood(x, covariance, mean)
    gaussian_hmc(50_000, np.pi / 2, covariance, cov_mat_est_interval=1000)



if __name__ == '__main__':
    main()