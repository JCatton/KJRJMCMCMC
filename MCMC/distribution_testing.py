from mcmc import MCMC, Statistics
from main import gaussian_error_ln_likelihood
from scipy.stats import multivariate_normal
from random import random#
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from corner import corner

def gaussian_log_likelihood(pos: np.ndarray, covariance_mat: np.mat, mean: np.ndarray):
    delta = pos - mean
    return -0.5 * np.log(np.linalg.det(covariance_mat) + delta.transpose() @ np.linalg.inv(covariance_mat) @ delta)

def gaussian_hmc(num_of_new_iterations: int, timestep: float, hessian: np.matrix):

    acceptance_number = 0
    rejection_number = 0
    iteration_number = 1
    chain = np.empty(shape=(num_of_new_iterations, hessian.shape[0]),  dtype=np.float64)
    likelihoods = np.empty(num_of_new_iterations, dtype=np.float64)

    pbar = tqdm(
        initial=1, total=num_of_new_iterations, desc="MCMC Run "
    )
    prev_iter = iteration_number - 1
    chain[prev_iter] = np.array([100,20,-40,560,10])
    current_position = chain[prev_iter]
    current_likelihood = likelihood_func(current_position)

    gradient = update_gradient(current_position, hessian)

    for i in range(num_of_new_iterations - 1):
        iteration_number += 1
        prev_iter += 1
        pbar.update(1)
        # new_pos, accept, new_likelihood = self.do_gaussian_hmc_step(current_position, timestep, hessian, gradient)

        accept, new_likelihood, new_pos = do_gaussian_hmc_step(current_likelihood, current_position, gradient,
                                                               hessian, timestep)
        if accept:
            current_position = new_pos
            current_likelihood = new_likelihood
            acceptance_number += 1
        else:
            rejection_number += 1

        chain[prev_iter] = current_position
        likelihoods[prev_iter] = current_likelihood
        hessian = update_hessian(hessian)
        gradient = update_gradient(current_position, hessian)

    acceptance_rate = acceptance_number / num_of_new_iterations
    autoc = autocorrelation(chain[1000:])
    print(f"{acceptance_rate=}")
    print(f"ESF={np.sum(autoc)}")
    domain = np.arange(iteration_number)
    fig, axs = plt.subplots(
            nrows=chain.shape[1], ncols=1, figsize=(10, 8)
        )
    for i in range(hessian.shape[0]):
        axs[i].plot(domain, chain[:, i])
    plt.show()
    fig, axs = plt.subplots(
        nrows=chain.shape[1], ncols=1, figsize=(10, 8)
    )
    for i in range(hessian.shape[0]):
        axs[i].plot(domain[:1000], chain[:1000, i])
    plt.show()
    plt.plot(domain, likelihoods)
    plt.show()
    corner(chain[1000:])
    plt.show()

def update_gradient(current_position, hessian):
    """
    Currently define gradient for multivariate Gaussian. WIP
    """
    gradient = np.linalg.inv(hessian) @ (current_position)
    return gradient

def update_hessian(hessian):
    return hessian

def do_gaussian_hmc_step(current_likelihood, current_position, gradient, hessian, timestep):
    covariance_mat = - np.linalg.inv(hessian)
    expected_mean = current_position + covariance_mat @ gradient
    current_normal = multivariate_normal(np.zeros(len(current_position)), covariance_mat)
    a_i =  current_normal.rvs() # Velocity sample
    b_i = current_position - expected_mean
    new_pos = expected_mean + a_i * np.sin(timestep) + b_i * np.cos(timestep)
    new_likelihood = likelihood_func(new_pos)
    acceptance_prob = np.exp(new_likelihood - current_likelihood)
    accept = random() < acceptance_prob
    return accept, new_likelihood, new_pos

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
    global likelihood_func
    mean = np.zeros(5, dtype=np.float64)
    covariance = np.diag(np.ones(5, dtype=np.float64))
    covariance[0,1] = 0.3
    covariance[1,0] = 0.3
    covariance[2,0] = -0.25
    covariance[0,2] = -0.25
    def likelihood_func(x):
        return gaussian_log_likelihood(x, covariance, mean)
    gaussian_hmc(50000, np.pi / 2, -np.linalg.inv(covariance))#



if __name__ == '__main__':
    main()