from typing import Optional, Callable

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

class Gaussian_HMC:
    def __init__(self, likelihood_func, initial_parameters):

        # Logistics
        self.initial_parameters = np.array(initial_parameters, dtype=np.float64)

        # Diagnostics
        self.acceptance_num = 0
        self.rejection_num = 0
        self.iteration_num = 1  # Can't start on the zeroth iteration

        # Statistics
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.burn_in_index: Optional[int] = None

        # MCMC-Inputs
        self.likelihood_func: Callable = likelihood_func

        # MCMC-Runtime
        empty_chain = np.empty_like(
            initial_parameters, shape=(1, *self.initial_parameters.shape)
        )
        empty_chain[0] = self.initial_parameters
        self.chain = empty_chain
        self.estimated_covariance_matrix = np.identity(self.initial_parameters.shape[0])
        self.likelihood_chain = np.atleast_1d(self.likelihood_func(self.initial_parameters))


    def gaussian_hmc(self, num_of_new_iterations: int, timestep: float, cov_mat_est: np.matrix, cov_mat_est_interval: int=1):

        self.prepare_chains_for_new_iters(num_of_new_iterations)

        pbar = tqdm(
            initial=1, total=num_of_new_iterations, desc="MCMC Run "
        )
        prev_iter = self.iteration_num - 1
        current_position = self.chain[prev_iter]
        current_ln_likelihood = ln_likelihood_func(current_position)

        for i in range(num_of_new_iterations - 1):
            accept, new_likelihood, new_pos = self.do_gaussian_hmc_step(current_ln_likelihood, current_position,
                                                                   cov_mat_est, timestep)
            if accept:
                current_position = new_pos
                current_ln_likelihood = new_likelihood
                self.acceptance_num += 1
            else:
                self.rejection_num += 1

            self.chain[self.iteration_num] = current_position
            self.likelihood_chain[self.iteration_num] = current_ln_likelihood
            self.iteration_num += 1
            prev_iter += 1
            if self.iteration_num % cov_mat_est_interval == 0:
                cov_mat_est = self.update_covariance(prev_iter)
            pbar.update(1)

        print("\n", cov_mat_est)
        acceptance_rate = self.acceptance_num / (self.rejection_num + self.acceptance_num)
        autoc = autocorrelation(self.chain[1000:])
        print(f"{acceptance_rate=}")
        print(f"ESF={1/(1+2*np.sum(autoc))}")
        domain = np.arange(self.iteration_num)
        fig, axs = plt.subplots(
                nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
            )
        fig.suptitle("Chains")
        for i in range(cov_mat_est.shape[0]):
            axs[i].plot(domain, self.chain[:, i])
        plt.show()
        fig, axs = plt.subplots(
            nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
        )
        fig.suptitle("Burn-in chains")
        for i in range(cov_mat_est.shape[0]):
            axs[i].plot(domain[:50], self.chain[:50, i])
        plt.show()

        fig, axs = plt.subplots(
            nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
        )
        fig.suptitle("Burn-in chains")
        for i in range(cov_mat_est.shape[0]):
            axs[i].plot(domain[50:], self.chain[50:, i])
        plt.show()

        plt.title("Likelihoods")
        plt.plot(domain, self.likelihood_chain)
        plt.show()
        corner(self.chain[1000:])
        plt.show()

    def prepare_chains_for_new_iters(self, num_of_new_iterations):
        max_iteration_number = self.iteration_num + num_of_new_iterations
        empty_chain = np.empty_like(
            self.chain, shape=(max_iteration_number - 1, *self.chain.shape[1:])
        )
        empty_likelihood = np.empty_like(
            self.likelihood_chain, shape=max_iteration_number - 1
        )
        empty_chain[: len(self.chain)] = self.chain
        empty_likelihood[: len(self.chain)] = self.likelihood_chain
        self.chain = empty_chain
        self.likelihood_chain = empty_likelihood
        return max_iteration_number

    def update_covariance(self, max_iter):
        return np.corrcoef(self.chain[:max_iter, :], rowvar=0)

    def hamiltonian(self, cov, pos, mean, mom):
        delta = pos - mean
        return -ln_likelihood_func(pos) + 0.5 * mom.transpose() @ np.linalg.inv(cov) @ mom

    def do_gaussian_hmc_step(self, current_ln_likelihood, current_pos, covariance_mat, timestep):

        expected_mean = mean
        current_normal = multivariate_normal(np.zeros(len(current_pos)), covariance_mat)

        current_mom = current_normal.rvs() # Velocity sample
        a_i = np.linalg.inv(covariance_mat) @ current_mom
        b_i = current_pos - expected_mean

        new_pos = expected_mean + a_i * np.sin(timestep) + b_i * np.cos(timestep)
        new_mom = covariance_mat @ (a_i * np.cos(timestep) - b_i * np.sin(timestep))

        new_ln_likelihood = ln_likelihood_func(new_pos)

        old_h = self.hamiltonian(covariance_mat, current_pos, expected_mean, current_mom)
        new_h = self.hamiltonian(covariance_mat, new_pos, expected_mean, new_mom)

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
    Gaussian_HMC_obj = Gaussian_HMC(ln_likelihood_func, [100.,20.,-40.,560.,10.])
    Gaussian_HMC_obj.gaussian_hmc(30_000, np.pi / 2, covariance, cov_mat_est_interval=1000)



if __name__ == '__main__':
    main()