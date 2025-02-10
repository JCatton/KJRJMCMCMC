import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from distribution_testing import gaussian_log_likelihood, logistic_log_likelihood, Gaussian_HMC
import imageio.v2 as imageio
import os

images = []
base_test_folder= './testing_results/'


def ensure_folder(folder_name):
    fp = os.path.join(base_test_folder, folder_name)
    if not os.path.exists(fp):
        os.mkdir(fp)


ensure_folder("")


def covariance_heatmap_gif(covariance, covs, num_iterations, func_name: str = None):

    iter_num = 0
    func_folder = os.path.join(base_test_folder, func_name)
    filename = f"heatmap_iter_{iter_num}.png"
    plt.figure()
    plt.title(f"Covariance Difference Iteration Number {iter_num}")
    sns.heatmap(covariance - np.identity(covariance.shape[1]), vmin=0, vmax=1, square=True, cmap="crest")
    plt.savefig(os.path.join(func_folder, filename))
    images.append(imageio.imread(os.path.join(func_folder, filename)))
    os.remove(os.path.join(func_folder, filename))
    plt.close()

    for i, new_iter in enumerate(num_iterations):
        iter_num = (i + 1) * new_iter
        filename = f"heatmap_iter_{iter_num}.png"
        plt.figure()
        plt.title(f"Covariance Difference Iteration Number {iter_num}")
        sns.heatmap(covariance - covs[i], vmin=0, vmax=1, square=True, cmap="crest")
        plt.savefig(os.path.join(func_folder, filename))
        plt.close()
        images.append(imageio.imread(os.path.join(func_folder, filename)))
        os.remove(os.path.join(func_folder, filename))
    imageio.mimsave(os.path.join(func_folder, 'scatterplot.gif'), images, loop=0, fps=4)

class GaussianHMCTests(unittest.TestCase):
    def setUp(self):
        self.covariance_ident = np.diag(np.ones(5, dtype=np.float64))
        self.covariance_correlation = np.diag(np.ones(5, dtype=np.float64))
        self.covariance_correlation[0, 1] = 0.3
        self.covariance_correlation[1, 0] = 0.3
        self.covariance_correlation[2, 1] = -0.25
        self.covariance_correlation[1, 2] = -0.25

        self.covariance_size_diff = np.diag(np.array([1, 30, 2, 0.15, 16],dtype=np.float64))
        self.covariance_size_diff_corr = np.diag(np.array([1, 30, 2, 0.15, 16], dtype=np.float64))
        self.covariance_size_diff_corr[0, 1] = 3
        self.covariance_size_diff_corr[1, 0] = 3
        self.covariance_size_diff_corr[2, 1] = -0.5
        self.covariance_size_diff_corr[1, 2] = -0.5

        self.num_iterations = np.full(60, 2500, dtype=np.int32)

    def test_gaussian_5d_identity(self):
        """Test sampling a 15D Gaussian using the analytic HMC (gaussian_hmc).
        For an underlying distribution using the identity matrix as its covariance"""
        mean = np.array([0.0, 50, 180, -50, -8], dtype=np.float64)
        covariance = self.covariance_ident

        def ln_like(x):
            return (gaussian_log_likelihood(x, covariance, mean))

        inital_params = np.array([-1000.0, 0, 180, 50, 10], dtype=np.float64)
        num_iterations = self.num_iterations
        means = np.empty(shape=(len(num_iterations), len(inital_params)))
        covs = np.empty(shape=(len(num_iterations), *covariance.shape))
        hmc = Gaussian_HMC(ln_like, initial_parameters=inital_params, diagnostic_mean=mean)
        for i, new_iter in enumerate(num_iterations):
            timestep = np.pi / 2
            hmc.gaussian_hmc(new_iter, timestep, cov_mat_est_interval=10)
            covs[i] = hmc.estimated_covariance_matrix
            means[i] = np.mean(hmc.chain, axis=0)

        self.plot_results(covariance, covs, mean, means, num_iterations, sys._getframe().f_code.co_name)

    def test_gaussian_5d_correlation(self):
        """Test sampling a 5D Gaussian using the analytic HMC (gaussian_hmc).
        For an underlying distribution using the correlation and anti-correlation in its covariance"""
        mean = np.array([0.0, 50, 180, -50, -8], dtype=np.float64)
        covariance = self.covariance_correlation

        def ln_like(x):
            return (gaussian_log_likelihood(x, covariance, mean))

        inital_params = np.array([-1000.0, 0, 180, 50, 10], dtype=np.float64)
        num_iterations = self.num_iterations
        means = np.empty(shape=(len(num_iterations), len(inital_params)))
        covs = np.empty(shape=(len(num_iterations), *covariance.shape))
        hmc = Gaussian_HMC(ln_like, initial_parameters=inital_params, diagnostic_mean=mean)
        for i, new_iter in enumerate(num_iterations):
            timestep = np.pi / 2
            hmc.gaussian_hmc(new_iter, timestep, cov_mat_est_interval=10)
            covs[i] = hmc.estimated_covariance_matrix
            means[i] = np.mean(hmc.chain, axis=0)

        self.plot_results(covariance, covs, mean, means, num_iterations, sys._getframe().f_code.co_name)

    def test_gaussian_5d_size_diff(self):
        """Test sampling a 5D Gaussian using the analytic HMC (gaussian_hmc).
        For an underlying distribution using the various scales in its diagonal covariance"""
        mean = np.array([0.0, 50, 180, -50, -8], dtype=np.float64)
        covariance = self.covariance_size_diff

        def ln_like(x):
            return gaussian_log_likelihood(x, covariance, mean)

        inital_params = np.array([-1000.0, 0, 180, 50, 10], dtype=np.float64)
        num_iterations = self.num_iterations
        means = np.empty(shape=(len(num_iterations), len(inital_params)))
        covs = np.empty(shape=(len(num_iterations), *covariance.shape))
        hmc = Gaussian_HMC(ln_like, initial_parameters=inital_params, diagnostic_mean=mean)
        for i, new_iter in enumerate(num_iterations):
            timestep = np.pi / 2
            hmc.gaussian_hmc(new_iter, timestep, cov_mat_est_interval=10)
            covs[i] = hmc.estimated_covariance_matrix
            means[i] = np.mean(hmc.chain, axis=0)

        self.plot_results(covariance, covs, mean, means, num_iterations, sys._getframe().f_code.co_name)

    def test_gaussian_5d_size_diff_correlation(self):
        """Test sampling a 5D Gaussian using the analytic HMC (gaussian_hmc).
        For an underlying distribution using the various scales in its covariance with correlation"""
        mean = np.array([0.0, 50, 180, -50, -8], dtype=np.float64)
        covariance = self.covariance_size_diff_corr

        def ln_like(x):
            return gaussian_log_likelihood(x, covariance, mean)

        inital_params = np.array([-1000.0, 0, 180, 50, 10], dtype=np.float64)
        num_iterations = self.num_iterations
        means = np.empty(shape=(len(num_iterations), len(inital_params)))
        covs = np.empty(shape=(len(num_iterations), *covariance.shape))
        hmc = Gaussian_HMC(ln_like, initial_parameters=inital_params, diagnostic_mean=mean)
        for i, new_iter in enumerate(num_iterations):
            timestep = np.pi / 2
            hmc.gaussian_hmc(new_iter, timestep, cov_mat_est_interval=10)
            covs[i] = hmc.estimated_covariance_matrix
            means[i] = np.mean(hmc.chain, axis=0)

        self.plot_results(covariance, covs, mean, means, num_iterations, sys._getframe().f_code.co_name)

    def test_gaussian_4d_logistic_1d(self):
        """Test sampling a 1D Gaussian using the analytic HMC (gaussian_hmc)."""
        mean = np.array([0.0, 50, 180, -50, -8], dtype=np.float64)
        covariance = np.diag(np.ones(5, dtype=np.float64))
        covariance[0, 1] = 0.3
        covariance[1, 0] = 0.3
        covariance[2, 1] = -0.25
        covariance[1, 2] = -0.25

        def ln_like(x):
            return (gaussian_log_likelihood(x[:4], covariance[:4, :4], mean[:4]) +
                    logistic_log_likelihood(x[4], 1, mean[4]))

        inital_params = np.array([-1000.0, 0, 180, 50, 10], dtype=np.float64)
        num_iterations = self.num_iterations
        means = np.empty(shape=(len(num_iterations), len(inital_params)))
        covs = np.empty(shape=(len(num_iterations), *covariance.shape))
        hmc = Gaussian_HMC(ln_like, initial_parameters=inital_params, diagnostic_mean=mean)
        for i, new_iter in enumerate(num_iterations):
            timestep = np.pi / 2
            hmc.gaussian_hmc(new_iter, timestep, cov_mat_est_interval=10)
            covs[i] = hmc.estimated_covariance_matrix
            means[i] = np.mean(hmc.chain, axis=0)

        self.plot_results(covariance, covs, mean, means, num_iterations, sys._getframe().f_code.co_name)

    def plot_results(self, covariance, covs, mean, means, num_iterations, func_name: str = None):
        ensure_folder(func_name)
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.suptitle('Gaussian HMC test Across Iteration Number')
        axs[0].set_title("Norm of Estimated Mean from True Values")
        axs[0].plot([(i + 1) * num for i, num in enumerate(num_iterations)], np.linalg.norm(means - mean, axis=1))
        axs[0].set_ylabel("Norm")
        axs[1].set_title("Norm of Estimated Covariance from True Values")
        axs[1].plot([(i + 1) * num for i, num in enumerate(num_iterations)],
                    [np.linalg.norm(covariance - cov) for cov in covs])
        axs[1].set_ylabel("Norm")
        axs[1].set_xlabel("Iteration #")
        plt.savefig(base_test_folder + f"{func_name}/norm_fig")
        plt.close()
        covariance_heatmap_gif(covariance, covs, num_iterations, func_name)


if __name__ == '__main__':
    unittest.main()
