import datetime
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
import numba
import dynesty
from dynesty import plotting as dyplot
from corner import corner
from numpy import ndarray
from numpy.random import normal
from scipy.stats import multivariate_normal
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


def find_first_greater(arr, x):
    idx = np.argmax(arr > x)
    if arr[idx] > x:
        return idx
    else:
        return -1


@numba.jit
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)


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


def build_folder_name(specified_folder_name: Optional[str | Path] = None) -> Path:
    if specified_folder_name:
        pth = Path(specified_folder_name)
        if not pth.is_dir():
            pth.mkdir(parents=True, exist_ok=True)
        return pth

    date_str = datetime.date.today().isoformat()

    base_dir = Path(".")
    existing_runs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(date_str)
    ]
    run_number = len(existing_runs) + 1
    folder_name = f"{date_str}_run{run_number}"
    data_folder = base_dir / folder_name

    # Create the folder
    data_folder.mkdir(parents=True, exist_ok=False)
    return data_folder


def safe_exp(x: np.array) -> np.array:
    threshold = 708
    # This value is chosen since `np.exp(709)` is near the limit for most systems
    result = np.zeros_like(x, dtype=float)

    # Only calc values that are safe in the threshold
    mask = x <= threshold
    result[mask] = np.exp(x[mask])
    result[~mask] = 1.0
    return result


def pad_array(array, max_pad):
    padded_array = []
    for subarray in array:
        pad_width = max_pad - subarray.shape[-1]
        if pad_width > 0:
            try:
                subarray_padded = np.pad(subarray, pad_width=((0, 0) , (0, pad_width)), constant_values=np.nan)
            except ValueError:
                subarray_padded = np.pad(subarray, pad_width=((0, pad_width)), constant_values=np.nan)
        else:
            subarray_padded = subarray
        padded_array.append(subarray_padded)
    return padded_array


class MCMC:

    def __init__(
        self,
        raw_data: ndarray,
        initial_parameters: ndarray,
        param_bounds: ndarray[List[Tuple[float, float]]],
        proposal_std: ndarray,
        likelihood_func: Callable[[ndarray], float],
        param_names=np.ndarray,
        specified_folder_name: Optional[str | Path] = None,
        inclination_rejection_func: Optional[Callable[[ndarray], bool]] = None,
        priors: Optional[ndarray[Callable]] = None,
        prior_transforms: Optional[ndarray[Callable]] = None,
        max_cpu_nodes: int = 16,
        **kwargs,
    ):

        # Logistics
        self.max_cpu_nodes = max_cpu_nodes
        self.sim_number = 1
        self.data_folder = build_folder_name(specified_folder_name)
        self.plot_save_folder = self.data_folder / "plots"
        self.plot_save_folder.mkdir(parents=True, exist_ok=True)
        self.param_names = param_names
        self.initial_parameters = initial_parameters

        # Diagnostics
        self.acceptance_num = 0
        self.rejection_num = 0
        self.iteration_num = 1  # Can't start on the zeroth iteration

        # Statistics
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.burn_in_index: Optional[int] = None
        self.remaining_chain_length = None
        self.fisher_information: Optional[np.ndarray] = None

        # MCMC-Inputs
        self.raw_data: ndarray = raw_data
        self.param_bounds = param_bounds
        self.lower_bounds = self.param_bounds[:, :, 0]
        self.upper_bounds = self.param_bounds[:, :, 1]
        self.proposal_std: ndarray = proposal_std
        self.varying_mask = np.where(self.proposal_std != 0, True, False).flatten()
        self.fixed_mask = np.where(self.proposal_std == 0, True, False).flatten()
        self.likelihood_func: Callable = likelihood_func
        self.inclination_rejection_func: Callable = inclination_rejection_func

        def flat_likelihood_func(current_varied_params: ndarray) -> float:
            params = self.initial_parameters.flatten().copy()
            params[self.varying_mask] = current_varied_params
            return self.likelihood_func(params)

        self.flat_likelihood_func = flat_likelihood_func

        # MCMC-Runtime
        empty_chain = np.empty_like(
            initial_parameters, shape=(1, *initial_parameters.shape)
        )
        empty_chain[0] = initial_parameters
        self.chain = empty_chain
        self.likelihood_chain = np.atleast_1d(self.likelihood_func(initial_parameters))
        self.momentum = None

        # Nested Sampling
        self.nested_results = None
        self.prior_transforms = prior_transforms

        if kwargs:
            # This is used for re-loading the object from a saved file.
            self.__dict__.update(kwargs)

    def save(self):
        """
        Saves the object state to a dynamically named folder.
        """
        # Use numpy saving for more efficient processing
        np.save(self.data_folder / "raw_data.npy", np.array(self.raw_data))
        np.save(self.data_folder / "chain.npy", self.chain)
        np.save(self.data_folder / "likelihoods.npy", self.likelihood_chain)

        with open(self.data_folder / "mcmc_attributes.pkl", "wb") as f:
            ignore_attrs = ["chain", "likelihood_chain", "raw_data"]
            attrs = {k: v for k, v in self.__dict__.items() if k not in ignore_attrs}
            dill.dump(attrs, f)

    @classmethod
    def load(cls, data_folder: Path):
        """
        Loads the object state from a specifically named folder.
        """
        data_folder = Path(data_folder)
        if data_folder.is_dir():
            try:
                # Load the object using dill
                with open(data_folder / "mcmc_attributes.pkl", "rb") as f:
                    mcmc_attributes = dill.load(f)
                    raw_data = np.load(data_folder / "raw_data.npy")
                    try:
                        obj = cls(
                            raw_data=raw_data,
                            specified_folder_name=data_folder,
                            **mcmc_attributes,
                        )
                    except TypeError as e:
                        print(
                            f"Error occurred due to missing attributes in"
                            f" {data_folder / 'mcmc_attributes.pkl'},"
                            f"Current attributes are {mcmc_attributes}."
                        )
                        raise e
                    try:
                        obj.chain = np.load(data_folder / "chain.npy")
                        obj.likelihood_chain = np.load(data_folder / "likelihoods.npy")
                    except FileNotFoundError as e:
                        print(
                            "Error occurred due to missing data in the MCMC save state"
                        )
                        raise e

                    try:
                        with open(data_folder / "nested_sampling_results.pkl", "rb") as f:
                            obj.nested_results = dill.load(f)
                    except FileNotFoundError as e:
                        print("Nested Sampling result not found. Skipping."
                        )
                return obj
            except Exception as e:
                raise TypeError(f"Failed to load object: {e}")
        else:
            raise FileNotFoundError(
                f"{data_folder} is not a valid saved state of {cls.__name__}"
            )

    def nested_sampling(self):
        li_fn = self.flat_likelihood_func
        flat_prior_trans = self.prior_transforms.flatten()[self.varying_mask]
        prior_transform = lambda u: [flat_prior_trans[i](u_i) for i, u_i in enumerate(u)]
        ndim = np.sum(self.varying_mask)
        sampler = dynesty.NestedSampler(loglikelihood=li_fn, prior_transform=prior_transform,
                                        ndim=ndim, nlive=20_000)
        sampler.run_nested(dlogz=0.0001)
        sresults = sampler.results
        self.nested_results = sresults
        with open(self.data_folder / "nested_sampling_results.pkl", "wb") as f:
            dill.dump(sresults, f)
        labels = self.param_names.flatten()[self.varying_mask]
        dyplot.runplot(sresults)
        plt.show()
        dyplot.cornerplot(sresults, labels=labels)
        plt.show()
        dyplot.traceplot(sresults,
                         truth_color='black', show_titles=True,
                         trace_cmap='viridis', connect=True,
                         connect_highlight=range(5), labels=labels)
        plt.show()


    def proposal_within_bounds(self, proposals):
        """
        Parameters:
        - proposals: np.ndarray of shape (sim_number, max_cpu_nodes, num_params)
          The array of proposed parameter values.
        Returns:
        - proposal_bools: np.ndarray of shape (sim_number, max_cpu_nodes)
          Boolean array indicating whether each proposal is within bounds.
        """
        is_within_bounds = (proposals >= self.lower_bounds) & (
            proposals <= self.upper_bounds
        )
        proposal_bools = is_within_bounds.all(axis=2)
        return proposal_bools

    def metropolis_hastings(
        self,
        num_of_new_iterations: int,
    ):
        """
        Performs Metropolis-Hastings MCMC sampling.
        """
        max_iteration_number = self.prepare_chains_for_new_iters(num_of_new_iterations)

        prev_iter = self.iteration_num - 1

        pbar = tqdm(initial=1, total=num_of_new_iterations, desc="MCMC Run ")

        remaining_iter = num_of_new_iterations
        while self.iteration_num < (max_iteration_number - 1):
            current_params = self.chain[prev_iter]
            current_likelihood = self.likelihood_chain[prev_iter]
            proposals = current_params + normal(
                0, self.proposal_std, size=(self.sim_number, *self.chain[0].shape)
            )

            proposal_within_bounds = self.proposal_within_bounds(proposals)

            if self.inclination_rejection_func and not self.inclination_rejection_func(
                proposals[:,1:,:]
            ):
                self.rejection_num += 1
                self.chain[prev_iter + 1] = current_params
                self.likelihood_chain[prev_iter + 1] = current_likelihood
                self.iteration_num += 1
                prev_iter += 1
                pbar.update(1)
                remaining_iter -= 1
                continue  # Skip to the next iteration

            # Keep clipping as easiest solution that works with multiprocessing and
            # negligible run cost
            for planet_index in range(self.param_bounds.shape[0]):  # Number of planets
                for param_index in range(
                    self.param_bounds.shape[1]
                ):  # Number of parameters per planet
                    lower, upper = self.param_bounds[planet_index, param_index]
                    proposals[:, planet_index, param_index] = np.clip(
                        proposals[:, planet_index, param_index], lower, upper
                    )

            if self.max_cpu_nodes == 1:
                proposal_likelihoods = np.atleast_1d(self.likelihood_func(proposals[0]))
            else:
                with Pool(nodes=self.max_cpu_nodes) as pool:
                    proposal_likelihoods = pool.map(self.likelihood_func, proposals)

            acceptance_probs = np.minimum(
                1,
                safe_exp(
                    np.array(proposal_likelihoods) - self.likelihood_chain[prev_iter]
                ),
            )

            for s in range(self.sim_number):
                self.iteration_num += 1
                prev_iter += 1
                pbar.update(1)
                remaining_iter -= 1
                if proposal_within_bounds[s].all() and (
                    np.random.rand() < acceptance_probs[s]
                ):
                    self.acceptance_num += 1
                    self.chain[prev_iter] = proposals[s]
                    self.likelihood_chain[prev_iter] = proposal_likelihoods[s]
                    break  # Exit after accepting a proposal
                else:
                    self.rejection_num += 1
                    self.chain[prev_iter] = current_params
                    self.likelihood_chain[prev_iter] = current_likelihood

            # Because iteration number is the number of the next iteration
            # due to 0 indexing
            acceptance_rate = self.acceptance_num / prev_iter
            self.sim_number = int(
                min(
                    (
                        np.ceil(1 / acceptance_rate)
                        if acceptance_rate > 0
                        else self.sim_number
                    ),
                    self.max_cpu_nodes,
                )
            )
            self.sim_number = min(self.sim_number, remaining_iter)
            # self.chain[self.iteration_num] = current_params
            # self.likelihood_chain[self.iteration_num] = current_likelihood

        # Ignore the last one in chain as this seems to go nan / inf
        self.chain = self.chain[:-1]
        self.likelihood_chain = self.likelihood_chain[:-1]
        print(f"{acceptance_rate=}")
        pbar.close()
        self.chain_wrap_up()

    def gaussian_hmc(self, num_of_new_iterations: int,
                     timestep: float=np.pi/2,
                     est_burn_in_end: int=5000,
                     estimate_interval: int=1):

        param_number_sq = self.proposal_std.flatten().shape[0] ** 3
        mh_iter = param_number_sq - self.iteration_num
        if self.iteration_num <= param_number_sq:
            self.metropolis_hastings(mh_iter)
            self.iteration_num -= 1 # Temporary Solution that will probably become permanent
            prev_iter = self.iteration_num - 1

            self.chain = self.chain.reshape(self.chain.shape[0], -1)

            self.estimated_covariance_matrix = self.update_covariance(estimate_interval, prev_iter)
            self.estimated_mean = self.update_mean(estimate_interval, prev_iter)

        self.prepare_chains_for_new_iters(max(10,num_of_new_iterations - mh_iter))

        pbar = tqdm(
            initial=1, total=num_of_new_iterations, desc="MCMC Run "
        )
        prev_iter = self.iteration_num - 1
        current_position = self.chain[prev_iter]
        current_ln_likelihood = self.likelihood_func(current_position)

        print(f"Pre GHMC {self.acceptance_num=}")
        for i in range(max(10,num_of_new_iterations - mh_iter)):
            accept, new_likelihood, new_pos = self.do_gaussian_hmc_step(current_ln_likelihood, current_position,
                                                                        self.estimated_covariance_matrix, timestep)
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
            if self.iteration_num >= est_burn_in_end and self.iteration_num % estimate_interval == 0:
                self.determine_burn_in_index()
                self.estimated_covariance_matrix = self.update_covariance(estimate_interval, prev_iter)
                self.estimated_mean = self.update_mean(estimate_interval, prev_iter)
            pbar.update(1)

        print("\n", self.estimated_covariance_matrix)
        print(f"Post GHMC {self.acceptance_num=}")
        acceptance_rate = self.acceptance_num / (self.rejection_num + self.acceptance_num)

        burn_in = self.determine_burn_in_index()
        autoc = autocorrelation(self.chain[burn_in:])
        print(f"{acceptance_rate=}")
        print(f"ESF={1/(1+2*np.sum(autoc))}")
        domain = np.arange(self.iteration_num)
        fig, axs = plt.subplots(
            nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
        )
        fig.suptitle("Chains")
        for i in range(self.estimated_covariance_matrix.shape[0]):
            axs[i].plot(domain, self.chain[:, i])
        plt.savefig(self.plot_save_folder / "chains.png")
        plt.close()
        fig, axs = plt.subplots(
            nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
        )
        fig.suptitle("Burn-in chains")
        for i in range(self.estimated_covariance_matrix.shape[0]):
            axs[i].plot(domain[:burn_in], self.chain[:burn_in, i])
        plt.savefig(self.plot_save_folder / "burn-in_chains.png")
        plt.close()

        fig, axs = plt.subplots(
            nrows=self.chain.shape[1], ncols=1, figsize=(10, 8)
        )
        fig.suptitle("Post-Burn-in chains")
        for i in range(self.estimated_covariance_matrix.shape[0]):
            axs[i].plot(domain[burn_in:], self.chain[burn_in:, i])
        plt.savefig(self.plot_save_folder / "post-burn-in_chains.png")
        plt.close()

        plt.title("Post-Burn-in Likelihoods")
        plt.plot(domain, self.likelihood_chain)
        plt.savefig(self.plot_save_folder / "post-burn-in_likelihood.png")
        plt.close()
        corner(self.chain[burn_in:, self.varying_mask])
        plt.savefig(self.plot_save_folder / "corner.png")
        plt.close()

        self.chain_wrap_up()

    def update_covariance(self, interval, max_iter):
        chain_segment = self.chain[self.burn_in_index:max_iter:interval, :]
        chain_varying = chain_segment[:, self.varying_mask]
        cov_varying = np.corrcoef(chain_varying, rowvar=False)
        return cov_varying

    def update_mean(self, interval, max_iter):
        chain_segment = self.chain[self.burn_in_index:max_iter:interval, :]
        mean_varying = np.mean(chain_segment[:, self.varying_mask], axis=0)
        return mean_varying

    def hamiltonian(self, cov, pos, mean, mom):
        delta = pos[self.varying_mask] - mean
        return -self.likelihood_func(pos) + 0.5 * mom.transpose() @ np.linalg.inv(cov) @ mom

    def do_gaussian_hmc_step(self, current_ln_likelihood, current_pos, covariance_mat, timestep):

        expected_mean = self.estimated_mean
        current_normal = multivariate_normal(np.zeros(len(current_pos[self.varying_mask])), covariance_mat)

        current_mom = current_normal.rvs() # Velocity sample
        a_i = np.linalg.inv(covariance_mat) @ current_mom
        b_i = current_pos[self.varying_mask] - expected_mean

        new_varying_pos = expected_mean + a_i * np.sin(timestep) + b_i * np.cos(timestep)
        new_mom = covariance_mat @ (a_i * np.cos(timestep) - b_i * np.sin(timestep))

        new_pos = current_pos.copy()
        new_pos[self.varying_mask] = new_varying_pos

        new_ln_likelihood = self.likelihood_func(new_pos)

        old_h = self.hamiltonian(covariance_mat, current_pos, expected_mean, current_mom)
        new_h = self.hamiltonian(covariance_mat, new_pos, expected_mean, new_mom)

        acceptance_prob = np.exp(-new_h + old_h)

        accept = random() < acceptance_prob
        return accept, new_ln_likelihood, new_pos

    def prepare_chains_for_new_iters(self, num_of_new_iterations):
        max_iteration_number = self.iteration_num + num_of_new_iterations
        empty_chain = np.empty_like(
            self.chain, shape=(max_iteration_number, *self.chain.shape[1:])
        )
        empty_likelihood = np.empty_like(
            self.likelihood_chain, shape=max_iteration_number
        )
        empty_chain[: len(self.chain)] = self.chain
        empty_likelihood[: len(self.chain)] = self.likelihood_chain
        self.chain = empty_chain
        self.likelihood_chain = empty_likelihood
        return max_iteration_number

    def chain_wrap_up(self):
        self.determine_burn_in_index()
        self.mean = np.mean(self.chain[self.burn_in_index :], axis=0)
        self.var = np.var(self.chain[self.burn_in_index :], axis=0)
        self.save()

    def chain_to_plot_and_estimate(
        self, true_vals: Optional[np.ndarray[float]] = None, manual_burn_in_idx: int = 0
    ):
        if not isinstance(manual_burn_in_idx, np.int64 | int):
            raise TypeError(f"{manual_burn_in_idx=} is not an integer")
        non_fixed_indexes = np.array(self.proposal_std, dtype=bool)
        max_pad = max(np.sum(non_fixed_indexes, axis=1))

        masked_chain = [
                self.chain[manual_burn_in_idx:, i, non_fixed_indexes[i]]
                for i in range(self.chain.shape[1])
            ]
        masked_names = [
                self.param_names[i, non_fixed_indexes[i]]
                for i in range(self.param_names.shape[0])
                ]
        padded_chain = pad_array(masked_chain, max_pad)
        padded_names = pad_array(masked_names, max_pad)

        chain = np.stack(padded_chain, axis=1)
        param_names = np.stack(padded_names, axis=0)
        likelihoods = self.likelihood_chain[manual_burn_in_idx:]

        # print(f"{chain.shape=}, {param_names.shape=}, {true_vals.shape=}")

        print("MCMC sampling completed.")

        plt.figure(figsize=(10, 8))
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].set_xlabel("Iteration #")
        x = np.arange(len(chain))

        axs[0].plot(x, likelihoods)
        axs[1].plot(x[self.burn_in_index:], likelihoods[self.burn_in_index:])

        if true_vals is not None:
            axs[1].hlines(
                self.likelihood_func(true_vals), xmin=0, xmax=len(chain),
                linestyles="--", color="red", label="True", alpha=0.5
            )
        axs[1].hlines(
            max(likelihoods), xmin=0, xmax=len(chain),
            linestyles=":", label="Max-Likelihood", alpha=0.5
        )
        plt.legend(loc="best")

        axs[0].set_ylabel(r"Log Likelihoods")
        plt.tight_layout()
        plt.savefig(self.data_folder / "Liklihood_plot.pdf", dpi=500)
        # plt.show()
        plt.close()

        fig, axs = plt.subplots(
            nrows=chain.shape[2], ncols=chain.shape[1], figsize=(10, 8)
        )
        # Ensure axs is always 2D
        if chain.shape[2] == 1 and chain.shape[1] == 1:
            axs = np.array([[axs]])  # Wrap single axes into a 2D array
        elif chain.shape[1] == 1:
            axs = np.expand_dims(axs, axis=1)  # Add column dimension
        elif chain.shape[2] == 1:
            axs = np.expand_dims(axs, axis=0)  # Add row dimension

        # axs = axs.reshape(chain[0].shape)
        fig.suptitle("Parameter Iterations")

        x = np.arange(len(chain))

        true_val_idx = 0
        remaining_true_vals = true_vals[non_fixed_indexes]
        for body in range(chain.shape[1]):
            for i, name in enumerate(param_names[body]):
                try:
                    if np.isnan(name):
                        continue
                except TypeError:
                    pass
                param_samples = chain[:, body, i]
                print(
                    f"Estimated {name}: {np.mean(param_samples):.3e}",
                    (
                        f", true {name}: {remaining_true_vals[true_val_idx]:.3e}"
                        if true_vals is not None
                        else None
                    ),
                )

                axs[i, body].plot(x, param_samples, label=name)
                if true_vals is not None:
                    axs[i, body].hlines(
                        remaining_true_vals[true_val_idx],
                        xmin=0,
                        xmax=len(chain),
                        linestyles="--",
                        color="red",
                    )
                min_val, max_val = minmax(param_samples)
                axs[i, body].vlines(
                    self.burn_in_index,
                    ymin=max_val,
                    ymax=min_val,
                    linestyles="dotted",
                    color="red",
                )
                axs[i, body].set_ylabel(f"{name}")
                true_val_idx += 1
        plt.xlabel("Iteration #")
        plt.tight_layout()
        plt.savefig(self.data_folder / "chain_plot_plot.pdf", dpi=500)
        plt.close()

        fig, axs = plt.subplots(
            nrows=chain.shape[2], ncols=chain.shape[1], figsize=(10, 8)
        )

        # Ensure axs is always 2D
        if chain.shape[2] == 1 and chain.shape[1] == 1:
            axs = np.array([[axs]])  # Wrap single axes into a 2D array
        elif chain.shape[1] == 1:
            axs = np.expand_dims(axs, axis=1)  # Add column dimension
        elif chain.shape[2] == 1:
            axs = np.expand_dims(axs, axis=0)  # Add row dimension

        fig.suptitle("Parameter Iterations After Burn In")
        plt.xlabel("Iteration #")
        chain = chain[self.burn_in_index :]
        x = np.arange(len(chain))

        true_val_idx = 0
        for body in range(chain.shape[1]):
            for i, name in enumerate(param_names[body]):
                try:
                    if np.isnan(name):
                        continue
                except TypeError:
                    pass
                param_samples = chain[:, body, i]
                print(
                    f"Estimated {name}: {np.mean(param_samples):.3e}",
                    (
                        f", true {name}: {remaining_true_vals[true_val_idx]:.3e}"
                        if true_vals is not None
                        else None
                    ),
                )
                axs[i, body].plot(x, param_samples, label=name)
                if true_vals is not None:
                    axs[i, body].hlines(
                        remaining_true_vals[true_val_idx],
                        xmin=0,
                        xmax=len(chain),
                        linestyles="--",
                        color="red",
                    )
                min_val, max_val = minmax(param_samples)
                axs[i, body].set_ylabel(f"{name}")
                true_val_idx += 1

        plt.tight_layout()
        plt.savefig(self.data_folder / "chain_post_burn_in_plot_plot.pdf", dpi=500)
        plt.close()

    def corner_plot(
        self, true_vals: Optional[np.ndarray] = None, burn_in_index: int = None
    ):
        non_fixed_indexes = np.array(self.proposal_std, dtype=bool)
        if burn_in_index is None:
            burn_in_index = self.burn_in_index

        # Flatten the chain to have shape (samples, parameters)
        flattened_chain = np.concatenate(
            [
                self.chain[burn_in_index:, i, non_fixed_indexes[i]]
                for i in range(self.chain.shape[1])
            ],
            axis=1,
        )

        # Flatten param_names and true_vals to match the flattened_chain dimensions
        flattened_param_names = np.concatenate(
            [
                self.param_names[i, non_fixed_indexes[i]]
                for i in range(self.param_names.shape[0])
            ]
        )

        if true_vals is not None:
            flattened_true_vals = np.concatenate(
                [true_vals[i, non_fixed_indexes[i]] for i in range(true_vals.shape[0])]
            )
        else:
            flattened_true_vals = None

        # print(f"{flattened_chain.shape=}, {flattened_param_names.shape=}, {flattened_true_vals.shape=}")

        # Pass the flattened arrays to the corner plot
        corner(
            flattened_chain,
            labels=flattened_param_names,
            truths=flattened_true_vals,
            show_titles=True,
            title_kwargs={"fontsize": 18},
            title_fmt=".2e",
        )
        plt.savefig(self.data_folder / "corner_plot.pdf", dpi=500)
        # plt.show()
        plt.close()

    def determine_burn_in_index(self) -> int:
        """
        Determines the burn-in cutoff index for an MCMC chain.
        Returns:
            int: The burn-in cutoff index.
        """
        max_idx = self.likelihood_chain[:self.iteration_num].argmax()
        max_likelihood = self.likelihood_chain[max_idx]
        two_perc_iter = self.iteration_num // 50
        upper_var_iter = min(self.iteration_num, max_idx + two_perc_iter)
        lower_var_iter = max(0, max_idx - two_perc_iter)
        var = np.std(self.likelihood_chain[lower_var_iter:upper_var_iter])
        lower_likelihood = max_likelihood - var
        burn_in_idx = find_first_greater(self.likelihood_chain, lower_likelihood)
        self.burn_in_index = int(burn_in_idx)
        return burn_in_idx


class Statistics:

    def __init__(self, folder_names: list[str | Path]):
        # MCMC handling
        self.folder_names = [Path(f_name) for f_name in folder_names]
        self.folder_indexing = {f_name: i for i, f_name in enumerate(folder_names)}
        self.chain_num: int = len(folder_names)
        self.loaded_mcmcs: List[MCMC] = []

        # Chain Details
        self._unique_stats = 2
        self.statistics_data = np.empty(shape=(self.chain_num, self._unique_stats))
        self._means_idx = 0
        self._var_idx = 1

        # Global Statistics
        self.mean_of_means = None
        self.variance_of_means = None
        self.gelman_rubin = None

        self.load_folders()

    def load_folders(self):
        mcmcs = []
        valid_paths = []
        for f_path in self.folder_names:
            if f_path.is_dir():
                mcmcs.append(MCMC.load(f_path))
                valid_paths.append(f_path)
            else:
                print(f"{f_path} is not a valid directory")

        self.chain_num = len(valid_paths)
        chain = mcmcs[0].chain
        self.statistics_data = np.empty(
            shape=(self.chain_num, self._unique_stats, *chain[0].shape)
        )

        if self.chain_num != len(self.folder_names):
            self.loaded_mcmcs = []

        for idx, (mcmc, f_path) in enumerate(zip(mcmcs, valid_paths)):
            self.folder_indexing[f_path.name] = idx
            self.loaded_mcmcs.append(mcmc)

    def calc_gelman_rubin(self) -> np.ndarray:
        """
        Calculates the Gelman_Rubin statistic of the loaded chains using the formula
        https://en.wikipedia.org/wiki/Gelman-Rubin_statistic
        Any parameters which do not vary are set to have a statistic
        of zero.
        """
        stats = self.statistics_data
        stats[:, self._means_idx] = [mcmc.mean for mcmc in self.loaded_mcmcs]
        stats[:, self._var_idx] = [mcmc.var for mcmc in self.loaded_mcmcs]
        mean_of_means = np.mean(stats[:, self._means_idx], axis=0)

        len_chain = len(self.loaded_mcmcs[0].chain)
        chain_num = self.chain_num
        var_of_means = (
            len_chain
            / (chain_num - 1)
            * np.sum(np.power(stats[:, self._means_idx] - mean_of_means, 2), axis=0)
        )
        mean_var = np.mean(stats[:, self._var_idx], axis=0)
        numerator = (len_chain - 1) / len_chain * mean_var + var_of_means / len_chain
        self.gelman_rubin = numerator / mean_var
        self.gelman_rubin *= np.where(
            self.loaded_mcmcs[0].proposal_std == 0, False, True
        )
        return self.gelman_rubin
