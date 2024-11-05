import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import dill
import numpy as np
from numpy import ndarray
from numpy.random import normal
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from corner import corner

CPU_NODES = 16


def build_folder_name(specified_folder_name: Optional[str] = None):
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


class MCMC:

    def __init__(
        self,
        raw_data: ndarray,
        initial_parameters: ndarray,
        param_bounds: List[Tuple[float, float]],
        proposal_std: ndarray,
        likelihood_func: Callable[[ndarray], float],
        param_names = list[str],
        specified_folder_name: Optional[str] = None,
        max_cpu_nodes: int = 16,
        **kwargs,
    ):

        # Logistics
        self.max_cpu_nodes = max_cpu_nodes
        self.sim_number = 1
        self.data_folder = build_folder_name(specified_folder_name)
        self.param_names = param_names
        self.initial_parameters = initial_parameters

        # Diagnostics
        self.acceptance_num = 0
        self.rejection_num = 0
        self.iteration_num = 1  # Can't start on the zeroth iteration

        # MCMC-Inputs
        self.raw_data: ndarray = raw_data
        self.param_bounds: List[Tuple[float, float]] = param_bounds
        self.proposal_std: ndarray = proposal_std
        self.likelihood_func: Callable = likelihood_func

        # MCMC-Runtime
        empty_chain = np.empty_like(initial_parameters, shape=(1, *initial_parameters.shape))
        empty_chain[0] = initial_parameters
        self.chain = empty_chain
        self.likelihood_chain = np.array(self.likelihood_func(initial_parameters))

        if kwargs:
            # This is used for re-loading the object from a saved file.
            self.__dict__.update(kwargs)

    def save(self):
        """
        Saves the object state to a dynamically named folder.
        """
        # Use numpy saving for more efficient processing
        np.save(self.data_folder / "raw_data.npy", self.raw_data)
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
                        obj = cls(raw_data=raw_data, **mcmc_attributes)
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
                return obj
            except Exception as e:
                raise TypeError(f"Failed to load object: {e}")
        else:
            raise FileNotFoundError(
                f"{data_folder} is not a valid saved state of {cls.__name__}"
            )

    def metropolis_hastings(
        self,
        num_of_new_iterations: int,
    ):
        """
        Performs Metropolis-Hastings MCMC sampling.
        """
        max_iteration_number = self.iteration_num + num_of_new_iterations
        empty_chain = np.empty_like(
            self.chain, shape=(max_iteration_number, *self.chain.shape[1:])
        )
        empty_likelihood = np.empty_like(
            self.likelihood_chain, shape=max_iteration_number
        )

        empty_chain[: self.iteration_num + 1] = self.chain
        empty_likelihood[: self.iteration_num + 1] = self.likelihood_chain

        self.chain = empty_chain
        self.likelihood_chain = empty_likelihood

        prev_iter = self.iteration_num - 1

        pbar = tqdm(
            initial=self.iteration_num, total=max_iteration_number - 1, desc="MCMC Run "
        )

        remaining_iter = num_of_new_iterations - 1
        while self.iteration_num < (max_iteration_number - 1):
            current_params = self.chain[prev_iter]
            current_likelihood = self.likelihood_chain[prev_iter]

            proposals = current_params + normal(
                0, self.proposal_std, size=(self.sim_number, *self.chain[0].shape)
            )

            for j, (lower, upper) in enumerate(self.param_bounds):
                proposals[:, :, j] = np.clip(proposals[:, :, j], lower, upper)

            if self.max_cpu_nodes == 1:
                proposal_likelihoods = np.atleast_1d(
                    self.likelihood_func(proposals[0])
                )
            else:
                with Pool(nodes=self.max_cpu_nodes) as pool:
                    proposal_likelihoods = pool.map(self.likelihood_func, proposals)

            acceptance_probs = np.minimum(
                1,
                np.exp(
                    np.array(proposal_likelihoods) - self.likelihood_chain[prev_iter]
                ),
            )

            for s in range(self.sim_number):
                self.iteration_num += 1
                prev_iter += 1
                pbar.update(1)
                remaining_iter -= 1
                if np.random.rand() < acceptance_probs[s]:
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
            self.chain[self.iteration_num] = current_params
            self.likelihood_chain[self.iteration_num] = current_likelihood

        print(f"{acceptance_rate=}")
        pbar.close()
        self.save()

    def chain_to_plot_and_estimate(self,
                                   true_vals: Optional[np.ndarray[float]] = None):

        non_fixed_indexes = np.array(self.proposal_std, dtype=bool)
        chain = self.chain[:, :, non_fixed_indexes]
        param_names = self.param_names[non_fixed_indexes]
        true_vals = true_vals[non_fixed_indexes] if true_vals.any() else None
        likelihoods = self.likelihood_chain

        print("MCMC sampling completed.")

        plt.figure(figsize=(10, 8))
        plt.xlabel("Iteration #")
        x = np.arange(len(chain))
        plt.plot(x, likelihoods)
        plt.ylabel(r"Log Likelihoods")
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(nrows=chain.shape[2], ncols=chain.shape[1], figsize=(10, 8))
        axs = axs.reshape(chain[0].shape)
        fig.suptitle("Parameter Iterations")
        plt.xlabel("Iteration #")
        x = np.arange(len(chain))

        for body in range(chain.shape[1]):
            for i, name in enumerate(param_names):
                param_samples = chain[:, body, i]
                print(f"Estimated {name}: {np.mean(param_samples):.3e}",
                      f", true {name}: {true_vals[i]}" if true_vals is not None else None)
                axs[body, i].plot(x, param_samples, label=name)
                axs[body, i].set_ylabel(f"{name}")

        plt.tight_layout()
        plt.show()

    def corner_plot(self, true_vals: Optional[np.ndarray]=None, burn_in_index: int=0):
        non_fixed_indexes = np.array(self.proposal_std, dtype=bool)
        chain = self.chain[:, :, non_fixed_indexes]
        param_names = self.param_names[non_fixed_indexes]
        true_vals = true_vals[non_fixed_indexes] if true_vals.any() else None
        fig = corner(
            chain[burn_in_index:, 0],
            labels=param_names,
            truths=true_vals,
            show_titles=True,
            title_kwargs={"fontsize": 18},
            title_fmt=".2e"
        )
        plt.show()

    def determine_burn_in_index(self) -> int:
        """
        Determines the burn-in cutoff index for an MCMC chain.
        Returns:
            int: The burn-in cutoff index.
        """
        num_params = self.chain.shape[1]
        indices = []

        for i in range(num_params):
            samples = self.chain[:, i]
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
