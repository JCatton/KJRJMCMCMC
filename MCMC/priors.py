from typing import Self

import scipy
from numba import Callable


def uniform_transform(lower_bound: float, upper_bound: float, x: float) -> float:
    domain = upper_bound - lower_bound
    return lower_bound + x * domain

def gaussian_truncated_transform(lower_bound: float, upper_bound: float, mean: float, std: float, x: float) -> float:
    return scipy.stats.truncnorm.ppf(x, lower_bound, upper_bound, loc=mean, scale=std)

def dirac_delta_transform(fixed_val: float, x: float) -> float:
    return fixed_val


class Priors:
    def __init__(self, transform_func: Callable, config: dict[str, float]):
        self.transform_func = transform_func
        self.config = config

    def get

    @classmethod
    def get_truncated_gaussian(lower_bound: float, upper_bound: float, mean: float, std: float) -> Self:
        config = {
                "lower_bound":lower_bound,
                "upper_bound": upper_bound,
                "mean": mean,
                "std": std,
        }
        transform_func =
        return Priors()
