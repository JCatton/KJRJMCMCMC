from typing import Self, Callable

import scipy


def uniform_transform(lower_bound: float, upper_bound: float, x: float) -> float:
    domain = upper_bound - lower_bound
    return lower_bound + x * domain

def uniform_density(lower_bound: float, upper_bound: float, x: float) -> float:
    domain = upper_bound - lower_bound
    return 1/domain if lower_bound <= x <= upper_bound else 0

def gaussian_truncated_transform(lower_bound: float, upper_bound: float, mean: float, std: float, x: float) -> float:
    lb_std = (lower_bound - mean) / std
    ub_std = (upper_bound - mean) / std
    return scipy.stats.truncnorm.ppf(x, lb_std, ub_std, loc=mean, scale=std)

def gaussian_truncated_density(lower_bound: float, upper_bound: float, mean: float, std: float, x: float) -> float:
    lb_std = (lower_bound - mean) / std
    ub_std = (upper_bound - mean) / std
    return scipy.stats.truncnorm.pdf(x, lb_std, ub_std, loc=mean, scale=std)

def dirac_delta_transform(fixed_val: float, x: float) -> float:
    return fixed_val

def dirac_delta_density(fixed_val: float, x: float) -> float:
    return float(fixed_val == x)


class Priors:
    def __init__(self, prior_func: Callable, transform_func: Callable, config: dict[str, float]):
        self.prior_func = prior_func
        self.transform_func = transform_func
        self.config = config

    @classmethod
    def get_prior_by_config(cls, prior_config: dict[str, ]) -> Self:
        if not (dist := prior_config.get("distribution")):
            raise ValueError("Prior distribution not provided, please provide in form {'distribution': ____ }")
        match dist:
            case "uniform":
                return cls.get_uniform_prior(prior_config["lower_bound"], prior_config["upper_bound"])
            case "gaussian":
                return cls.get_truncated_gaussian_prior(prior_config["lower_bound"], prior_config["upper_bound"],
                                                        prior_config["mean"], prior_config["std"])


    @classmethod
    def get_truncated_gaussian_prior(cls, lower_bound: float, upper_bound: float, mean: float, std: float) -> Self:
        config = {
                "lower_bound":lower_bound,
                "upper_bound": upper_bound,
                "mean": mean,
                "std": std,
        }
        transform_func = lambda x, config=config: gaussian_truncated_transform(**config, x=x)
        prior_func = lambda x, config=config: gaussian_truncated_density(**config, x=x)
        return Priors(prior_func, transform_func, config)

    @classmethod
    def get_uniform_prior(cls, lower_bound: float, upper_bound: float) -> Self:
        config = {
                "lower_bound":lower_bound,
                "upper_bound": upper_bound,
        }
        transform_func = lambda x, config=config: uniform_transform(**config, x=x)
        prior_func = lambda x, config=config: uniform_density(**config, x=x)
        return Priors(prior_func, transform_func, config)

    @classmethod
    def get_dirac_prior(cls, fixed_val: float) -> Self:
        config = {
                "fixed_val": fixed_val,
        }
        transform_func = lambda x, config=config: dirac_delta_transform(**config, x=x)
        prior_func = lambda x, config=config: dirac_delta_density(**config, x=x)
        return Priors(prior_func, transform_func, config)