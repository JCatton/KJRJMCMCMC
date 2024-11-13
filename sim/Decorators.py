from time import perf_counter_ns
from functools import wraps


def TimeMeasure(func):
    """Measure performance of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter_ns()
        func_return = func(*args, **kwargs)
        finish_time = perf_counter_ns()
        print(
            f"Function: {func.__name__} took {(finish_time - start_time)*1e-6:.2f} ms"
        )
        return func_return

    return wrapper
