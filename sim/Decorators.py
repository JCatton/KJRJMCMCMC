from time import perf_counter
from functools import wraps
def TimeMeasure(func):
    '''Measure performance of a function'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'{"-"*40}')
        start_time = perf_counter()
        func(*args, **kwargs)
        finish_time = perf_counter()
        print(f'Function: {func.__name__}')
        print(f'Time elapsed in seconds: {finish_time - start_time:.2f}')
        print(f'{"-"*40}')
    return wrapper