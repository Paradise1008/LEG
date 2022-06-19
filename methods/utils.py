import functools
from time import time

def timing_func(keep_args=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            time_start = time()
            result = func(*args, **kwargs)
            time_end = time()
            if keep_args:
                print(f'func:{func.__name__}, args:{args, kwargs}, took:{time_end-time_start} seconds.\n')
            else:
                print(f'func:{func.__name__}, took:{time_end - time_start} seconds.\n')
            return result
        return wrapper
    return decorator


res = 1
@timing_func()
def multiplication(a, b):
    global res
    for i in range(10):
        res = res*a*b
    return res

res = multiplication(10,10)

