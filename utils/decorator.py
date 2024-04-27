from utils.std_logger import logger
from functools import wraps
import time


def time_this(func):
    @wraps(func)
    def timing(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        msg = f"{func.__name__} runtime: {round(runtime, 6)} (s)"
        logger.info(msg)
        return result
    return timing
