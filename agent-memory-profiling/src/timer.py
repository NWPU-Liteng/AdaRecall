import time
from contextlib import contextmanager


@contextmanager
def timer(stats: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        stats[key] = time.perf_counter() - start
