"""Check if GIL is released when computing filters."""

import os
import sys
import time
from threading import Thread

import numpy

import fastfilters2

DATA = numpy.zeros((128, 128, 128), dtype=numpy.float32)
SCALE = 10


def compute(iters=1):
    for _ in range(iters):
        fastfilters2.gaussian_smoothing(DATA, SCALE)


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def start_join(threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main():
    n = os.cpu_count()
    if n is None or n < 2:
        sys.exit(f"error: need at least 2 CPUs, found {n}")
    print(f"concurrency: {n}")

    single_threaded = timed(compute, n)
    multi_threaded = timed(start_join, [Thread(target=compute) for _ in range(n)])
    print(f"1 thread: {single_threaded:.3f} seconds")
    print(f"{n} threads: {multi_threaded:.3f} seconds")

    speedup = single_threaded / multi_threaded
    print(f"speedup: {speedup:.2f}x")
    if speedup < 1:
        sys.exit("fail: GIL is not released")


if __name__ == "__main__":
    main()
