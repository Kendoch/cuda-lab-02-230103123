import numpy as np
import time
from numba import cuda

ITERATIONS = 1000
N = 1 << 20
THREADS = 256


@cuda.jit
def uniform_kernel(arr, out):
    idx = cuda.grid(1)

    if idx < arr.size:
        value = arr[idx]

        for j in range(ITERATIONS):
            value = value * 1.000001 + 0.000001

        out[idx] = value


@cuda.jit
def divergent_kernel(arr, out):
    idx = cuda.grid(1)

    if idx < arr.size:
        value = arr[idx]

        if idx % 2 == 0:
            for j in range(ITERATIONS):
                value = value * 1.000001 + 0.000001
        else:
            for j in range(ITERATIONS):
                value = (value - 0.000001) / 1.000001

        out[idx] = value


@cuda.jit
def warp_aligned_kernel(arr, out):
    idx = cuda.grid(1)

    if idx < arr.size:
        value = arr[idx]
        warp_id = idx // 32

        if warp_id % 2 == 0:
            for j in range(ITERATIONS):
                value = value * 1.000001 + 0.000001
        else:
            for j in range(ITERATIONS):
                value = (value - 0.000001) / 1.000001

        out[idx] = value


def benchmark(kernel, d_in, d_out):
    blocks = (d_in.size + THREADS - 1) // THREADS

    kernel[blocks, THREADS](d_in, d_out)
    cuda.synchronize()

    times = []

    for _ in range(10):
        start = time.perf_counter()

        kernel[blocks, THREADS](d_in, d_out)
        cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)


def main():
    h_in = np.ones(N, dtype=np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    print("TASK 1: WARP DIVERGENCE")
    print("Elements:", N)
    print("Iterations:", ITERATIONS)

    kernels = [
        ("Uniform", uniform_kernel),
        ("Divergent", divergent_kernel),
        ("Warp-Aligned", warp_aligned_kernel),
    ]

    for name, kernel in kernels:
        avg = benchmark(kernel, d_in, d_out)
        print(f"{name}: {avg:.4f} ms")


if __name__ == "__main__":
    main()
