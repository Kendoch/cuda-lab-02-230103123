import numpy as np
from numba import cuda


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:
        left = d_in[idx - 1] if idx > 0 else d_in[0]
        center = d_in[idx]
        right = d_in[idx + 1] if idx < N - 1 else d_in[N - 1]

        d_out[idx] = (
            0.25 * left
            + 0.5 * center
            + 0.25 * right
        )


def run_stencil(h_in):
    h_in = np.asarray(h_in, dtype=np.float32)
    N = len(h_in)

    if N == 0:
        return np.empty(0, dtype=np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    threads = 256
    blocks = (N + threads - 1) // threads

    stencil_1d[blocks, threads](d_in, d_out, N)

    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode="edge")

    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )


if __name__ == "__main__":
    N = 10007

    h_in = np.sin(
        np.linspace(0, 10, N)
    ).astype(np.float32)

    h_gpu = run_stencil(h_in)
    h_cpu = cpu_stencil(h_in)

    assert np.allclose(h_gpu, h_cpu, atol=1e-4)

    delta = np.max(np.abs(h_gpu - h_cpu))

    print("TASK 2 PASSED: MAX DELTA =", delta)
