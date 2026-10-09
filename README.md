
# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103123  
**Allocated GPU Node:** NVIDIA Tesla T4
**CUDA Compute Capability:** 7.5 
**Official Verification Token:** 290BEABB9253F6A53806


TASK 1: WARP DIVERGENCE
Elements: 1048576
Iterations: 1000
Uniform: 21.6702 ms
Divergent: 93.9924 ms
Warp-Aligned: 46.9140 ms

### Observations

The uniform kernel executes the same operations for every thread.

The divergent kernel uses different branches for adjacent threads, which can reduce performance.

The warp-aligned kernel groups threads into warps that follow the same branch.

Actual performance depends on the measured results and GPU architecture.

## Task 2: 1D Stencil

Implemented a three-point smoothing filter with boundary clamping.

Grid size 40 will likely result in GPU under-utilization due to low occupancy.
TASK 2 PASSED: MAX DELTA = 5.9604645e-08



## Task 3: Grid-Stride Scaling
TASK 3 PASSED
Elements: 16777216
Threads launched: 16384
Expected factor: 4.25
First values: [4.25 4.25 4.25 4.25 4.25]

## Task 4: Sobel Horizontal Filter

TASK 4 PASSED
Image shape: (2048, 2048)
Maximum gradient: 0.0


============================================================
RUNNING CUDA LAB 02 AUTONOMOUS VERIFICATION
============================================================
Enter your Student ID: 230103123
/usr/local/lib/python3.13/dist-packages/numba_cuda/numba/cuda/dispatcher.py:716: NumbaPerformanceWarning: Grid size 40 will likely result in GPU under-utilization due to low occupancy.
  warn(errors.NumbaPerformanceWarning(msg))
[PASS] Task 2 (1D Stencil & Clamping)
/usr/local/lib/python3.13/dist-packages/numba_cuda/numba/cuda/dispatcher.py:716: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
  warn(errors.NumbaPerformanceWarning(msg))
[PASS] Task 3 (Grid-Stride Scaling)
/usr/local/lib/python3.13/dist-packages/numba_cuda/numba/cuda/dispatcher.py:716: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(errors.NumbaPerformanceWarning(msg))
[PASS] Task 4 (2D Sobel Horizontal)

============================================================
VERIFICATION SUCCESSFUL
OFFICIAL SUBMISSION TOKEN: 290BEABB9253F6A53806
============================================================

Official Verification Token: 290BEABB9253F6A53806
