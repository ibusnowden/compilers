# Week 8 assignment — GPUs

**Grade:** `python tools/grade.py 8`, which also reruns **all of week 6 with `UC_DEVICE=CUDA`**. Plus `pytest tests/week08 -m perf -s` in `REPORT.md`.

## Deliverables

1. `compiler/uc/runtime/cuda.py`: NVRTC + driver API through `ctypes`. Allocation, copies, launches, events, error messages.
2. `compiler/uc/renderer_cuda.py`: CUDA C, with `SPECIAL` (grid and block indices), shared memory, barriers, and warp shuffles.
3. Axis binding as an optimization action, with guards. Defaults for elementwise, reduce, and matmul-like kernels.
4. The in-block reduction scheduling rule (softmax-style).
5. **Your tests:** at least **10** in `tests/week08/mine/`: odd sizes, large grids, mixed dtypes on GPU, and a
   differential test that runs a random UIR program on CPU and GPU and compares the results.

## Perf targets (H100, vs PyTorch)

add ≥ 80% · sum ≥ 70% · softmax ≥ 60% · sgemm (TF32 off) ≥ 50%

## Stretch

* A HIP backend for an AMD GPU (same renderer, `hiprtc`).
* `UC_BEAM` on GPU: search block sizes, register tiles, and the shared-memory K tile.
