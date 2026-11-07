#!/usr/bin/env python3
"""Weeks 7-9: run a sweep of kernels and print achieved FLOP/s and bytes/s next to the roofline.
  python benchmarks/roofline.py CPU      python benchmarks/roofline.py CUDA
Uses uc.stats.flops / uc.stats.bytes, which YOUR runtime fills in for each launch."""
import sys, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "compiler"))
from uc import Tensor, stats

dev = sys.argv[1] if len(sys.argv) > 1 else "CPU"
rng = np.random.default_rng(0)
def T(*s): return Tensor(rng.standard_normal(s).astype(np.float32), device=dev).realize()
cases = {
  "add 16M": lambda a=T(1 << 24), b=T(1 << 24): a + b,
  "sum 16M": lambda a=T(1 << 24): a.sum(),
  "softmax 4096x1024": lambda a=T(4096, 1024): a.softmax(),
  "gemm 512": lambda a=T(512, 512), b=T(512, 512): a @ b,
  "gemm 2048": lambda a=T(2048, 2048), b=T(2048, 2048): a @ b,
  "conv 64x3x3": lambda x=T(8, 64, 56, 56), w=T(64, 64, 3, 3): x.conv2d(w, padding=1),
}
print(f"{'kernel':22s} {'ms':>8s} {'GFLOP/s':>10s} {'GB/s':>8s} {'FLOP/B':>8s}")
for name, fn in cases.items():
  fn().realize()
  stats.reset(); t0 = time.perf_counter()
  for _ in range(5): fn().realize()
  dt = (time.perf_counter() - t0) / 5
  f, b = stats.flops / 5, stats.bytes / 5
  print(f"{name:22s} {dt*1e3:8.2f} {f/dt/1e9:10.1f} {b/dt/1e9:8.1f} {f/max(b,1):8.2f}")
