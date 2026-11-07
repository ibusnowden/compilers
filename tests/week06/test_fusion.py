"""Week 6 'speed': kernel fusion, the compile cache, and no wasted launches.  Needs your compiler (not the shim)."""
import numpy as np
from uc import Tensor, stats
from helpers import skip_if_shim, DEVICE, close

pytestmark = skip_if_shim
def T(a): return Tensor(a, device=DEVICE)
x = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)

def kernels(fn):
  stats.reset(); out = fn(); out.realize() if hasattr(out, "realize") else None; return stats.kernels

def test_elementwise_chain_is_one_kernel():
  assert kernels(lambda: ((T(x) * 2.0 + 1.0).relu() - T(x)).exp2()) <= 1

def test_reduce_fuses_its_input():
  assert kernels(lambda: (T(x) * T(x)).sum(axis=1)) <= 1

def test_softmax_kernels():
  assert kernels(lambda: T(x).softmax()) <= 3

def test_movement_ops_are_free():
  assert kernels(lambda: T(x).reshape(128, 64).permute(1, 0).flip(0).pad(((1, 1), (0, 0))) + 1.0) <= 1

def test_matmul_relu_is_one_kernel():
  w = np.random.default_rng(1).standard_normal((128, 32)).astype(np.float32)
  assert kernels(lambda: (T(x) @ T(w)).relu()) <= 1

def test_compile_cache():
  a = T(x)
  (a * 3.0 + 1.0).realize()
  stats.reset()
  b = T(x * 2)                      # different data, same program
  close(b * 3.0 + 1.0, x * 6 + 1)
  assert stats.compiles == 0, "same kernel shape: must come from the cache"

def test_realized_value_not_recomputed():
  y = (T(x) @ T(x.T)).realize()
  stats.reset()
  (y + 1.0).realize(); (y * 2.0).realize()
  assert stats.kernels <= 2         # y itself is not recomputed
