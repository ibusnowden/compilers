"""Week 8: everything from week 6 must also run on the GPU.
  UC_DEVICE=CUDA pytest tests/week06 tests/week08
This file adds GPU-specific correctness checks."""
import numpy as np
import pytest
from uc import Tensor
from helpers import close, needs_cuda, skip_if_shim

pytestmark = [needs_cuda, skip_if_shim]
rng = np.random.default_rng(0)

def test_device_roundtrip():
  a = rng.standard_normal((3, 5)).astype(np.float32)
  t = Tensor(a, device="CUDA")
  assert t.device.startswith("CUDA")
  close(t, a); close(t.to("CPU"), a); close(Tensor(a).to("CUDA") * 2.0, a * 2)

@pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 1 << 20, (1 << 20) + 7])
def test_reduce_sizes(n):
  # sizes that aren't multiples of the block size are where GPU reductions break
  x = (np.arange(n) % 13).astype(np.float32)
  assert Tensor(x, device="CUDA").sum().item() == float(x.sum())

@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 33, 9), (128, 128, 128), (1000, 3, 1000), (1, 4096, 1)])
def test_matmul_shapes(shape):
  M, K, N = shape
  a, b = rng.standard_normal((M, K)).astype(np.float32), rng.standard_normal((K, N)).astype(np.float32)
  close(Tensor(a, device="CUDA") @ Tensor(b, device="CUDA"), a @ b, atol=1e-3, rtol=1e-3)

def test_big_grid():
  # more blocks than gridDim.y / gridDim.z allow: map large axes to x, or loop
  x = np.ones((70000, 3), np.float32)
  close((Tensor(x, device="CUDA") * 2.0).sum(axis=1), np.full(70000, 6.0))

def test_mixed_device_error():
  with pytest.raises(Exception):
    (Tensor([1.0], device="CUDA") + Tensor([1.0], device="CPU")).numpy()

def test_int_and_bool_on_gpu():
  x = np.arange(-50, 50, dtype=np.int32)
  t = Tensor(x, device="CUDA")
  assert np.array_equal(((t * 3) % 7).numpy(), np.fmod(x * 3, 7))   # C semantics
  assert np.array_equal((t > 0).numpy(), x > 0)
