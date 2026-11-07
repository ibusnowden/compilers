"""Shared helpers for the Python tests (weeks 6+)."""
import os, time
import numpy as np
import pytest

DEVICE = os.environ.get("UC_DEVICE", "CPU")

def is_shim():
  import uc
  return getattr(uc, "SHIM", False)

skip_if_shim = pytest.mark.skipif("os.environ.get('UC_TORCH_SHIM') == '1'", reason="needs a real compiler (kernel counts / perf)")

def torch_or_skip():
  return pytest.importorskip("torch")

def has_cuda():
  try:
    import torch
    return torch.cuda.is_available()
  except ImportError:
    return False

needs_cuda = pytest.mark.skipif(not has_cuda(), reason="needs an NVIDIA GPU (and torch for reference timings)")
perf = pytest.mark.perf          # deselect with -m "not perf"
slow = pytest.mark.slow          # deselect with -m "not slow"

def close(a, b, atol=1e-5, rtol=1e-4):
  a = a.numpy() if hasattr(a, "numpy") and not isinstance(a, np.ndarray) else a
  b = b.numpy() if hasattr(b, "numpy") and not isinstance(b, np.ndarray) else b
  np.testing.assert_allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol, rtol=rtol)

def bench(fn, warmup=3, iters=10, sync=None):
  """median seconds per call; `sync` is called after each call (e.g. torch.cuda.synchronize)"""
  for _ in range(warmup): fn(); sync and sync()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter(); fn(); sync and sync(); ts.append(time.perf_counter() - t0)
  return float(np.median(ts))

def RefTensor():
  """The torch-backed reference implementation of the uc API (reference/torch_shim), as a class.
  Lets a test write a computation ONCE and run it on both uc and torch."""
  import importlib.util, pathlib, sys
  if "uc_torch_ref" not in sys.modules:
    p = pathlib.Path(__file__).resolve().parent.parent / "reference" / "torch_shim" / "uc" / "tensor.py"
    spec = importlib.util.spec_from_file_location("uc_torch_ref", p)
    m = importlib.util.module_from_spec(spec); sys.modules["uc_torch_ref"] = m; spec.loader.exec_module(m)
  return sys.modules["uc_torch_ref"].Tensor
