"""Week 7: optimizations must never change results.  Every kernel optimization (tiling, reordering,
unrolling, upcasting/vectorizing, threading) is a rewrite; this file fuzzes them."""
import numpy as np
import pytest
from uc import Tensor
from helpers import close, skip_if_shim

pytestmark = skip_if_shim
rng = np.random.default_rng(0)
def r(*s): return rng.standard_normal(s).astype(np.float32)

PROGRAMS = {
  "matmul": (lambda a, b: a @ b, lambda: (r(37, 53), r(53, 29))),           # awkward sizes: remainders everywhere
  "matmul_square": (lambda a, b: a @ b, lambda: (r(64, 64), r(64, 64))),
  "batched": (lambda a, b: (a @ b).relu().sum(axis=-1), lambda: (r(3, 17, 9), r(9, 31))),
  "softmax": (lambda a, b: (a + b).softmax(axis=-1), lambda: (r(33, 70), r(33, 70))),
  "conv": (lambda a, b: a.conv2d(b, padding=1), lambda: (r(2, 3, 13, 11), r(5, 3, 3, 3))),
  "reduce_axis0": (lambda a, b: (a * b).sum(axis=0), lambda: (r(101, 7), r(101, 7))),
  "transpose_add": (lambda a, b: a.transpose() + b, lambda: (r(45, 67), r(67, 45))),
}

def ref(name):
  fn, mk = PROGRAMS[name]
  args = mk()
  with pytest.MonkeyPatch.context() as mp:
    mp.setenv("UC_OPT", "0")
    base = fn(*[Tensor(a) for a in args]).numpy()
  return fn, args, base

@pytest.mark.parametrize("name", list(PROGRAMS))
def test_opt_on_off_agree(name, monkeypatch):
  fn, args, base = ref(name)
  monkeypatch.setenv("UC_OPT", "1")
  close(fn(*[Tensor(a) for a in args]), base, atol=1e-4)

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("name", list(PROGRAMS))
def test_fuzzed_opts(name, seed, monkeypatch):
  fn, args, base = ref(name)
  monkeypatch.setenv("UC_OPT_FUZZ", str(seed))
  close(fn(*[Tensor(a) for a in args]), base, atol=1e-4)

@pytest.mark.parametrize("threads", ["1", "2", "7"])
def test_threads(threads, monkeypatch):
  fn, args, base = ref("matmul")
  monkeypatch.setenv("UC_THREADS", threads)
  close(fn(*[Tensor(a) for a in args]), base, atol=1e-4)

def test_reduction_across_threads(monkeypatch):
  # a full reduction split across threads needs a second pass (or atomics); either way, the answer is exact here
  monkeypatch.setenv("UC_THREADS", "4")
  x = np.arange(1 << 20, dtype=np.int32) % 7
  assert Tensor(x).sum().item() == int(x.sum())
