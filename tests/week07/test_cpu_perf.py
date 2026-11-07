"""Week 7: CPU performance targets, relative to numpy on the same machine.
Run alone on a quiet machine:  pytest tests/week07 -m perf -s"""
import numpy as np
from uc import Tensor
from helpers import bench, perf, skip_if_shim

pytestmark = [perf, skip_if_shim]
rng = np.random.default_rng(0)

def report(name, ours, theirs, need):
  ratio = theirs / ours
  print(f"\n{name}: uc {ours*1e3:.2f} ms, numpy {theirs*1e3:.2f} ms -> {ratio:.0%} of numpy (need {need:.0%})")
  assert ratio >= need

def test_matmul_1024():
  a, b = rng.standard_normal((1024, 1024), dtype=np.float32), rng.standard_normal((1024, 1024), dtype=np.float32)
  A, B = Tensor(a).realize(), Tensor(b).realize()
  report("sgemm 1024", bench(lambda: (A @ B).realize()), bench(lambda: a @ b), need=0.25)

def test_add_bandwidth():
  a, b = rng.standard_normal(1 << 24, dtype=np.float32), rng.standard_normal(1 << 24, dtype=np.float32)
  A, B = Tensor(a).realize(), Tensor(b).realize()
  report("add 16M", bench(lambda: (A + B).realize()), bench(lambda: a + b), need=0.5)

def test_sum_bandwidth():
  a = rng.standard_normal(1 << 24, dtype=np.float32); A = Tensor(a).realize()
  report("sum 16M", bench(lambda: A.sum().realize()), bench(lambda: a.sum()), need=0.5)

def test_softmax_rows():
  a = rng.standard_normal((4096, 1024), dtype=np.float32); A = Tensor(a).realize()
  def np_softmax():
    e = np.exp(a - a.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
  report("softmax 4096x1024", bench(lambda: A.softmax().realize()), bench(np_softmax), need=1.0)
