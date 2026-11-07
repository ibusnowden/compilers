"""Week 8: competitive CUDA.  Targets are fractions of PyTorch on the same GPU (H100 assumed for the numbers in
the comments).  pytest tests/week08 -m perf -s"""
from uc import Tensor
from helpers import bench, perf, needs_cuda, skip_if_shim

pytestmark = [perf, needs_cuda, skip_if_shim]

def run(name, ours, theirs, need, flops=None, nbytes=None):
  import torch
  s = torch.cuda.synchronize
  to, tt = bench(ours, sync=s), bench(theirs, sync=s)
  extra = (f", {flops/to/1e12:.1f} TFLOP/s" if flops else "") + (f", {nbytes/to/1e9:.0f} GB/s" if nbytes else "")
  print(f"\n{name}: uc {to*1e6:.0f} us vs torch {tt*1e6:.0f} us = {tt/to:.0%} (need {need:.0%}){extra}")
  assert tt / to >= need

def test_add_bandwidth():
  import torch
  n = 1 << 28
  a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
  A, B = Tensor(a.cpu().numpy(), device="CUDA").realize(), Tensor(b.cpu().numpy(), device="CUDA").realize()
  run("add 256M f32", lambda: (A + B).realize(), lambda: a + b, need=0.8, nbytes=3 * 4 * n)

def test_sum_bandwidth():
  import torch
  n = 1 << 28
  a = torch.randn(n, device="cuda"); A = Tensor(a.cpu().numpy(), device="CUDA").realize()
  run("sum 256M f32", lambda: A.sum().realize(), lambda: a.sum(), need=0.7, nbytes=4 * n)

def test_softmax():
  import torch
  a = torch.randn(8192, 4096, device="cuda"); A = Tensor(a.cpu().numpy(), device="CUDA").realize()
  run("softmax 8192x4096", lambda: A.softmax().realize(), lambda: a.softmax(-1), need=0.6, nbytes=2 * 4 * a.numel())

def test_sgemm():
  import torch
  torch.backends.cuda.matmul.allow_tf32 = False      # compare FP32 SIMT against FP32 SIMT
  n = 4096
  a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
  A, B = Tensor(a.cpu().numpy(), device="CUDA").realize(), Tensor(b.cpu().numpy(), device="CUDA").realize()
  run("sgemm 4096 (no tensor cores)", lambda: (A @ B).realize(), lambda: a @ b, need=0.5, flops=2 * n ** 3)
