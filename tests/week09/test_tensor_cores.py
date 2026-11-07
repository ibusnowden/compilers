"""Week 9: half precision, tensor cores, fast GEMM, convolutions, flash attention."""
import numpy as np
import pytest
from uc import Tensor, dtypes, stats
from helpers import needs_cuda, skip_if_shim, bench, perf

pytestmark = [needs_cuda]
torch = pytest.importorskip("torch")
rng = np.random.default_rng(0)

def to_uc(t): return Tensor(t.float().cpu().numpy(), device="CUDA").cast({torch.float16: dtypes.f16, torch.bfloat16: dtypes.bf16, torch.float32: dtypes.f32}[t.dtype]).realize()

@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 512, 384), (1000, 1000, 1000), (64, 4096, 64), (4096, 64, 4096)])
def test_half_matmul_correct(dt, shape):
  M, K, N = shape
  a, b = torch.randn(M, K, device="cuda", dtype=dt), torch.randn(K, N, device="cuda", dtype=dt)
  ref = (a.float() @ b.float()).cpu().numpy()   # accumulate in f32, like a tensor core
  out = (to_uc(a) @ to_uc(b)).cast(dtypes.f32).numpy()
  tol = 2e-2 if dt == torch.bfloat16 else 5e-3
  np.testing.assert_allclose(out, ref, atol=tol * np.sqrt(K), rtol=tol)

def test_bf16_roundtrip():
  x = np.array([1.0, -2.5, 3.140625, 1e-3], np.float32)
  y = Tensor(x, device="CUDA").cast(dtypes.bf16).cast(dtypes.f32).numpy()
  ref = torch.tensor(x).bfloat16().float().numpy()
  assert np.array_equal(y, ref)

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T,D", [(1, 64), (17, 64), (256, 64), (1000, 128), (2048, 128)])
def test_flash_attention_correct(T, D, causal):
  B, H = 2, 8
  q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal).cpu().numpy()
  out = to_uc(q).scaled_dot_product_attention(to_uc(k), to_uc(v), is_causal=causal).cast(dtypes.f32).numpy()
  np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)

@skip_if_shim
def test_flash_attention_is_one_kernel():
  q = to_uc(torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16))
  q.realize(); stats.reset()
  q.scaled_dot_product_attention(q, q, is_causal=True).realize()
  assert stats.kernels <= 1, "QK^T, softmax and PV should be one fused kernel"

@skip_if_shim
def test_flash_attention_long_context_memory():
  # S x S scores for T = 65536, H = 16 would be 128 GiB in f32: only a fused kernel fits in 80 GB
  q = to_uc(torch.randn(1, 16, 65536, 64, device="cuda", dtype=torch.bfloat16))
  out = q.scaled_dot_product_attention(q, q, is_causal=True).cast(dtypes.f32)
  assert np.isfinite(out[0, 0, -1].numpy()).all()

def _run(name, ours, theirs, need, flops):
  s = torch.cuda.synchronize
  to, tt = bench(ours, sync=s), bench(theirs, sync=s)
  print(f"\n{name}: uc {flops/to/1e12:.0f} TFLOP/s, torch {flops/tt/1e12:.0f} TFLOP/s -> {tt/to:.0%} (need {need:.0%})")
  assert tt / to >= need

@perf
@skip_if_shim
def test_bf16_gemm_perf():
  n = 8192
  a, b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16), torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
  A, B = to_uc(a), to_uc(b)
  _run("bf16 gemm 8192", lambda: (A @ B).realize(), lambda: a @ b, need=0.5, flops=2 * n ** 3)

@perf
@skip_if_shim
def test_conv_perf():
  # a ResNet-50 3x3 layer: N=64, C=K=256, 14x14
  x = torch.randn(64, 256, 14, 14, device="cuda", dtype=torch.bfloat16)
  w = torch.randn(256, 256, 3, 3, device="cuda", dtype=torch.bfloat16)
  X, W = to_uc(x), to_uc(w)
  flops = 2 * 64 * 256 * 14 * 14 * 256 * 9
  _run("conv3x3 bf16", lambda: X.conv2d(W, padding=1).realize(), lambda: torch.nn.functional.conv2d(x, w, padding=1), need=0.3, flops=flops)

@perf
@skip_if_shim
def test_flash_attention_perf():
  B, H, T, D = 4, 16, 4096, 128
  q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  Q, K, V = to_uc(q), to_uc(k), to_uc(v)
  flops = 4 * B * H * T * T * D / 2
  _run("flash attention causal", lambda: Q.scaled_dot_product_attention(K, V, is_causal=True).realize(),
       lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), need=0.4, flops=flops)
