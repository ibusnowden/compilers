"""Week 6: the Tensor front end, checked against numpy.  Week 8 reruns this file with UC_DEVICE=CUDA."""
import numpy as np
import pytest
from uc import Tensor, dtypes
from helpers import DEVICE, close

rng = np.random.default_rng(0)
def r(*s): return rng.standard_normal(s).astype(np.float32)
def T(a, **kw): return Tensor(a, device=DEVICE, **kw)

def test_roundtrip_and_dtypes():
  assert T([1, 2, 3]).dtype == dtypes.i32
  assert T([1.0, 2.0]).dtype == dtypes.f32
  assert T([True, False]).dtype == dtypes.bool
  a = r(3, 4); assert np.array_equal(T(a).numpy(), a)
  assert T(3.5).item() == 3.5 and T([[1, 2]]).tolist() == [[1, 2]]
  assert T(a).shape == (3, 4) and T(a).ndim == 2 and T(a).numel() == 12

def test_creation():
  assert np.array_equal(Tensor.arange(5).numpy(), np.arange(5))
  assert np.array_equal(Tensor.arange(2, 11, 3).numpy(), [2, 5, 8])
  assert np.array_equal(Tensor.zeros(2, 3).numpy(), np.zeros((2, 3)))
  assert np.array_equal(Tensor.full((2, 2), 7).numpy(), np.full((2, 2), 7))
  assert np.array_equal(Tensor.eye(3).numpy(), np.eye(3))
  Tensor.manual_seed(1); a = Tensor.rand(1000).numpy()
  Tensor.manual_seed(1); b = Tensor.rand(1000).numpy()
  assert np.array_equal(a, b) and 0 <= a.min() and a.max() < 1 and abs(a.mean() - 0.5) < 0.05
  n = Tensor.randn(20000).numpy(); assert abs(n.mean()) < 0.05 and abs(n.std() - 1) < 0.05

@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "maximum", "minimum"])
def test_binary_broadcast(op):
  a, b = r(4, 1, 5), r(3, 1)
  f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "div": lambda x, y: x / y,
       "maximum": lambda x, y: x.maximum(y) if isinstance(x, Tensor) else np.maximum(x, y),
       "minimum": lambda x, y: x.minimum(y) if isinstance(x, Tensor) else np.minimum(x, y)}[op]
  close(f(T(a), T(b)), f(a, b))

def test_scalars_both_sides():
  a = r(3, 3)
  close(2.0 * T(a) - 1.0, 2 * a - 1); close(1.0 / (T(a) + 3.0), 1 / (a + 3)); close(T(a) ** 2, a ** 2)

@pytest.mark.parametrize("fn", ["exp", "log", "exp2", "log2", "sqrt", "rsqrt", "sin", "cos", "abs", "relu", "sigmoid", "tanh", "silu", "gelu", "reciprocal"])
def test_unary(fn):
  a = r(64)
  if fn in ("log", "log2", "sqrt", "rsqrt", "reciprocal"): a = np.abs(a) + 0.1
  ref = {"exp": np.exp, "log": np.log, "exp2": np.exp2, "log2": np.log2, "sqrt": np.sqrt, "rsqrt": lambda x: 1 / np.sqrt(x),
         "sin": np.sin, "cos": np.cos, "abs": np.abs, "relu": lambda x: np.maximum(x, 0), "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
         "tanh": np.tanh, "silu": lambda x: x / (1 + np.exp(-x)), "reciprocal": lambda x: 1 / x,
         "gelu": lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))}[fn]
  close(getattr(T(a), fn)(), ref(a), atol=1e-5, rtol=1e-4)

def test_compare_where():
  a, b = r(10), r(10)
  assert np.array_equal((T(a) < T(b)).numpy(), a < b)
  assert np.array_equal((T(a) >= 0.0).numpy(), a >= 0)
  assert (T(a) == T(a)).dtype == dtypes.bool
  close((T(a) > 0.0).where(T(a), 0.0), np.where(a > 0, a, 0))
  close(T(a).clip(-0.5, 0.5), np.clip(a, -0.5, 0.5))

def test_int_division_is_an_error():
  with pytest.raises(Exception): (T([1, 2]) / T([2, 2])).numpy()

@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reductions(axis, keepdim):
  a = r(3, 4, 5)
  for m in ("sum", "max", "min", "mean", "var"):
    got = getattr(T(a), m)(axis=axis, keepdim=keepdim)
    ref = getattr(np, m)(a, axis=axis, keepdims=keepdim)
    close(got, ref, atol=1e-5)

def test_argmax_softmax():
  a = r(5, 7)
  assert np.array_equal(T(a).argmax(axis=-1).numpy(), a.argmax(-1))
  e = np.exp(a - a.max(-1, keepdims=True)); sm = e / e.sum(-1, keepdims=True)
  close(T(a).softmax(), sm); close(T(a).log_softmax(), np.log(sm), atol=1e-5)
  close(T(a * 50).softmax(axis=0), np.exp(50 * a - (50 * a).max(0)) / np.exp(50 * a - (50 * a).max(0)).sum(0), atol=1e-6)  # stable
  close(T(a).logsumexp(), np.log(np.exp(a).sum(-1)))

def test_movement():
  a = r(2, 3, 4)
  close(T(a).reshape(6, -1), a.reshape(6, -1))
  close(T(a).permute(2, 0, 1), a.transpose(2, 0, 1))
  close(T(a[:, :1]).expand(2, 5, 4), np.broadcast_to(a[:, :1], (2, 5, 4)))
  close(T(a).pad(((1, 0), (0, 2), (1, 1))), np.pad(a, ((1, 0), (0, 2), (1, 1))))
  close(T(a).shrink(((0, 1), (1, 3), (0, 4))), a[0:1, 1:3, 0:4])
  close(T(a).flip(0, 2), a[::-1, :, ::-1])
  close(T(a).transpose(), a.swapaxes(-2, -1)); close(T(a).flatten(1), a.reshape(2, 12))
  close(T(a).unsqueeze(1).squeeze(1), a)

def test_getitem():
  a = r(4, 5, 6)
  close(T(a)[1], a[1]); close(T(a)[:, 2:4], a[:, 2:4]); close(T(a)[1:3, None, :, -1], a[1:3, None, :, -1])
  close(T(a)[-1, -2, -3], a[-1, -2, -3])

def test_gather_by_index_tensor():
  emb = r(10, 4); idx = np.array([[1, 3], [9, 0]], dtype=np.int32)
  close(T(emb)[T(idx)], emb[idx])   # embedding lookup

def test_cat_stack():
  a, b = r(2, 3), r(4, 3)
  close(Tensor.cat(T(a), T(b), dim=0), np.concatenate([a, b]))
  close(Tensor.stack([T(a), T(a)], dim=1), np.stack([a, a], 1))

@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((2, 3, 4), (4, 6)), ((2, 1, 3, 4), (5, 4, 2)), ((7,), (7,)), ((7,), (7, 2))])
def test_matmul(shapes):
  a, b = r(*shapes[0]), r(*shapes[1])
  close(T(a) @ T(b), a @ b, atol=1e-4)

def test_linear():
  x, w, b = r(5, 8), r(3, 8), r(3)
  close(T(x).linear(T(w), T(b)), x @ w.T + b, atol=1e-5)

@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv2d(stride, padding):
  torch = pytest.importorskip("torch")
  x, w, b = r(2, 3, 9, 9), r(4, 3, 3, 3), r(4)
  ref = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride, padding).numpy()
  close(T(x).conv2d(T(w), T(b), stride, padding), ref, atol=1e-4)

def test_pools():
  torch = pytest.importorskip("torch")
  x = r(1, 2, 8, 8); tx = torch.tensor(x)
  close(T(x).max_pool2d(2), torch.nn.functional.max_pool2d(tx, 2).numpy())
  close(T(x).max_pool2d(3, 2, 1), torch.nn.functional.max_pool2d(tx, 3, 2, 1).numpy())
  close(T(x).avg_pool2d(4), torch.nn.functional.avg_pool2d(tx, 4).numpy())

def test_norms():
  x = r(4, 16)
  close(T(x).layernorm(), (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5), atol=1e-5)
  close(T(x).rmsnorm(), x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6), atol=1e-5)

def test_cross_entropy():
  logits, labels = r(6, 10), np.array([0, 3, 9, 2, 2, 7], dtype=np.int32)
  lsm = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
  close(T(logits).cross_entropy(T(labels)), -lsm[np.arange(6), labels].mean(), atol=1e-5)

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_sdpa(causal, kv_heads):
  q, k, v = r(2, 4, 7, 8), r(2, kv_heads, 7, 8), r(2, kv_heads, 7, 8)
  kk, vv = np.repeat(k, 4 // kv_heads, 1), np.repeat(v, 4 // kv_heads, 1)
  s = q @ kk.swapaxes(-1, -2) / np.sqrt(8)
  if causal: s = np.where(np.tril(np.ones((7, 7), bool)), s, -np.inf)
  p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
  close(T(q).scaled_dot_product_attention(T(k), T(v), is_causal=causal), p @ vv, atol=1e-5)

def test_assign_is_versioned():
  """week 2's versioned memory, again: a tensor built BEFORE an assign sees the OLD value,
  even though nothing was computed yet when the assign happened."""
  old = r(4)
  a = T(old.copy())
  b = a + 1.0                       # lazy: nothing has run
  a.assign(T(np.zeros(4, np.float32)))
  c = a + 1.0
  close(b, old + 1.0)
  close(c, np.ones(4))
  close(a, np.zeros(4))

def test_cast():
  a = np.array([1.7, -1.7, 3.0], np.float32)
  assert np.array_equal(T(a).cast(dtypes.i32).numpy(), [1, -1, 3])
  close(T([1, 2]).cast(dtypes.f32) / 2.0, [0.5, 1.0])
