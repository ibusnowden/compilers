"""The uc.Tensor API (docs/tensor-api.md) implemented eagerly with PyTorch."""
from __future__ import annotations
import contextlib, os
import numpy as np
import torch
import torch.nn.functional as F
from uc.dtype import dtypes, DType

_T2U = {torch.float32: dtypes.f32, torch.float16: dtypes.f16, torch.bfloat16: dtypes.bf16, torch.int32: dtypes.i32,
        torch.int64: dtypes.i64, torch.uint8: dtypes.u8, torch.int8: dtypes.i8, torch.bool: dtypes.bool}
_U2T = {v: k for k, v in _T2U.items()}

def _dev(d):
  d = d or os.environ.get("UC_DEVICE", "CPU")
  if isinstance(d, tuple): return d, "cpu"
  if d.upper().startswith("CUDA") and torch.cuda.is_available(): return d.upper(), "cuda"
  return d.upper(), "cpu"

class Tensor:
  training = False
  _gen = torch.Generator().manual_seed(0)

  def __init__(self, data, dtype: DType | None = None, device=None, requires_grad=False):
    name, tdev = _dev(device)
    if isinstance(data, Tensor): t = data.t
    elif isinstance(data, torch.Tensor): t = data
    else:
      arr = np.asarray(data)
      if dtype is None:
        if arr.dtype.kind == "f": dtype = dtypes.f32
        elif arr.dtype.kind in "iu": dtype = dtypes.i32
        elif arr.dtype.kind == "b": dtype = dtypes.bool
      t = torch.from_numpy(np.ascontiguousarray(arr.astype(arr.dtype if dtype is None or dtype.np is None else dtype.np)))
    if dtype is not None: t = t.to(_U2T[dtype])
    self.t = t.to(tdev)
    self._device = name
    if requires_grad: self.t.requires_grad_(True)

  @staticmethod
  def _wrap(t, device="CPU"):
    out = Tensor.__new__(Tensor); out.t, out._device = t, device; return out
  def _w(self, t): return Tensor._wrap(t, self._device)

  # ---- creation
  @staticmethod
  def zeros(*shape, dtype=dtypes.f32, device=None, requires_grad=False): return Tensor(torch.zeros(_shape(shape), dtype=_U2T[dtype]), device=device, requires_grad=requires_grad)
  @staticmethod
  def ones(*shape, dtype=dtypes.f32, device=None, requires_grad=False): return Tensor(torch.ones(_shape(shape), dtype=_U2T[dtype]), device=device, requires_grad=requires_grad)
  @staticmethod
  def full(shape, value, dtype=None, device=None):
    dtype = dtype or (dtypes.f32 if isinstance(value, float) else dtypes.bool if isinstance(value, bool) else dtypes.i32)
    return Tensor(torch.full(_shape((shape,)), value, dtype=_U2T[dtype]), device=device)
  @staticmethod
  def empty(*shape, dtype=dtypes.f32, device=None): return Tensor.zeros(*shape, dtype=dtype, device=device)
  @staticmethod
  def arange(start, stop=None, step=1, dtype=None, device=None):
    if stop is None: start, stop = 0, start
    is_int = all(isinstance(x, int) for x in (start, stop, step))
    dtype = dtype or (dtypes.i32 if is_int else dtypes.f32)
    return Tensor(torch.arange(start, stop, step, dtype=_U2T[dtype]), device=device)
  @staticmethod
  def eye(n, dtype=dtypes.f32, device=None): return Tensor(torch.eye(n, dtype=_U2T[dtype]), device=device)
  @staticmethod
  def manual_seed(seed): Tensor._gen.manual_seed(seed)
  @staticmethod
  def rand(*shape, dtype=dtypes.f32, device=None, requires_grad=False): return Tensor(torch.rand(_shape(shape), generator=Tensor._gen).to(_U2T[dtype]), device=device, requires_grad=requires_grad)
  @staticmethod
  def randn(*shape, dtype=dtypes.f32, device=None, requires_grad=False): return Tensor(torch.randn(_shape(shape), generator=Tensor._gen).to(_U2T[dtype]), device=device, requires_grad=requires_grad)
  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, dtype=dtypes.f32, device=None, requires_grad=False):
    return Tensor((torch.rand(_shape(shape), generator=Tensor._gen) * (high - low) + low).to(_U2T[dtype]), device=device, requires_grad=requires_grad)

  # ---- properties
  @property
  def shape(self): return tuple(self.t.shape)
  @property
  def dtype(self): return _T2U[self.t.dtype]
  @property
  def device(self): return self._device
  @property
  def ndim(self): return self.t.ndim
  @property
  def requires_grad(self): return self.t.requires_grad
  @property
  def grad(self): return None if self.t.grad is None else self._w(self.t.grad)
  @property
  def T(self): return self.transpose()
  def numel(self): return self.t.numel()
  def numpy(self):
    t = self.t.detach().cpu()
    return (t.float() if t.dtype == torch.bfloat16 else t).numpy()
  def item(self): return self.t.item()
  def tolist(self): return self.t.tolist()
  def realize(self): return self
  def contiguous(self): return self._w(self.t.contiguous())
  def detach(self): return self._w(self.t.detach())
  def to(self, device):
    name, tdev = _dev(device); return Tensor._wrap(self.t.to(tdev), name)
  def cast(self, dtype): return self._w(self.t.to(_U2T[dtype]))
  def assign(self, other):
    o = other.t if isinstance(other, Tensor) else torch.as_tensor(other)
    with torch.no_grad(): self.t.copy_(o)
    return self
  def uir(self): return "# (torch shim: no UOps)"
  def shard(self, devices, axis=None):
    if axis is not None and self.shape[axis] % len(devices): raise ValueError("uneven shard")
    return Tensor._wrap(self.t, tuple(devices))
  def __repr__(self): return f"<Tensor {self.shape} {self.dtype} {self._device}>"

  # ---- autograd
  def backward(self):
    if self.t.numel() != 1: raise RuntimeError("backward needs a scalar")
    self.t.backward()
  @staticmethod
  @contextlib.contextmanager
  def no_grad():
    with torch.no_grad(): yield

  # ---- elementwise
  def _o(self, x):
    if isinstance(x, Tensor): return x.t
    return x
  def __add__(self, x): return self._w(self.t + self._o(x))
  def __radd__(self, x): return self._w(self._o(x) + self.t)
  def __sub__(self, x): return self._w(self.t - self._o(x))
  def __rsub__(self, x): return self._w(self._o(x) - self.t)
  def __mul__(self, x): return self._w(self.t * self._o(x))
  def __rmul__(self, x): return self._w(self._o(x) * self.t)
  def __truediv__(self, x):
    if not self.t.is_floating_point() and not (isinstance(x, float) or (isinstance(x, Tensor) and x.t.is_floating_point())):
      raise TypeError("int / int: cast first")
    return self._w(self.t / self._o(x))
  def __rtruediv__(self, x): return self._w(self._o(x) / self.t)
  def __neg__(self): return self._w(-self.t)
  def __pow__(self, k): return self._w(self.t ** k)
  def __matmul__(self, x): return self._w(self.t @ x.t)
  def __lt__(self, x): return self._w(self.t < self._o(x))
  def __le__(self, x): return self._w(self.t <= self._o(x))
  def __gt__(self, x): return self._w(self.t > self._o(x))
  def __ge__(self, x): return self._w(self.t >= self._o(x))
  def __eq__(self, x): return self._w(self.t == self._o(x))
  def __ne__(self, x): return self._w(self.t != self._o(x))
  __hash__ = object.__hash__
  def matmul(self, x): return self @ x
  def exp(self): return self._w(self.t.exp())
  def log(self): return self._w(self.t.log())
  def exp2(self): return self._w(torch.exp2(self.t))
  def log2(self): return self._w(torch.log2(self.t))
  def sqrt(self): return self._w(self.t.sqrt())
  def rsqrt(self): return self._w(self.t.rsqrt())
  def sin(self): return self._w(self.t.sin())
  def cos(self): return self._w(self.t.cos())
  def abs(self): return self._w(self.t.abs())
  def neg(self): return -self
  def reciprocal(self): return self._w(self.t.reciprocal())
  def relu(self): return self._w(F.relu(self.t))
  def sigmoid(self): return self._w(torch.sigmoid(self.t))
  def tanh(self): return self._w(torch.tanh(self.t))
  def silu(self): return self._w(F.silu(self.t))
  def gelu(self): return self._w(F.gelu(self.t, approximate="tanh"))
  def maximum(self, x): return self._w(torch.maximum(self.t, torch.as_tensor(self._o(x), dtype=self.t.dtype, device=self.t.device)))
  def minimum(self, x): return self._w(torch.minimum(self.t, torch.as_tensor(self._o(x), dtype=self.t.dtype, device=self.t.device)))
  def where(self, a, b):
    a = a.t if isinstance(a, Tensor) else a; b = b.t if isinstance(b, Tensor) else b
    return self._w(torch.where(self.t, a, b))
  def clip(self, lo, hi): return self._w(self.t.clamp(lo, hi))

  # ---- reductions
  def sum(self, axis=None, keepdim=False): return self._w(self.t.sum(dim=axis, keepdim=keepdim) if axis is not None else (self.t.sum().reshape([1] * self.ndim) if keepdim else self.t.sum()))
  def mean(self, axis=None, keepdim=False): return self._w(self.t.mean(dim=axis, keepdim=keepdim) if axis is not None else self.t.mean())
  def max(self, axis=None, keepdim=False): return self._w(self.t.amax(dim=axis, keepdim=keepdim) if axis is not None else self.t.max())
  def min(self, axis=None, keepdim=False): return self._w(self.t.amin(dim=axis, keepdim=keepdim) if axis is not None else self.t.min())
  def var(self, axis=None, keepdim=False): return self._w(self.t.var(dim=axis, keepdim=keepdim, correction=0) if axis is not None else self.t.var(correction=0))
  def argmax(self, axis=-1): return self._w(self.t.argmax(dim=axis).to(torch.int32))
  def softmax(self, axis=-1): return self._w(self.t.softmax(axis))
  def log_softmax(self, axis=-1): return self._w(self.t.log_softmax(axis))
  def logsumexp(self, axis=-1): return self._w(self.t.logsumexp(axis))

  # ---- movement
  def reshape(self, *shape): return self._w(self.t.reshape(_shape(shape)))
  def permute(self, *order): return self._w(self.t.permute(*_shape(order)))
  def expand(self, *shape): return self._w(self.t.expand(*_shape(shape)))
  def pad(self, padding):
    flat = [x for b, a in reversed(padding) for x in (b, a)]
    return self._w(F.pad(self.t, flat))
  def shrink(self, arg): return self._w(self.t[tuple(slice(s, e) for s, e in arg)])
  def flip(self, *axes): return self._w(self.t.flip(_shape(axes)))
  def transpose(self, a=-2, b=-1): return self._w(self.t.transpose(a, b))
  def flatten(self, start_dim=0): return self._w(self.t.flatten(start_dim))
  def unsqueeze(self, d): return self._w(self.t.unsqueeze(d))
  def squeeze(self, d): return self._w(self.t.squeeze(d))
  def __getitem__(self, idx):
    if isinstance(idx, Tensor): return self._w(self.t[idx.t.long()])
    return self._w(self.t[idx])
  @staticmethod
  def cat(*ts, dim=0): return ts[0]._w(torch.cat([t.t for t in ts], dim))
  @staticmethod
  def stack(ts, dim=0): return ts[0]._w(torch.stack([t.t for t in ts], dim))

  # ---- nn
  def linear(self, w, b=None): return self._w(F.linear(self.t, w.t, None if b is None else b.t))
  def conv2d(self, w, b=None, stride=1, padding=0): return self._w(F.conv2d(self.t, w.t, None if b is None else b.t, stride, padding))
  def max_pool2d(self, kernel=2, stride=None, padding=0): return self._w(F.max_pool2d(self.t, kernel, stride or kernel, padding))
  def avg_pool2d(self, kernel=2, stride=None): return self._w(F.avg_pool2d(self.t, kernel, stride or kernel))
  def layernorm(self, axis=-1, eps=1e-5):
    m = self.t.mean(axis, keepdim=True); v = self.t.var(axis, keepdim=True, correction=0)
    return self._w((self.t - m) * torch.rsqrt(v + eps))
  def rmsnorm(self, axis=-1, eps=1e-6): return self._w(self.t * torch.rsqrt((self.t * self.t).mean(axis, keepdim=True) + eps))
  def cross_entropy(self, labels): return self._w(F.cross_entropy(self.t, labels.t.long()))
  def scaled_dot_product_attention(self, k, v, is_causal=False):
    q, k, v = self.t, k.t, v.t
    if k.shape[1] != q.shape[1]:
      rep = q.shape[1] // k.shape[1]; k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
    return self._w(F.scaled_dot_product_attention(q, k, v, is_causal=is_causal))
  def dropout(self, p=0.5): return self._w(F.dropout(self.t, p, Tensor.training))

def _shape(s):
  if len(s) == 1 and isinstance(s[0], (tuple, list)): return tuple(s[0])
  return tuple(s)
