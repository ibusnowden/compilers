"""Week 6: reverse-mode autodiff, checked against torch.autograd."""
import numpy as np
import pytest
from uc import Tensor
from helpers import DEVICE, close, RefTensor

torch = pytest.importorskip("torch")
rng = np.random.default_rng(1)
def r(*s): return rng.standard_normal(s).astype(np.float32)

def grads(fn, *arrays):
  """run fn on uc and on torch, return (uc grads, torch grads)"""
  ut = [Tensor(a, device=DEVICE, requires_grad=True) for a in arrays]
  fn(*ut).backward()
  R = RefTensor()
  rt = [R(a, requires_grad=True) for a in arrays]
  fn(*rt).backward()
  return [u.grad.numpy() for u in ut], [t.grad.numpy() for t in rt]

def check(fn, *arrays, atol=1e-4):
  g, ref = grads(fn, *arrays)
  for a, b in zip(g, ref): close(a, b, atol=atol, rtol=1e-3)

CASES = {
  "add_mul": lambda a, b: (a * b + a).sum(),
  "div": lambda a, b: (a / (b * b + 1.0)).sum(),
  "broadcast": lambda a, b: (a + b[:1]).sum(),
  "exp_log": lambda a, b: ((a * 0.1).exp() + (b * b + 1.0).log()).sum(),
  "relu_max": lambda a, b: (a.relu() * b).max(),
  "reduce_axes": lambda a, b: (a.sum(axis=0) * b.mean(axis=0)).sum(),
  "matmul": lambda a, b: (a @ b.transpose(0, 1)).sum(),
  "expand": lambda a, b: (a[:1].expand(4, 4) * b).sum(),
  "where_cmp": lambda a, b: ((a > b).where(a * 2.0, b * 3.0)).sum(),
  "softmax": lambda a, b: (a.softmax(axis=-1) * b).sum(),
  "tanh_sigmoid": lambda a, b: (a.tanh() * b.sigmoid()).sum(),
  "movement": lambda a, b: (a.reshape(-1).flip(0).reshape(b.shape) * b.permute(1, 0).permute(1, 0)).sum(),
  "pad_shrink": lambda a, b: (a.pad(((1, 1), (0, 2))).shrink(((1, 4), (2, 6))) * b[:3]).sum(),
  "pow_sqrt": lambda a, b: ((a * a + 1.0).sqrt() + b ** 3).sum(),
}

@pytest.mark.parametrize("name", list(CASES))
def test_grad(name):
  check(CASES[name], r(4, 4), r(4, 4))

def test_reused_node_accumulates():
  check(lambda a: (a * a * a).sum() + a.sum(), r(5))

def test_max_ties():
  # gradient of max with ties: torch splits evenly among the tied elements; so must you
  a = np.array([1.0, 3.0, 3.0, 2.0], np.float32)
  check(lambda x: x.max(), a)

def test_cross_entropy_grad():
  labels = np.array([1, 0, 3], np.int32)
  u = Tensor(r(3, 4), requires_grad=True); t = torch.tensor(u.numpy(), requires_grad=True)
  u.cross_entropy(Tensor(labels)).backward()
  torch.nn.functional.cross_entropy(t, torch.tensor(labels).long()).backward()
  close(u.grad, t.grad.numpy(), atol=1e-5)

def test_conv_grad():
  x, w = r(1, 2, 6, 6), r(3, 2, 3, 3)
  ux, uw = Tensor(x, requires_grad=True), Tensor(w, requires_grad=True)
  ux.conv2d(uw, padding=1).relu().sum().backward()
  tx, tw = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
  torch.nn.functional.conv2d(tx, tw, padding=1).relu().sum().backward()
  close(ux.grad, tx.grad.numpy(), atol=1e-4); close(uw.grad, tw.grad.numpy(), atol=1e-4)

def test_embedding_grad():
  emb = r(6, 3); idx = np.array([1, 1, 4], np.int32)
  u = Tensor(emb, requires_grad=True); u[Tensor(idx)].sum().backward()
  ref = np.zeros_like(emb); np.add.at(ref, idx, 1.0)
  close(u.grad, ref)

def test_grads_accumulate_across_backward():
  u = Tensor(r(3), requires_grad=True)
  (u * 2.0).sum().backward(); (u * 3.0).sum().backward()
  close(u.grad, np.full(3, 5.0))

def test_no_grad_and_detach():
  u = Tensor(r(3), requires_grad=True)
  with Tensor.no_grad():
    y = (u * 2.0).sum()
  assert not y.requires_grad
  z = (u.detach() * u).sum(); z.backward()
  close(u.grad, u.detach().numpy())

def test_sgd_step():
  from uc import optim
  w = Tensor(np.array([1.0, -2.0], np.float32), requires_grad=True)
  opt = optim.SGD([w], lr=0.1)
  for _ in range(3):
    opt.zero_grad(); (w * w).sum().backward(); opt.step()
  close(w, np.array([1.0, -2.0]) * 0.8 ** 3, atol=1e-6)
