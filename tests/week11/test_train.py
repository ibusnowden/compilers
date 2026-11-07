"""Week 11: training must match PyTorch step for step (same init, same data, same AdamW)."""
import numpy as np
import pytest
from uc import Tensor, optim
from uc.nn.state import get_parameters
from helpers import close, DEVICE
from ref import configs

torch = pytest.importorskip("torch")
from ref import gpt2_torch  # noqa: E402

def test_gpt2_tiny_training_matches_torch():
  from uc.models.gpt2 import GPT2
  cfg, steps = configs.GPT2_TINY, 20
  sd = gpt2_torch.init(cfg, seed=0)
  # one fixed batch, seen every step: a model that works memorizes it, so the loss must fall
  batch = np.random.default_rng(0).integers(0, cfg["vocab_size"], (4, 17)).astype(np.int64)
  data = np.stack([batch] * steps)

  # reference
  ref_sd = {k: v.clone().requires_grad_(True) for k, v in sd.items()}
  topt = torch.optim.AdamW(list(ref_sd.values()), lr=3e-3, weight_decay=0.1)
  ref_losses = []
  for s in range(steps):
    x, y = torch.tensor(data[s, :, :-1]), torch.tensor(data[s, :, 1:])
    topt.zero_grad()
    loss = torch.nn.functional.cross_entropy(gpt2_torch.forward(ref_sd, x, cfg).reshape(-1, cfg["vocab_size"]), y.reshape(-1))
    loss.backward(); topt.step(); ref_losses.append(loss.item())

  # yours
  m = GPT2.from_hf({k: Tensor(v.numpy(), device=DEVICE) for k, v in sd.items()}, cfg)
  params = get_parameters(m)
  for p in params: assert p.requires_grad
  opt = optim.AdamW(params, lr=3e-3, weight_decay=0.1)
  losses = []
  Tensor.training = True
  for s in range(steps):
    x, y = Tensor(data[s, :, :-1].astype(np.int32), device=DEVICE), Tensor(data[s, :, 1:].astype(np.int32), device=DEVICE)
    opt.zero_grad()
    loss = m(x).reshape(-1, cfg["vocab_size"]).cross_entropy(y.reshape(-1))
    loss.backward(); opt.step(); losses.append(loss.item())
  Tensor.training = False

  assert ref_losses[-1] < ref_losses[0] - 0.3, "sanity: the reference learns"
  close(np.array(losses), np.array(ref_losses), atol=2e-3, rtol=2e-3)

def test_jit_replays_without_recompiling():
  from uc.jit import jit
  from uc import stats
  w = Tensor(np.ones((8, 8), np.float32), device=DEVICE).realize()
  @jit
  def f(x): return (x @ w).relu().sum(axis=1)
  xs = [np.random.default_rng(i).standard_normal((4, 8)).astype(np.float32) for i in range(5)]
  outs = []
  for i, x in enumerate(xs):
    if i == 2: stats.reset()
    outs.append(f(Tensor(x, device=DEVICE)).numpy())
  for x, o in zip(xs, outs): close(o, np.maximum(x @ np.ones((8, 8)), 0).sum(1), atol=1e-4)
  if not getattr(stats, "SHIM", False): assert stats.compiles == 0
