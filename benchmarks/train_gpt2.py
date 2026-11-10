#!/usr/bin/env python3
"""Week 11: the same run as train_gpt2_torch.py, on YOUR compiler.  Writes benchmarks/out/uc.csv.
Required: bf16 compute with f32 master weights, fused flash attention (forward AND backward, or recompute),
gradient clipping, AdamW, and the JIT so a step doesn't re-schedule.  Plot benchmarks/out/{uc,torch}.csv together."""
import csv, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "compiler"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from uc import Tensor, optim
from uc.jit import jit
from uc.nn.state import get_parameters
from uc.models.gpt2 import GPT2
from ref import gpt2_torch, configs
from common import HP, Loader, lr_at, flops_per_token, PEAK_FLOPS

cfg = dict(configs.GPT2_124M, vocab_size=50304)
init = gpt2_torch.init(cfg, seed=HP["seed"])            # identical initialization to the reference run
model = GPT2.from_hf({k: Tensor(v.numpy(), device="CUDA") for k, v in init.items()}, cfg)
params = get_parameters(model)
n_params = sum(p.numel() for p in params) - model.wpe.weight.numel()
# like the reference run: weight decay on matrices only, never on biases and norm weights
opts = [optim.AdamW([p for p in params if p.ndim >= 2], lr=HP["lr"], betas=HP["betas"], weight_decay=HP["weight_decay"]),
        optim.AdamW([p for p in params if p.ndim < 2], lr=HP["lr"], betas=HP["betas"], weight_decay=0.0)]

B, T, A = HP["batch"], HP["seq_len"], HP["grad_accum"]
steps = HP["total_tokens"] // (B * T * A)

@jit
def micro_step(x, y):
  loss = model(x).reshape(-1, cfg["vocab_size"]).cross_entropy(y.reshape(-1)) / A
  loss.backward()
  return loss.realize()

@jit
def eval_step(x, y):
  with Tensor.no_grad(): return model(x).reshape(-1, cfg["vocab_size"]).cross_entropy(y.reshape(-1)).realize()

train, val = Loader("train", B, T), Loader("val", B, T)
out = pathlib.Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
log = csv.writer(open(out / "uc.csv", "w")); log.writerow(["step", "tokens", "train_loss", "val_loss", "tok_per_s", "mfu"])
def dev(a): return Tensor(a, device="CUDA")

for step in range(steps + 1):
  vl = ""
  if step % HP["eval_every"] == 0 or step == steps:
    val.i, val.pos = 0, 0
    n = HP["eval_tokens"] // (B * T)
    vl = sum(eval_step(*map(dev, val.next())).item() for _ in range(n)) / n
    print(f"step {step}/{steps}  val {vl:.4f}")
  if step == steps: break
  t0 = time.perf_counter()
  for opt in opts: opt.lr = lr_at(step, steps); opt.zero_grad()
  tl = sum(micro_step(*map(dev, train.next())).item() for _ in range(A))
  # TODO(you): clip the global grad norm to HP["grad_clip"] before stepping
  for opt in opts: opt.step()
  dt = time.perf_counter() - t0
  tps = B * T * A / dt; mfu = tps * flops_per_token(n_params, cfg) / PEAK_FLOPS
  log.writerow([step, (step + 1) * B * T * A, f"{tl:.4f}", vl if vl == "" else f"{vl:.4f}", f"{tps:.0f}", f"{mfu:.3f}"])
  if step % 10 == 0: print(f"step {step:5d}  loss {tl:.4f}  {tps / 1e3:.0f}k tok/s  MFU {mfu:.1%}")
