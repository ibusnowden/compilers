#!/usr/bin/env python3
"""The PyTorch reference run for week 11.  Same model math as tests/ref/gpt2_torch.py, same data, same schedule.
  python benchmarks/train_gpt2_torch.py            # writes benchmarks/out/torch.csv
Eager mode by default; COMPILE=1 uses torch.compile (report both, compare to eager)."""
import csv, os, sys, time, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from ref import gpt2_torch, configs
from common import HP, Loader, lr_at, flops_per_token, PEAK_FLOPS

torch.manual_seed(HP["seed"])
cfg = dict(configs.GPT2_124M, vocab_size=50304)          # pad the vocab to a multiple of 128
sd = {k: v.cuda().requires_grad_(True) for k, v in gpt2_torch.init(cfg, seed=HP["seed"]).items()}
n_params = sum(v.numel() for k, v in sd.items() if k != "wpe.weight")
decay = [v for k, v in sd.items() if v.ndim >= 2]
no_decay = [v for k, v in sd.items() if v.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": HP["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=HP["lr"], betas=HP["betas"], fused=True)
fwd = gpt2_torch.forward
if os.environ.get("COMPILE") == "1": fwd = torch.compile(fwd)

B, T, A = HP["batch"], HP["seq_len"], HP["grad_accum"]
steps = HP["total_tokens"] // (B * T * A)
train, val = Loader("train", B, T), Loader("val", B, T)
out = pathlib.Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
log = csv.writer(open(out / "torch.csv", "w")); log.writerow(["step", "tokens", "train_loss", "val_loss", "tok_per_s", "mfu"])

def loss_fn(x, y):
  with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = fwd(sd, x, cfg)
  return torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

for step in range(steps + 1):
  if step % HP["eval_every"] == 0 or step == steps:
    with torch.no_grad():
      val.i, val.pos = 0, 0
      vl = sum(loss_fn(*(torch.from_numpy(a).cuda().long() for a in val.next())).item() for _ in range(HP["eval_tokens"] // (B * T)))
      vl /= HP["eval_tokens"] // (B * T)
    print(f"step {step}/{steps}  val {vl:.4f}")
  else: vl = ""
  if step == steps: break
  t0 = time.perf_counter()
  for g in opt.param_groups: g["lr"] = lr_at(step, steps)
  opt.zero_grad(set_to_none=True)
  tl = 0.0
  for _ in range(A):
    x, y = (torch.from_numpy(a).cuda().long() for a in train.next())
    loss = loss_fn(x, y) / A; loss.backward(); tl += loss.item()
  torch.nn.utils.clip_grad_norm_(list(sd.values()), HP["grad_clip"])
  opt.step(); torch.cuda.synchronize()
  dt = time.perf_counter() - t0
  tps = B * T * A / dt; mfu = tps * flops_per_token(n_params, cfg) / PEAK_FLOPS
  log.writerow([step, (step + 1) * B * T * A, f"{tl:.4f}", vl if vl == "" else f"{vl:.4f}", f"{tps:.0f}", f"{mfu:.3f}"])
  if step % 10 == 0: print(f"step {step:5d}  loss {tl:.4f}  {tps / 1e3:.0f}k tok/s  MFU {mfu:.1%}")
