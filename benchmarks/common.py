"""Shared by the training scripts: hyperparameters, the llm.c token-file loader, and MFU."""
import glob, os, pathlib
import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "fineweb"
HP = dict(
  seq_len=1024, batch=16, grad_accum=32,          # 16 * 32 * 1024 = 524k tokens per step (GPT-2 paper scale)
  total_tokens=int(float(os.environ.get("TOTAL_TOKENS", 1e9))),
  lr=6e-4, min_lr=6e-5, warmup_steps=200, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0,
  eval_every=100, eval_tokens=10 * 1024 * 1024, seed=1337,
)
PEAK_FLOPS = float(os.environ.get("PEAK_TFLOPS", 989)) * 1e12   # H100 SXM dense bf16. PCIe: ~756

def load_shard(path):
  header = np.fromfile(path, dtype=np.int32, count=256)
  assert header[0] == 20240520, "not an llm.c token file"
  return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(int(header[2]),))

class Loader:
  def __init__(self, split, B, T, seed=0):
    self.files = sorted(glob.glob(str(DATA / f"fineweb_{split}_*.bin")))
    if not self.files: raise SystemExit("python tools/download.py fineweb")
    self.B, self.T, self.i, self.pos = B, T, 0, 0
    self.tok = load_shard(self.files[0])
  def next(self):
    n = self.B * self.T + 1
    if self.pos + n > len(self.tok):
      self.i = (self.i + 1) % len(self.files); self.tok = load_shard(self.files[self.i]); self.pos = 0
    buf = np.asarray(self.tok[self.pos:self.pos + n], dtype=np.int32); self.pos += self.B * self.T
    return buf[:-1].reshape(self.B, self.T), buf[1:].reshape(self.B, self.T)

def lr_at(step, total_steps):
  if step < HP["warmup_steps"]: return HP["lr"] * (step + 1) / HP["warmup_steps"]
  p = (step - HP["warmup_steps"]) / max(1, total_steps - HP["warmup_steps"])
  return HP["min_lr"] + 0.5 * (HP["lr"] - HP["min_lr"]) * (1 + np.cos(np.pi * p))

def flops_per_token(n_params, cfg):
  # 6N for the matmuls (fwd + bwd) + attention scores/values: 12 * layers * T * C   (PaLM appendix B)
  return 6 * n_params + 12 * cfg["n_layer"] * HP["seq_len"] * cfg["n_embd"]
