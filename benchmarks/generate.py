#!/usr/bin/env python3
"""Week 11: batch-1 decode speed for Qwen2.5-1.5B on your compiler, against the memory-bandwidth bound.
  python tools/download.py qwen && python benchmarks/generate.py "The capital of France is"
Tokenization uses the `tokenizers` package (not part of the compiler)."""
import json, sys, time, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "compiler")); sys.path.insert(0, str(ROOT / "tests"))
from uc import Tensor
from uc.nn.state import safe_load
from uc.models.llama import Llama
from ref.configs import llama_config_from_hf

BW = float(__import__("os").environ.get("HBM_GBPS", 3350)) * 1e9     # H100 SXM HBM3. PCIe: ~2000
d = ROOT / "data" / "qwen"
cfg = llama_config_from_hf(json.load(open(d / "config.json")))
sd = safe_load(str(d / "model.safetensors"))                          # bf16 on disk; keep it bf16 on the GPU
model = Llama.from_hf({k: v.to("CUDA") for k, v in sd.items()}, cfg)
weight_bytes = sum(v.numel() * 2 for k, v in sd.items())

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(d / "tokenizer.json"))
prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
ids = tok.encode(prompt).ids
logits = model(Tensor(np.array([ids], np.int32), device="CUDA"), start_pos=0)
out, times = [], []
for i in range(64):
  nxt = int(logits[0, -1].argmax().item()); out.append(nxt)
  t0 = time.perf_counter()
  logits = model(Tensor(np.array([[nxt]], np.int32), device="CUDA"), start_pos=len(ids) + i)
  logits.realize(); times.append(time.perf_counter() - t0)
print(prompt + tok.decode(out))
tps = 1 / np.median(times[8:]); bound = BW / weight_bytes
print(f"\n{tps:.1f} tok/s; bandwidth bound {bound:.1f} tok/s -> {tps / bound:.0%} (need 50%)")
