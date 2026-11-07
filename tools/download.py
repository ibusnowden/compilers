#!/usr/bin/env python3
"""Download data and weights into data/.   python tools/download.py mnist gpt2 resnet18 qwen fineweb
Uses only the standard library.  Everything lands in data/<name>/ (gitignored)."""
import os, sys, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"
HF = "https://huggingface.co"
SETS = {
  "mnist": [(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", f) for f in
            ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]],
  "gpt2": [(f"{HF}/openai-community/gpt2/resolve/main/model.safetensors", "model.safetensors")],
  "resnet18": [(f"{HF}/timm/resnet18.a1_in1k/resolve/main/model.safetensors", "model.safetensors")],
  # Apache-2.0 and ungated.  (Llama 3.2 needs a license click-through on HF and llama3 RoPE scaling: a good stretch goal.)
  "qwen": [(f"{HF}/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/{f}", f) for f in ["model.safetensors", "config.json", "tokenizer.json"]],
  # pre-tokenized GPT-2 tokens (llm.c format): 256 int32 header, then uint16 tokens. 100M tokens per shard.
  "fineweb": [(f"{HF}/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin", "fineweb_val_000000.bin")] +
             [(f"{HF}/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_train_{i:06d}.bin", f"fineweb_train_{i:06d}.bin")
              for i in range(1, int(os.environ.get("FINEWEB_SHARDS", "10")) + 1)],
}

def fetch(url, dst):
  if dst.exists(): print("have", dst); return
  dst.parent.mkdir(parents=True, exist_ok=True)
  tmp = dst.with_suffix(dst.suffix + ".part")
  print("get ", url)
  with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
    total, done = int(r.headers.get("Content-Length", 0)), 0
    while chunk := r.read(1 << 22):
      f.write(chunk); done += len(chunk)
      if total: print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
  print(); tmp.rename(dst)

if __name__ == "__main__":
  names = sys.argv[1:] or ["mnist"]
  for n in names:
    if n not in SETS: sys.exit(f"unknown dataset {n}; choose from {', '.join(SETS)}")
    for url, f in SETS[n]: fetch(url, ROOT / n / f)
