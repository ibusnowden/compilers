"""Minimal safetensors reader for the *reference* side of tests (numpy + torch only)."""
import json, struct
import numpy as np

def load_torch(path):
  import torch
  with open(path, "rb") as f: data = f.read()
  n = struct.unpack("<Q", data[:8])[0]; header = json.loads(data[8:8 + n]); base = 8 + n
  out = {}
  for k, v in header.items():
    if k == "__metadata__": continue
    s, e = v["data_offsets"]; raw = bytearray(data[base + s: base + e])
    if v["dtype"] == "BF16": t = torch.frombuffer(raw, dtype=torch.bfloat16).float()
    else: t = torch.from_numpy(np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}[v["dtype"]]).copy()).float()
    out[k] = t.reshape(v["shape"])
  return out
