import json, struct
import numpy as np
import torch
from uc.tensor import Tensor
from uc.dtype import dtypes

_ST = {"F32": dtypes.f32, "F16": dtypes.f16, "BF16": dtypes.bf16, "I32": dtypes.i32, "I64": dtypes.i64, "U8": dtypes.u8, "I8": dtypes.i8, "BOOL": dtypes.bool}
_ST_INV = {v: k for k, v in _ST.items()}

def safe_load(path):
  with open(path, "rb") as f: data = f.read()
  n = struct.unpack("<Q", data[:8])[0]
  header = json.loads(data[8:8 + n]); base = 8 + n
  out = {}
  for k, info in header.items():
    if k == "__metadata__": continue
    s, e = info["data_offsets"]; dt = _ST[info["dtype"]]; raw = data[base + s: base + e]
    if dt == dtypes.bf16:
      t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"])
      out[k] = Tensor._wrap(t.clone())
    else:
      out[k] = Tensor(np.frombuffer(raw, dtype=dt.np).reshape(info["shape"]).copy(), dtype=dt)
  return out

def safe_save(tensors, path):
  header, chunks, off = {}, [], 0
  for k, t in tensors.items():
    b = t.t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    header[k] = {"dtype": _ST_INV[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
    chunks.append(b); off += len(b)
  h = json.dumps(header).encode(); h += b" " * (-len(h) % 8)
  with open(path, "wb") as f: f.write(struct.pack("<Q", len(h)) + h + b"".join(chunks))

def get_state_dict(obj, prefix=""):
  if isinstance(obj, Tensor): return {prefix.rstrip("."): obj}
  if isinstance(obj, (list, tuple)): it = ((str(i), v) for i, v in enumerate(obj))
  elif isinstance(obj, dict): it = ((str(k), v) for k, v in obj.items())
  elif hasattr(obj, "__dict__"): it = vars(obj).items()
  else: return {}
  out = {}
  for k, v in it: out.update(get_state_dict(v, f"{prefix}{k}."))
  return out

def get_parameters(obj):
  seen, out = set(), []
  for t in get_state_dict(obj).values():
    if t.requires_grad and id(t) not in seen: seen.add(id(t)); out.append(t)
  return out

def load_state_dict(obj, sd, strict=True):
  mine = get_state_dict(obj)
  if strict:
    missing, extra = set(mine) - set(sd), set(sd) - set(mine)
    if missing or extra: raise KeyError(f"missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
  for k, t in mine.items():
    if k in sd:
      src = sd[k]
      if tuple(src.shape) != tuple(t.shape): raise ValueError(f"{k}: shape {src.shape} vs {t.shape}")
      t.assign(src)
