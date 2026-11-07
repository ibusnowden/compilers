"""Week 6: reading weights from disk.  safetensors = 8-byte little-endian header size, JSON header, raw bytes."""
import json, struct
import numpy as np
from uc import Tensor, dtypes
from uc.nn.state import safe_load, safe_save, get_state_dict, get_parameters, load_state_dict
from helpers import close

def write_safetensors(path, arrays, pad=True):
  header, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
  names = {np.float32: "F32", np.float16: "F16", np.int32: "I32", np.int64: "I64", np.uint8: "U8"}
  for k, a in arrays.items():
    b = np.ascontiguousarray(a).tobytes()
    header[k] = {"dtype": names[a.dtype.type], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
  h = json.dumps(header).encode()
  if pad: h += b" " * (-len(h) % 8)
  with open(path, "wb") as f: f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))

def test_load(tmp_path):
  arrays = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b.c": np.array([1, -2], np.int32),
            "h": np.array([0.5, 1.5], np.float16), "ids": np.array([7], np.int64)}
  write_safetensors(tmp_path / "x.safetensors", arrays)
  sd = safe_load(str(tmp_path / "x.safetensors"))
  assert set(sd) == set(arrays)
  for k, a in arrays.items():
    assert sd[k].shape == a.shape
    close(sd[k].cast(dtypes.f32) if sd[k].dtype != dtypes.f32 else sd[k], a.astype(np.float32))
  assert sd["b.c"].dtype == dtypes.i32 and sd["h"].dtype == dtypes.f16

def test_bf16(tmp_path):
  # bf16 = the top 16 bits of an f32
  f = np.array([1.0, -2.5, 3.140625], np.float32)
  raw = (f.view(np.uint32) >> 16).astype(np.uint16)
  h = json.dumps({"w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
  (tmp_path / "b.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + raw.tobytes())
  w = safe_load(str(tmp_path / "b.safetensors"))["w"]
  assert w.dtype == dtypes.bf16
  close(w.cast(dtypes.f32), f)

def test_roundtrip(tmp_path):
  t = {"x": Tensor(np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)), "y": Tensor([1, 2, 3])}
  safe_save(t, str(tmp_path / "r.safetensors"))
  back = safe_load(str(tmp_path / "r.safetensors"))
  close(back["x"], t["x"]); close(back["y"], t["y"])

def test_state_dict_names():
  from uc import nn
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 3); self.blocks = [nn.Linear(3, 3), nn.Linear(3, 2, bias=False)]
  n = Net()
  sd = get_state_dict(n)
  assert set(sd) == {"l1.weight", "l1.bias", "blocks.0.weight", "blocks.0.bias", "blocks.1.weight"}
  assert len(get_parameters(n)) == 5
  new = {k: Tensor(np.full(v.shape, 0.5, np.float32)) for k, v in sd.items()}
  load_state_dict(n, new)
  close(n.blocks[1].weight, np.full((2, 3), 0.5))
