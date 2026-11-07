"""Data types.  Provided complete -- you shouldn't need to change this until week 9.

A DType is a `str` subclass, so `dtypes.i32 == "i32"` and DTypes are hashable, printable,
and cheap to compare.  Extra facts hang off attributes.
"""
from __future__ import annotations
import numpy as np

class DType(str):
  itemsize: int; cname: str; np: type; kind: str
  def __new__(cls, name: str, itemsize: int, cname: str, np_type, kind: str):
    self = super().__new__(cls, name)
    self.itemsize, self.cname, self.np, self.kind = itemsize, cname, np_type, kind
    return self
  @property
  def is_float(self) -> bool: return self.kind == "f"
  @property
  def is_int(self) -> bool: return self.kind in ("i", "u")
  def __repr__(self): return f"dtypes.{str(self)}"

class dtypes:
  bool = DType("bool", 1, "bool", np.bool_, "b")
  i8   = DType("i8", 1, "signed char", np.int8, "i")
  u8   = DType("u8", 1, "unsigned char", np.uint8, "u")
  i32  = DType("i32", 4, "int", np.int32, "i")
  i64  = DType("i64", 8, "long long", np.int64, "i")
  f16  = DType("f16", 2, "half", np.float16, "f")          # week 9 (CUDA: __half)
  bf16 = DType("bf16", 2, "__bf16", None, "f")              # week 9 (CUDA: __nv_bfloat16); numpy has no bf16
  f32  = DType("f32", 4, "float", np.float32, "f")
  @staticmethod
  def from_name(name: str) -> DType:
    d = getattr(dtypes, name, None)
    if not isinstance(d, DType): raise KeyError(name)
    return d
  @staticmethod
  def from_np(t) -> DType:
    for d in vars(dtypes).values():
      if isinstance(d, DType) and d.np is not None and np.dtype(d.np) == np.dtype(t): return d
    raise KeyError(t)
