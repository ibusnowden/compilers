"""uc -- a small ML compiler.  Grows every week; see docs/.

Week 6 adds the front end:  from uc import Tensor.  You create uc/tensor.py then (docs/tensor-api.md):
a Tensor is a thin wrapper around a UOp; every method builds graph; realize() schedules, compiles, runs.
"""
try:
  from uc.tensor import Tensor          # noqa: F401  (you create it in week 6)
except ModuleNotFoundError as e:
  if e.name != "uc.tensor": raise       # tensor.py exists but one of ITS imports failed: show that error
from uc.dtype import dtypes             # noqa: F401
