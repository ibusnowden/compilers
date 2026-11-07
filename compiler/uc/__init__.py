"""uc -- a small ML compiler.  Grows every week; see docs/.

Week 6 adds the front end:  from uc import Tensor
"""
try:
  from uc.tensor import Tensor          # noqa: F401  (exists from week 6)
except ImportError:
  pass
from uc.dtype import dtypes             # noqa: F401
