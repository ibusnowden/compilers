"""The entry point the test harness calls.  You will rewrite this function many times.

  week 1: parse -> UOps -> graph_rewrite(symbolic) -> every assert/prove must fold to true
  week 2: + linearize and run on an interpreter (backend="interp")
  week 3: + render C, compile, run (backend="c", the default from now on)
  week 4: + rangeify tensors into kernels; report how many kernels ran
  ...
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Result:
  kernels: int = 0     # number of kernel launches (week 4+)

def run_uir(src: str, backend: str | None = None) -> Result:
  raise NotImplementedError("week 1: implement run_uir")
