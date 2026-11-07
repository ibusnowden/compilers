"""Week 1: UOps, patterns, rewriting.  Start here.

Required interface (tests/week01/test_uop.py):

  Ops                      an Enum of operations (at least CONST, VAR, ADD, MUL, ... -- you choose the rest)
  UOp(op, dtype, src, arg) hash-consed and immutable: equal arguments give the *same object*
    .op .dtype .src .arg
    UOp.const(dtype, value)
    .toposort() -> list[UOp]       sources before users, each node once, no recursion limit
  UPat(op=None, dtype=None, src=None, name=None, arg=None)
    UPat.var(name)                 matches anything and binds it
  PatternMatcher([(UPat, fn), ...])   fn(**bindings) -> UOp | None
  graph_rewrite(sink, pm) -> UOp      bottom-up, to a fixed point, memoized

Design notes (docs/week01/lecture.md):
  * The hash-cons table must be a weak-value dictionary, or every UOp ever made lives forever.
  * Key floats by their bits: 0.0 and -0.0 are different constants; nan is one constant.
  * Keep `src` a tuple; keep `arg` hashable.
"""
from __future__ import annotations
from enum import Enum, auto

class Ops(Enum):
  CONST = auto()
  # ... add yours

class UOp:
  def __new__(cls, op, dtype, src=(), arg=None):
    raise NotImplementedError

class UPat:
  pass

class PatternMatcher:
  pass

def graph_rewrite(sink, pm):
  raise NotImplementedError
