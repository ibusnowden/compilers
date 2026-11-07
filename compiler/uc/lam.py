"""Week 1 (lecture 1.3): the untyped lambda calculus, built on YOUR UOps and rewrite engine.

Required interface (tests/week01/test_lambda.py):
  parse(src) -> UOp        "\\x y. x" or "λx y. x"; application is juxtaposition; parens group
  app(f, x) -> UOp
  normalize(t, max_steps=10_000) -> UOp   normal-order (leftmost-outermost) reduction
  church(n) -> UOp, unchurch(t) -> int
  class NonTerminating(Exception)

Use de Bruijn indices: then alpha-equivalent terms are the same hash-consed node.
"""
class NonTerminating(Exception):
  pass
