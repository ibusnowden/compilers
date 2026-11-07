"""Week 1: the UOp graph API.  See docs/week01/assignment.md for the required interface."""
import gc
from uc.dtype import dtypes
from uc.uop import UOp, Ops, UPat, PatternMatcher, graph_rewrite

i32, f32 = dtypes.i32, dtypes.f32

def c(v, dt=i32): return UOp.const(dt, v)

def test_const_hash_consed():
  assert c(3) is c(3)
  assert c(3) is not c(4)
  assert c(3) is not c(3.0, f32)

def test_alu_hash_consed():
  a = UOp(Ops.ADD, i32, (c(1), c(2)))
  b = UOp(Ops.ADD, i32, (c(1), c(2)))
  assert a is b
  assert UOp(Ops.ADD, i32, (c(2), c(1))) is not a   # hash-consing is structural, not semantic

def test_float_keys():
  assert c(0.0, f32) is not c(-0.0, f32)            # 0.0 == -0.0 in Python, but they are different values
  assert c(float("nan"), f32) is c(float("nan"), f32)  # nan != nan in Python, but it is one constant

def test_uops_are_immutable():
  a = UOp(Ops.ADD, i32, (c(1), c(2)))
  try:
    a.op = Ops.MUL
  except (AttributeError, TypeError):
    pass
  else:
    raise AssertionError("UOps must be immutable: hash-consing shares them")

def test_cache_does_not_leak():
  before = len(UOp._cache) if hasattr(UOp, "_cache") else None
  x = c(123456)
  for i in range(1000): x = UOp(Ops.ADD, i32, (x, c(1)))
  del x; gc.collect()
  if before is not None: assert len(UOp._cache) < before + 100, "use a weak-value cache"

def test_toposort():
  a, b = c(1), c(2)
  s = UOp(Ops.ADD, i32, (a, b))
  m = UOp(Ops.MUL, i32, (s, s))
  order = m.toposort()
  assert order[-1] is m
  assert len(order) == len(set(order)) == 4
  pos = {u: i for i, u in enumerate(order)}
  for u in order:
    for src in u.src: assert pos[src] < pos[u]

def test_toposort_deep():
  x = c(0)
  for i in range(20_000): x = UOp(Ops.ADD, i32, (x, c(1)))
  assert len(x.toposort()) == 20_002   # must not hit Python's recursion limit

def test_pattern_match_and_rewrite():
  x = UOp(Ops.VAR, i32, (), (0, 10, "x"))
  pm = PatternMatcher([
    # x * 2 -> x + x
    (UPat(Ops.MUL, src=(UPat.var("a"), UPat(Ops.CONST, arg=2))), lambda a: UOp(Ops.ADD, a.dtype, (a, a))),
  ])
  e = UOp(Ops.MUL, i32, (x, c(2)))
  assert graph_rewrite(e, pm) is UOp(Ops.ADD, i32, (x, x))

def test_named_pattern_must_bind_same_node():
  x = UOp(Ops.VAR, i32, (), (0, 10, "x"))
  y = UOp(Ops.VAR, i32, (), (0, 10, "y"))
  hits = []
  pm = PatternMatcher([(UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("a"))), lambda a: hits.append(a))])
  graph_rewrite(UOp(Ops.ADD, i32, (x, y)), pm)
  assert hits == []
  graph_rewrite(UOp(Ops.ADD, i32, (x, x)), pm)
  assert hits == [x]

def test_rewrite_to_fixed_point():
  # count down: k -> k-1 until 0
  pm = PatternMatcher([(UPat(Ops.CONST, name="k"), lambda k: c(k.arg - 1) if k.arg > 0 else None)])
  assert graph_rewrite(c(50), pm) is c(0)

def test_rewrite_is_bottom_up_and_memoized():
  calls = []
  def count(x): calls.append(x); return None
  pm = PatternMatcher([(UPat(Ops.ADD, name="x"), count)])
  x = c(1)
  for _ in range(60): x = UOp(Ops.ADD, i32, (x, x))   # 2^60 paths, 61 nodes
  graph_rewrite(x, pm)
  assert len(calls) == 60, f"each node should be visited once, got {len(calls)} visits"
