#!/usr/bin/env python3
"""
oracle.py -- an executable specification of UIR (docs/uir-spec.md).

This is NOT a compiler and is deliberately naive: it parses a .uir file and
evaluates it eagerly with numpy, binding ranges/vars to concrete integers and
enumerating them.  It exists so that you can check *your tests* before you
check *your compiler*:

    python reference/oracle.py tests/week01/*.uir          # check expectations
    python reference/oracle.py -v tests/week04/dot.uir     # print every asserted value

What it does NOT check, because it has no compiler inside:

  * `# max-kernels` (it has no kernels).
  * Provability.  `prove c` passes here if `c` is true for every value of its
    variables, and a load/store passes if every index it actually computes is
    in bounds.  Your compiler must *derive* both from the required rules
    (docs/uir-spec.md section 3).  So a test can pass the oracle and still be
    one that no sound compiler accepts: when you write a `prove` or a bounds
    test, check by hand that the required rules are enough.
"""
from __future__ import annotations
import itertools, math, re, sys
from dataclasses import dataclass, field
import numpy as np

class CompileError(Exception): pass
class AssertFailure(Exception): pass

DTYPES = {"bool": np.bool_, "i32": np.int32, "f32": np.float32}
ALU2 = {"add", "sub", "mul", "div", "mod", "max", "min", "cmplt", "cmpeq", "cmpne", "and", "or", "xor", "shl", "shr"}
ALU1 = {"neg", "not", "sqrt", "exp2", "log2", "sin", "recip"}
OPS = ALU2 | ALU1 | {"where", "cast", "var", "assert", "assertclose", "prove", "buffer", "store", "load", "range", "end",
                     "expand", "reduce", "index", "contiguous", "reshape", "permute", "pad", "shrink", "flip"}
KWARGS = {"reduce": {"op", "pop"}, "assertclose": {"atol", "rtol"}}
I32_MIN, I32_MAX = -2**31, 2**31 - 1

# ---------------------------------------------------------------- parsing
TOKEN = re.compile(r"\s*(?:(%[A-Za-z_][\w.]*)|(-?inf\b|nan\b|-?\d+\.\d*(?:[eE]-?\d+)?|-?\d+[eE]-?\d+)|(-?\d+)|([A-Za-z_][\w]*)|([()\[\]=]))")

@dataclass(eq=False)
class Node:
  op: str
  args: list = field(default_factory=list)   # Node | str (ident)
  kw: dict = field(default_factory=dict)
  value: object = None                       # literal payload
  name: str = ""
  line: int = 0
  ty: tuple | None = None                    # ("val", dtype, shape) | ("buf", dtype, n, bufid)

def tokenize(s: str, line: int):
  out, pos = [], 0
  s = s.rstrip()
  while pos < len(s):
    m = TOKEN.match(s, pos)
    if not m or m.end() == pos:
      if s[pos:].strip() == "": break
      raise CompileError(f"line {line}: cannot tokenize {s[pos:]!r}")
    pos = m.end()
    ref, flt, integer, ident, punct = m.groups()
    if ref: out.append(("ref", ref[1:]))
    elif flt: out.append(("float", flt))
    elif integer: out.append(("int", integer))
    elif ident: out.append(("ident", ident))
    else: out.append(("p", punct))
  return out

class Parser:
  def __init__(self, toks, env, line):
    self.t, self.i, self.env, self.line = toks, 0, env, line
  def peek(self, k=0): return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)
  def take(self):
    tok = self.peek(); self.i += 1
    if tok[0] is None: raise CompileError(f"line {self.line}: unexpected end of line")
    return tok
  def expect(self, p):
    tok = self.take()
    if tok != ("p", p): raise CompileError(f"line {self.line}: expected {p!r}, got {tok[1]!r}")

  def atom(self) -> Node | str:
    kind, v = self.take()
    if kind == "ref":
      if v not in self.env: raise CompileError(f"line {self.line}: undefined %{v}")
      return self.env[v]
    if kind == "int":
      x = int(v)
      if not I32_MIN <= x <= I32_MAX: raise CompileError(f"line {self.line}: i32 literal out of range")
      return Node("const", value=x, ty=("val", "i32", ()), line=self.line)
    if kind == "float": return Node("const", value=float(v), ty=("val", "f32", ()), line=self.line)
    if kind == "ident":
      if v in ("true", "false"): return Node("const", value=(v == "true"), ty=("val", "bool", ()), line=self.line)
      return v                                         # bare identifier (dtype name, op name...)
    if (kind, v) == ("p", "("):
      n = self.expr(); self.expect(")"); return n
    if (kind, v) == ("p", "["):
      items = []
      while self.peek() != ("p", "]"):
        items.append(self.atom())
      self.expect("]")
      return Node("list", value=items, line=self.line)
    raise CompileError(f"line {self.line}: unexpected token {v!r}")

  def expr(self) -> Node:
    kind, v = self.peek()
    if kind == "ident" and v in OPS:
      self.take()
      args, kw = [], {}
      while self.peek()[0] is not None and self.peek() != ("p", ")"):
        if self.peek()[0] == "ident" and self.peek(1) == ("p", "="):
          key = self.take()[1]; self.take(); kw[key] = self.atom()
        else: args.append(self.atom())
      n = Node(v, args, kw, line=self.line)
      if bad := set(kw) - KWARGS.get(v, set()):     # a misspelled kwarg must not silently fall back to a default
        raise CompileError(f"line {self.line}: {v} takes no keyword argument {sorted(bad)[0]!r}")
      typecheck(n)
      return n
    a = self.atom()
    if isinstance(a, str): raise CompileError(f"line {self.line}: unknown op {a!r}")
    if a.op == "list": a = list_to_tensor(a)
    return a

def parse(src: str):
  env, stmts, directives = {}, [], {}
  for ln, raw in enumerate(src.splitlines(), 1):
    code, _, comment = raw.partition("#")
    m = re.match(r"\s*([\w-]+)\s*:\s*(.*)", comment)
    if m and not code.strip(): directives[m.group(1)] = m.group(2).strip()
    if not code.strip(): continue
    m = re.match(r"\s*%([A-Za-z_][\w.]*)\s*=(.*)", code)
    if not m: raise CompileError(f"line {ln}: expected '%name = ...'")
    name, rest = m.groups()
    if name in env: raise CompileError(f"line {ln}: %{name} redefined")
    p = Parser(tokenize(rest, ln), env, ln)
    n = p.expr()
    if p.peek()[0] is not None: raise CompileError(f"line {ln}: trailing tokens")
    if not n.name: n.name = name
    env[name] = n
    stmts.append(n)
  return stmts, directives

# ---------------------------------------------------------------- static typing
def static_ints(a) -> list[int]:
  """A shape-like argument: a list literal of ints (possibly nested one level) or a single int."""
  if isinstance(a, Node) and a.op == "const" and a.ty[1] == "i32": return [a.value]
  if isinstance(a, Node) and a.op == "list":
    out = []
    for x in a.value: out.extend(static_ints(x) if not (isinstance(x, Node) and x.op == "list") else [static_ints(x)])
    return out
  raise CompileError(f"expected static integers, got {a!r}")

def list_to_tensor(n: Node) -> Node:
  def conv(x):
    if isinstance(x, Node) and x.op == "list": return [conv(y) for y in x.value]
    if isinstance(x, Node) and x.op == "const": return x
    raise CompileError(f"line {n.line}: list literals may only contain literals")
  nested = conv(n)
  flat, shape = [], []
  def walk(x, depth):
    if isinstance(x, list):
      if not x: raise CompileError(f"line {n.line}: empty list literal")
      if len(shape) <= depth: shape.append(len(x))
      elif shape[depth] != len(x): raise CompileError(f"line {n.line}: ragged list literal")
      for y in x: walk(y, depth + 1)
    else:
      if depth != len(shape): raise CompileError(f"line {n.line}: ragged list literal")
      flat.append(x)
  walk(nested, 0)
  dts = {c.ty[1] for c in flat}
  if len(dts) != 1: raise CompileError(f"line {n.line}: mixed dtypes in list literal")
  dt = dts.pop()
  arr = np.array([c.value for c in flat], dtype=DTYPES[dt]).reshape(shape)
  return Node("tensor", value=arr, ty=("val", dt, tuple(shape)), line=n.line)

def val(a, line):
  if not isinstance(a, Node): raise CompileError(f"line {line}: expected a value, got {a!r}")
  if a.op == "list": a = list_to_tensor(a)
  if a.ty[0] != "val": raise CompileError(f"line {line}: expected a value, got a buffer")
  return a

_bufid = itertools.count()
def typecheck(n: Node):
  L, op, a = n.line, n.op, n.args
  def need(k):
    if len(a) != k: raise CompileError(f"line {L}: {op} takes {k} arguments, got {len(a)}")
  if op in ALU2 or op in ALU1:
    need(2 if op in ALU2 else 1)
    n.args = a = [val(x, L) for x in a]
    dts, shapes = {x.ty[1] for x in a}, {x.ty[2] for x in a}
    if len(dts) != 1: raise CompileError(f"line {L}: {op}: dtype mismatch {sorted(dts)}")
    if len(shapes) != 1: raise CompileError(f"line {L}: {op}: shape mismatch {sorted(shapes)}")
    dt, shape = dts.pop(), shapes.pop()
    allowed = {"add": "i32 f32", "sub": "i32 f32", "mul": "i32 f32", "div": "i32 f32", "mod": "i32", "neg": "i32 f32",
               "max": "i32 f32", "min": "i32 f32", "cmplt": "bool i32 f32", "cmpeq": "bool i32 f32", "cmpne": "bool i32 f32",
               "and": "bool i32", "or": "bool i32", "xor": "bool i32", "not": "bool", "shl": "i32", "shr": "i32",
               "sqrt": "f32", "exp2": "f32", "log2": "f32", "sin": "f32", "recip": "f32"}[op].split()
    if dt not in allowed: raise CompileError(f"line {L}: {op} not defined for {dt}")
    n.ty = ("val", "bool" if op.startswith("cmp") else dt, shape)
    if op in ("div", "mod") and dt == "i32" and a[1].op in ("const", "tensor") and np.any(np.asarray(a[1].value) == 0):
      raise CompileError(f"line {L}: integer division by zero")
    if op in ("shl", "shr") and a[1].op in ("const", "tensor") and np.any((np.asarray(a[1].value) < 0) | (np.asarray(a[1].value) > 31)):
      raise CompileError(f"line {L}: shift amount out of range")
  elif op == "where":
    need(3); n.args = a = [val(x, L) for x in a]
    if a[0].ty[1] != "bool": raise CompileError(f"line {L}: where condition must be bool")
    if a[1].ty[1] != a[2].ty[1]: raise CompileError(f"line {L}: where: dtype mismatch")
    if len({x.ty[2] for x in a}) != 1: raise CompileError(f"line {L}: where: shape mismatch")
    n.ty = ("val", a[1].ty[1], a[1].ty[2])
  elif op == "cast":
    need(2)
    if a[1] not in DTYPES: raise CompileError(f"line {L}: cast: unknown dtype {a[1]!r}")
    n.args = a = [val(a[0], L), a[1]]
    n.ty = ("val", a[1], a[0].ty[2])
  elif op == "var":
    need(2); lo, hi = static_ints(a[0])[0], static_ints(a[1])[0]
    if lo > hi: raise CompileError(f"line {L}: var: empty interval")
    n.value, n.ty = (lo, hi), ("val", "i32", ())
  elif op == "range":
    need(1); N = static_ints(a[0])[0]
    if N <= 0: raise CompileError(f"line {L}: range size must be positive")
    n.value, n.ty = N, ("val", "i32", ())
  elif op in ("assert", "assertclose"):
    if op == "assert" and len(a) == 1:
      x = val(a[0], L)
      if x.ty[1] != "bool": raise CompileError(f"line {L}: assert with one argument needs a bool")
      n.args = [x]
    else:
      need(2); n.args = a = [val(x, L) for x in a]
      if a[0].ty[1:] != a[1].ty[1:]: raise CompileError(f"line {L}: {op}: type/shape mismatch {a[0].ty[1:]} vs {a[1].ty[1:]}")
      if op == "assertclose" and a[0].ty[1] != "f32": raise CompileError(f"line {L}: assertclose needs f32")
    n.ty = ("val", "bool", ())
  elif op == "prove":
    need(1); x = val(a[0], L)
    if x.ty[1:] != ("bool", ()): raise CompileError(f"line {L}: prove needs a scalar bool")
    n.args, n.ty = [x], ("val", "bool", ())
  elif op == "buffer":
    need(2)
    if a[0] not in DTYPES: raise CompileError(f"line {L}: buffer: unknown dtype")
    if isinstance(a[1], Node) and a[1].op == "list":
      t = list_to_tensor(a[1])
      if len(t.ty[2]) != 1 or t.ty[1] != a[0]: raise CompileError(f"line {L}: buffer initializer must be a flat list of {a[0]}")
      n.value, size = t.value.copy(), t.ty[2][0]
    else:
      size = static_ints(a[1])[0]
      if size <= 0: raise CompileError(f"line {L}: buffer size must be positive")
      n.value = np.zeros(size, dtype=DTYPES[a[0]])
    n.ty = ("buf", a[0], size, next(_bufid))
  elif op in ("store", "load"):
    need(3 if op == "store" else 2)
    b = a[0]
    if not isinstance(b, Node) or b.ty[0] != "buf": raise CompileError(f"line {L}: {op}: first argument must be a buffer")
    idx = val(a[1], L)
    if idx.ty[1:] != ("i32", ()): raise CompileError(f"line {L}: {op}: index must be a scalar i32")
    if op == "store":
      v = val(a[2], L)
      if v.ty[1:] != (b.ty[1], ()): raise CompileError(f"line {L}: store: value must be a scalar {b.ty[1]}")
      n.args, n.ty = [b, idx, v], b.ty
    else:
      n.args, n.ty = [b, idx], ("val", b.ty[1], ())
  elif op == "end":
    if len(a) < 2: raise CompileError(f"line {L}: end needs a store chain and at least one range")
    if not isinstance(a[0], Node) or a[0].ty[0] != "buf": raise CompileError(f"line {L}: end: first argument must be a buffer")
    for r in a[1:]:
      if not isinstance(r, Node) or r.op != "range": raise CompileError(f"line {L}: end: expected ranges")
    n.ty = a[0].ty
  elif op == "expand":
    x = val(a[0], L); shape = tuple(int(s) for s in flat_shape(a[1:]))
    xs = x.ty[2]
    if xs != () and (len(xs) != len(shape) or any(s != d and s != 1 for s, d in zip(xs, shape))):
      raise CompileError(f"line {L}: cannot expand {xs} to {shape}")
    n.args, n.value, n.ty = [x], shape, ("val", x.ty[1], shape)
  elif op == "reduce":
    need(1); x = val(a[0], L)
    rop = n.kw.get("op"); pop = n.kw.get("pop")
    if rop not in ("add", "mul", "max", "min"): raise CompileError(f"line {L}: reduce: bad op {rop!r}")
    pop = static_ints(pop)[0] if pop is not None else 1
    if not 0 <= pop <= len(x.ty[2]): raise CompileError(f"line {L}: reduce: pop={pop} on rank {len(x.ty[2])}")
    if x.ty[1] == "bool": raise CompileError(f"line {L}: reduce on bool")
    n.args, n.value, n.ty = [x], (rop, pop), ("val", x.ty[1], x.ty[2][:len(x.ty[2]) - pop])
  elif op == "index":
    x = val(a[0], L); idx = [static_ints(i)[0] for i in a[1:]]
    if len(idx) != len(x.ty[2]): raise CompileError(f"line {L}: index: need {len(x.ty[2])} indices")
    if any(not 0 <= i < d for i, d in zip(idx, x.ty[2])): raise CompileError(f"line {L}: index out of bounds")
    n.args, n.value, n.ty = [x], tuple(idx), ("val", x.ty[1], ())
  elif op == "contiguous":
    need(1); x = val(a[0], L); n.args, n.ty = [x], x.ty
  elif op in ("reshape", "permute", "pad", "shrink", "flip"):
    x = val(a[0], L); xs = x.ty[2]
    if op in ("reshape", "permute", "flip"): arg = tuple(flat_shape(a[1:]))
    else:
      need(2); arg = tuple(tuple(p) for p in static_ints(a[1]))
      if len(arg) != len(xs) or any(len(p) != 2 for p in arg): raise CompileError(f"line {L}: {op}: need one pair per axis")
    if op == "reshape":
      if any(s <= 0 for s in arg) or math.prod(arg) != math.prod(xs): raise CompileError(f"line {L}: cannot reshape {xs} to {arg}")
      out = arg
    elif op == "permute":
      if sorted(arg) != list(range(len(xs))): raise CompileError(f"line {L}: bad permutation {arg}")
      out = tuple(xs[p] for p in arg)
    elif op == "flip":
      if any(not 0 <= d < len(xs) for d in arg) or len(set(arg)) != len(arg): raise CompileError(f"line {L}: bad flip axes")
      out = xs
    elif op == "pad":
      if any(b < 0 or e < 0 for b, e in arg): raise CompileError(f"line {L}: negative padding")
      out = tuple(d + b + e for d, (b, e) in zip(xs, arg))
    else:
      if any(not 0 <= s <= e <= d for d, (s, e) in zip(xs, arg)): raise CompileError(f"line {L}: bad shrink")
      out = tuple(e - s for s, e in arg)
    n.args, n.value, n.ty = [x], arg, ("val", x.ty[1], out)
  else:
    raise CompileError(f"line {L}: unknown op {op}")

def flat_shape(args):
  out = []
  for x in args:
    s = static_ints(x)
    if any(isinstance(y, list) for y in s): raise CompileError("nested shape")
    out.extend(s)
  return out

# ---------------------------------------------------------------- dependencies
def free_syms(n: Node, memo=None) -> frozenset:
  """vars and ranges a node depends on, minus ranges closed by an `end` inside it."""
  memo = {} if memo is None else memo
  if id(n) in memo: return memo[id(n)]
  if n.op in ("var", "range"): r = frozenset([n])
  else:
    r = frozenset().union(*[free_syms(x, memo) for x in n.args if isinstance(x, Node)]) if n.args else frozenset()
    if n.op == "end": r = r - frozenset(n.args[1:])
  memo[id(n)] = r
  return r

# ---------------------------------------------------------------- evaluation
def wrap32(x):
  return ((np.asarray(x, dtype=np.int64) + 2**31) % 2**32 - 2**31).astype(np.int32)

def c_div(a, b):
  a64, b64 = a.astype(np.int64), b.astype(np.int64)
  if np.any(b64 == 0): raise CompileError("integer division by zero")
  q = np.abs(a64) // np.abs(b64) * np.sign(a64) * np.sign(b64)
  return wrap32(q)

def alu(op, xs, dt):
  with np.errstate(all="ignore"):
    a = xs[0]; b = xs[1] if len(xs) > 1 else None
    if dt == "i32" and op in ("add", "sub", "mul", "neg"):
      a64 = a.astype(np.int64); b64 = None if b is None else b.astype(np.int64)
      return wrap32({"add": lambda: a64 + b64, "sub": lambda: a64 - b64, "mul": lambda: a64 * b64, "neg": lambda: -a64}[op]())
    if op == "div": return c_div(a, b) if dt == "i32" else (a / b).astype(np.float32)
    if op == "mod":
      return wrap32(a.astype(np.int64) - c_div(a, b).astype(np.int64) * b.astype(np.int64))
    if op in ("shl", "shr"):
      if np.any((b < 0) | (b > 31)): raise CompileError("shift amount out of range")
      return wrap32(a.astype(np.int64) << b.astype(np.int64)) if op == "shl" else np.right_shift(a, b).astype(np.int32)
    f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "neg": np.negative, "max": np.maximum, "min": np.minimum,
         "cmplt": np.less, "cmpeq": np.equal, "cmpne": np.not_equal, "and": np.bitwise_and, "or": np.bitwise_or,
         "xor": np.bitwise_xor, "not": np.logical_not, "sqrt": np.sqrt, "exp2": np.exp2, "log2": np.log2, "sin": np.sin,
         "recip": np.reciprocal}[op]
    out = f(*xs)
    return np.asarray(out, dtype=np.bool_ if op.startswith("cmp") else DTYPES[dt])

class Evaluator:
  def __init__(self, verbose=False):
    self.live: dict[int, np.ndarray] = {}   # id(root node) -> current contents (inside an `end`)
    self.epoch = 0
    self.memo = {}
    self.verbose = verbose

  def ev(self, n: Node, env: dict):
    fs = free_syms(n)
    missing = [s for s in fs if s not in env]
    if missing:
      what = "range not closed" if any(s.op == "range" for s in missing) else "symbolic var used outside prove"
      raise CompileError(f"line {n.line}: {what} (%{missing[0].name})")
    key = (id(n), tuple(sorted((id(s), env[s]) for s in fs)), self.epoch)
    if key in self.memo: return self.memo[key]
    r = self._ev(n, env)
    if n.ty[0] == "val": r = np.asarray(r, dtype=DTYPES[n.ty[1]])
    self.memo[key] = r
    return r

  def _ev(self, n: Node, env):
    op = n.op
    if id(n) in self.live: return self.live[id(n)]
    if op in ("const", "tensor"): return np.asarray(n.value, dtype=DTYPES[n.ty[1]])
    if op in ("var", "range"): return np.int32(env[n])
    if op in ALU2 or op in ALU1: return alu(op, [self.ev(x, env) for x in n.args], n.args[0].ty[1])
    if op == "where":
      c, a, b = (self.ev(x, env) for x in n.args); return np.where(c, a, b)
    if op == "cast":
      x = self.ev(n.args[0], env); src = n.args[0].ty[1]; dst = n.args[1]
      if dst == "bool": return x != 0
      if src == "f32" and dst == "i32":
        with np.errstate(all="ignore"): return np.trunc(x).astype(np.int32)
      return x.astype(DTYPES[dst])
    if op == "buffer": return n.value.copy()
    if op in ("store", "load"):
      b = self.ev(n.args[0], env); i = int(self.ev(n.args[1], env))
      size = n.args[0].ty[2]
      if not 0 <= i < size: raise CompileError(f"line {n.line}: {op} index {i} out of bounds [0, {size})")
      if op == "load": return b[i]
      out = b.copy(); out[i] = self.ev(n.args[2], env); return out
    if op == "end":
      ranges = n.args[1:]
      root = self.chain_root(n.args[0])
      cur = self.ev(root, env).copy()
      outer_live = id(root) in self.live
      for vals in itertools.product(*[range(r.value) for r in ranges]):
        self.live[id(root)] = cur
        self.epoch += 1
        env2 = dict(env); env2.update(zip(ranges, vals))
        cur = self.ev(n.args[0], env2).copy()
      if outer_live: self.live[id(root)] = cur
      else: self.live.pop(id(root), None)
      self.epoch += 1
      return cur
    if op == "expand":
      x = self.ev(n.args[0], env); return np.broadcast_to(x, n.value).copy()
    if op == "reduce":
      x = self.ev(n.args[0], env); rop, pop = n.value
      if pop == 0: return x
      axes = tuple(range(x.ndim - pop, x.ndim))
      with np.errstate(all="ignore"):
        if rop == "add": r = np.sum(x, axis=axes, dtype=x.dtype)
        elif rop == "mul": r = np.prod(x, axis=axes, dtype=x.dtype)
        elif rop == "max": r = np.max(x, axis=axes)
        else: r = np.min(x, axis=axes)
      return r
    if op == "index": return self.ev(n.args[0], env)[n.value]
    if op == "contiguous": return self.ev(n.args[0], env)
    if op == "reshape": return self.ev(n.args[0], env).reshape(n.value)
    if op == "permute": return np.transpose(self.ev(n.args[0], env), n.value)
    if op == "flip": return np.flip(self.ev(n.args[0], env), n.value).copy()
    if op == "pad": return np.pad(self.ev(n.args[0], env), n.value)
    if op == "shrink":
      x = self.ev(n.args[0], env); return x[tuple(slice(s, e) for s, e in n.value)].copy()
    raise CompileError(f"line {n.line}: cannot evaluate {op}")

  def chain_root(self, n: Node):
    """Structural (never dependence-based, so value rewrites can't change it):
    unwrap directly nested `end`s, then walk down `store`s to the first buffer / end."""
    while n.op == "end": n = n.args[0]
    while n.op == "store": n = n.args[0]
    return n

  def check(self, n: Node):
    seen, stack = set(), list(n.args)
    while stack:   # every buffer version an assert reads must be closed
      x = stack.pop()
      if not isinstance(x, Node) or id(x) in seen: continue
      seen.add(id(x))
      if x.ty[0] == "buf":
        if free_syms(x): raise CompileError(f"line {n.line}: reads a store whose range is not closed by `end`")
        continue
      stack.extend(x.args)
    syms = sorted(free_syms(n), key=lambda s: s.line)
    if n.op != "prove" and any(s.op == "var" for s in syms):
      raise CompileError(f"line {n.line}: assert depends on a symbolic var")
    domains = [range(s.value[0], s.value[1] + 1) if s.op == "var" else range(s.value) for s in syms]
    total = math.prod(len(d) for d in domains)
    if total > 200_000:   # sample: endpoints + random points
      rng = np.random.default_rng(0)
      pts = list(itertools.product(*[(d[0], d[-1]) for d in domains]))
      pts += [tuple(int(rng.integers(d[0], d[-1] + 1)) for d in domains) for _ in range(20_000)]
    else: pts = itertools.product(*domains)
    for p in pts:
      env = dict(zip(syms, p))
      if n.op == "prove":
        if not bool(self.ev(n.args[0], env)): raise CompileError(f"line {n.line}: prove: false for {dict((s.name, v) for s, v in env.items())}")
        continue
      vals = [self.ev(x, env) for x in n.args]
      if self.verbose: print(f"  line {n.line}: {[v.tolist() for v in vals]}")
      if len(vals) == 1: ok = bool(np.all(vals[0]))
      elif n.op == "assert":
        ok = np.array_equal(vals[0], vals[1], equal_nan=(n.args[0].ty[1] == "f32"))
      else:
        atol = float(n.kw["atol"].value) if "atol" in n.kw else 1e-5
        rtol = float(n.kw["rtol"].value) if "rtol" in n.kw else 1e-5
        ok = bool(np.all(np.abs(vals[0] - vals[1]) <= atol + rtol * np.abs(vals[1])) or np.allclose(vals[0], vals[1], rtol, atol, equal_nan=True))
      if not ok: raise AssertFailure(f"line {n.line}: {n.op} failed: {[v.tolist() for v in vals]}" + (f" at {[(s.name, v) for s, v in env.items()]}" if env else ""))

def run(src: str, verbose=False):
  stmts, directives = parse(src)
  ev = Evaluator(verbose)
  for n in stmts:
    if n.op in ("assert", "assertclose", "prove"): ev.check(n)
  return directives

def check_file(path, verbose=False) -> tuple[bool, str]:
  src = open(path).read()
  expect = "pass"
  for line in src.splitlines():
    m = re.match(r"\s*#\s*expect\s*:\s*(\S+)", line)
    if m: expect = m.group(1)
  try:
    run(src, verbose); got = "pass"; msg = ""
  except CompileError as e: got, msg = "compile-error", str(e)
  except AssertFailure as e: got, msg = "assert-fail", str(e)
  return got == expect, f"expected {expect}, got {got}" + (f" ({msg})" if msg else "")

if __name__ == "__main__":
  args = sys.argv[1:]
  verbose = "-v" in args
  paths = [a for a in args if a != "-v"]
  bad = 0
  for p in paths:
    ok, msg = check_file(p, verbose)
    bad += not ok
    print(("ok   " if ok else "FAIL ") + p + ("" if ok and not verbose else "  " + msg))
  print(f"{len(paths) - bad}/{len(paths)} files behave as expected")
  sys.exit(1 if bad else 0)
