# UIR — the course's intermediate representation

UIR ("UOp IR") is the text format of every `.uir` test. Your compiler
parses it into a hash-consed graph of **UOps** and then compiles and runs it.
The language grows each week. **An op is legal only from the week it is introduced.**
Code that relies on a later week's feature is still valid input once you reach that week.
Nothing is ever removed.

> The semantics in this document are executable: `reference/oracle.py` is a slow,
> straightforward interpreter of exactly this spec. When you write your own tests
> ("tests due" each week), run them through the oracle first:
> `python reference/oracle.py tests/week01/mine/*.uir`.
> If this document and the oracle disagree, it's a bug. Fix both.

---

## 1. Lexical structure

```
program   := line*
line      := [stmt] [comment] NEWLINE
comment   := '#' any-text
stmt      := '%' NAME '=' expr
expr      := OP arg* kwarg*        # an operation
           | atom                  # alias / literal:  %a = 5
arg       := atom | IDENT          # IDENT: dtype names, e.g. f32
atom      := '%' NAME | INT | FLOAT | 'true' | 'false'
           | '(' expr ')'          # nested, anonymous expression
           | '[' atom* ']'         # list literal (week 2: buffer init, week 4+: tensors)
kwarg     := IDENT '=' (atom | IDENT)
NAME      := [A-Za-z_][A-Za-z0-9_.]*
INT       := -?[0-9]+
FLOAT     := -?[0-9]+ '.' [0-9]* ([eE] -?[0-9]+)? | -?[0-9]+ [eE] -?[0-9]+ | inf | -inf | nan
```

* One statement per line. A name may be defined only once (SSA). Using a
  name before its definition is a compile error.
* Every statement must be well-typed even if nothing uses it. Every `assert`
  and `prove` in a file is checked. Other statements can be dead code and the compiler may delete them.
* Tests can carry **directives** in comments at the top of the file:

| directive | meaning |
|---|---|
| `# expect: pass` | (default) compiles, runs, every assert holds |
| `# expect: compile-error` | compiler must raise `CompileError` (bad types, out-of-bounds, unprovable `prove`, …) |
| `# expect: assert-fail` | compiles, but at least one `assert` must fail at run time with `AssertFailure` |
| `# max-kernels: N` | (week 4+) the run must launch at most N kernels |
| `# week: N` | informational: the first week whose features this test needs |

## 2. Types

| dtype | literal | notes |
|---|---|---|
| `bool` | `true`, `false` | |
| `i32`  | `3`, `-7` | two's-complement, **wraps on overflow** |
| `f32`  | `1.5`, `2.0`, `1e-3`, `inf`, `nan` | IEEE-754 binary32, round-to-nearest-even. *Every* intermediate result is rounded to f32 |

There is **no implicit promotion**. `add 1 2.0` is a compile error. Use `cast`.
Weeks 6+ add `f16`, `bf16`, `i8`, `u8`, `i64` through the Python Tensor API (see `tensor-api.md`).

## 3. Week 1 — scalar ALU, symbolic variables, proofs

| op | signature | semantics |
|---|---|---|
| `add a b`, `sub a b`, `mul a b` | T,T→T (T ∈ i32,f32) | i32 wraps mod 2³² |
| `neg a` | T→T | |
| `div a b` | i32: truncates toward zero (C `/`). f32: IEEE division | i32 division by a zero constant is a **compile error**. f32 `/0` gives inf/nan |
| `mod a b` | i32 only | C `%`: sign follows the dividend. `mod -7 2 = -1` |
| `max a b`, `min a b` | T,T→T | for f32, if either is nan the result is nan |
| `cmplt a b`, `cmpeq a b`, `cmpne a b` | T,T→bool (T ∈ bool,i32,f32) | IEEE: every comparison with nan is false, except `cmpne`, which is true |
| `and a b`, `or a b`, `xor a b` | T,T→T (T ∈ bool,i32) | bitwise for i32 |
| `not a` | bool→bool | |
| `shl a b`, `shr a b` | i32,i32→i32 | `shr` is arithmetic. A shift amount outside [0,31] that is a constant is a compile error |
| `where c a b` | bool,T,T→T | select |
| `cast a DT` | any→DT | f32→i32 truncates toward zero (out-of-range is undefined; tests avoid it). bool→num gives 0/1. num→bool means `≠ 0` |
| `sqrt a`, `exp2 a`, `log2 a`, `sin a`, `recip a` | f32→f32 | libm semantics, checked with `assertclose` |
| `var LO HI` | →i32 | a **symbolic** integer with LO ≤ v ≤ HI (inclusive, constants) |
| `assert a b` | | checks `a == b` (same dtype, elementwise for tensors; f32 compared exactly, but `nan == nan` counts as equal here) |
| `assert c` | | `c` must be `bool` and true |
| `assertclose a b [atol=1e-5] [rtol=1e-5]` | f32 | `|a-b| ≤ atol + rtol·|b|` elementwise |
| `prove c` | bool | **compile-time** check: your simplifier must rewrite `c` to the constant `true`. If it reaches `false`, or gets stuck, raise `CompileError` |

Rules for week 1:

* `var` may only feed `prove` (and, from week 2 on, it may also feed bounds proofs). An `assert` whose operands depend on a `var` is a compile error.
  The same holds for `range` from week 3: an `assert` may depend on an open `range`, but not on a `var`.
* Constant folding must use the **target** semantics: i32 wraparound, truncating division, and f32 rounding at every step.
  Folding `16777216.0 + 1.0` with Python floats gives the wrong answer.
* Rewrites must be **sound**. For example, `cmpeq x x → true` is wrong for f32 (`x` may be nan),
  and `cmplt x (add x 1) → true` is wrong for i32 (`x` may be `2147483647`).

### The required simplification rules (for `prove`)

`prove` tests use only facts that follow from this rule set. You may implement more rules, but every extra rule must be sound.

1. **Constant folding** of every op, with the target semantics.
2. **Identities** (i32 and bool only): `x+0`, `x*1`, `x*0→0`, `x-x→0`, `x/1`, `x%1→0`,
   `max(x,x)`, `min(x,x)`, `where(true,a,b)→a`, `where(false,a,b)→b`, `where(c,a,a)→a`,
   `cmpeq(x,x)→true`, `cmpne(x,x)→false`, `cmplt(x,x)→false`, `and(x,true)`, `or(x,false)`, `not(not x)`.
3. **Canonical form for i32 `add`/`mul`.** Constants go on the right. Sums are flattened and
   *like terms are collected* (`x*2 + x*3 → x*5`, `x + x → x*2`). Constant terms are combined
   (`(x+3)+4 → x+7`). All of this is sound mod 2³², so it needs no bounds.
4. **Bounds (interval) analysis.** Each i32 node has `[vmin, vmax]`. These are computed **exactly** and
   widened to the full i32 range whenever the exact result could leave it.
   * `cmplt a b → true` if `vmax(a) < vmin(b)`. `→ false` if `vmin(a) ≥ vmax(b)`.
   * `cmpeq a b → false` if the intervals don't overlap.
   * `max a b → a` if `vmin(a) ≥ vmax(b)` (and symmetrically). Likewise for `min`.
   * any i32 node whose interval is a single point `[c, c]` → `c`.
5. **Division and modulo by a positive constant `c`.**
   For `e = a*c + b`, with `vmin(e) ≥ 0`, `vmin(b) ≥ 0`, and `e` provably not overflowing:
   `e / c → a + b/c` and `e % c → b % c`.
   Also `b % c → b` whenever `0 ≤ vmin(b)` and `vmax(b) < c`.
   The interval rule above then finishes jobs like `(4x+3)/4 → x` when `x ≥ 0`.

> These are exactly the rules you need in week 5 to simplify the index math produced by `reshape`.

## 4. Week 2 — buffers, loads, stores, memory ordering

Memory is **versioned**. A buffer value names one *version* (a snapshot) of a buffer.
`store` does not mutate anything visible: it returns a *new version*.

| op | signature | semantics |
|---|---|---|
| `buffer DT N` | → buf\<DT,N\> | fresh buffer of N elements (N a constant > 0), zero-initialized |
| `buffer DT [v0 v1 …]` | → buf | fresh buffer initialized from the list (N = length) |
| `store B i v` | buf, i32, DT → buf | the version of `B` with element `i` replaced by `v` |
| `load B i` | buf, i32 → DT | element `i` of version `B` |

* **Bounds checking.** For every `load`/`store`, the compiler must *prove* `0 ≤ i < N` with the
  week-1 machinery (intervals). If it can't, that is a `CompileError`. No runtime checks are allowed.
* **Ordering.** Versions form chains: `b0 → b1 → b2`. In a physical implementation all versions share one
  allocation that gets mutated in place. Your schedule must therefore run every `load` of version `v`
  *before* the store that overwrites `v` (write-after-read). If that's impossible (for example, two stores
  both build on `v`, or a load of `v` depends on a newer version), the compiler must **copy** the buffer
  (or prove the conflict away). The answer must always match the versioned semantics.

## 5. Week 3 — loops, rendering, runtime

| op | signature | semantics |
|---|---|---|
| `range N` | → i32 | a loop index over `0 … N-1` (N a constant > 0). Each `range` statement is a **distinct** loop, even if another `range N` looks identical. Intervals: `[0, N-1]` |
| `end S r1 … rk` | buf, ranges → buf | runs the computation of store-chain `S` for every `(r1,…,rk)` in lexicographic order (`r1` outermost). Returns the version after the last iteration |

**Loop semantics.** The *body* of `end S r…` is found **structurally**:
if `S` is itself an `end` (a directly nested loop), unwrap it. Then follow the chain of `store`s down
to the first version that is not a `store` (a `buffer`, or the result of an earlier `end`).
That version is the chain's **root**. Every store on the chain runs once per iteration,
*even if its index and value don't mention the loop variable*. Inside the loop, the root denotes
the buffer's **current** contents. State is carried from one iteration to the next. That is
what lets `store C i (add (load C i) x)` accumulate, and `p[i] = p[i-1] + a[i]` work.

> Why structural? If the body were defined by *data dependence* ("stores that use `i`"),
> then a correct rewrite like `mul i 0 → 0` could move a store out of its loop and change
> what the program means. Value rewrites must never change memory semantics.

* A chain may contain only `store`s and directly nested `end`s. To run a store *before* a loop, close it
  first: put it in its own `end`, or use a separate buffer. (Tests never read any *other* outside
  version of the same buffer inside a loop.)

* Every store whose index or value depends on a range must reach an `end` that closes that range before its
  version is used outside the loop. Using an unclosed version outside its loop is a `CompileError`.
* An `assert` may depend on open ranges through *values*. It is checked for **every** value of them.
  But every buffer version it reads must be closed: reading a store whose range has no `end` is a `CompileError`.
* Bounds proofs apply to code that runs. You may delete dead code before checking bounds.
* `prove` may depend on ranges, which act like `var 0 N-1`.
* Programs now **really run**: from this week on, the tests expect a C backend (`UC_BACKEND=c`, the default).
  Signed overflow must wrap, so compile with `-fwrapv`.

## 6. Week 4 — tensors: rangeify, reduce, materialization

Values may now have a **shape**. A list literal `[1 2 3]` is an i32 tensor of shape `(3)`. A scalar has shape `()`.

* Every ALU op from week 1 now works **elementwise** on tensors. Operand shapes must be **identical**
  (there is no implicit broadcasting; use `expand`). A shape mismatch is a compile error.
* `where` needs all three shapes to match.

| op | signature | semantics |
|---|---|---|
| `[a b c …]` | | rank-1 tensor. Elements are literals of one dtype |
| `expand x S…` | | broadcast. In week 4: `x` is a scalar and `S` is one size, so `expand 1 3` gives `[1 1 1]` |
| `reduce x op=OP pop=K` | | reduces the **last K axes** with `OP ∈ {add, mul, max, min}`. The result has rank − K |
| `index x i…` | | scalar element at static indices (constants) |
| `contiguous x` | | identity, but forces `x` to be materialized into a buffer (a kernel boundary) |

**Kernels.** One *kernel* is one compiled function launch. Its iteration space is a single loop nest:
outer *output* loops (iterations independent of each other) and, inside them, any number of *reduce* loops.
Materializing a tensor means writing it to a buffer with one kernel. Later kernels then load from that buffer.
`# max-kernels: N` limits the number of launches the whole program may use, including the kernels that
evaluate the asserts. Asserts on tensors are checked elementwise.

f32 reductions may be computed in any order, so tests compare float reductions with `assertclose`,
or use values whose sums are exact.

## 7. Week 5 — movement ops, rank > 1

Shape arguments are list literals (`[2 2]`) or bare constant integers (`2 2`).
Nested list literals build higher-rank tensors: `[[1 2] [3 4]]` has shape `(2 2)`.

| op | semantics |
|---|---|
| `reshape x [S…]` | same elements, row-major order, new shape. `prod(S)` must equal `x`'s size |
| `permute x [p…]` | output axis `d` is input axis `p[d]`. `p` must be a permutation |
| `expand x [S…]` | same rank as `x`. Each axis of size 1 may grow to any size, other axes must match. A scalar may expand to any shape |
| `pad x [[b0 a0] [b1 a1] …]` | add `b` zeros before and `a` zeros after on each axis (`b,a ≥ 0`) |
| `shrink x [[s0 e0] …]` | keep `[s, e)` on each axis (`0 ≤ s ≤ e ≤ dim`) |
| `flip x [axes…]` | reverse the listed axes |
| `index x i…` | as before, now with one index per axis |

Movement ops are **free**. They never copy data on their own. Each one only changes how indices map to memory.

## 8. Week 6+ — the Python front end

From week 6 on, tests use the Python `Tensor` API (`docs/tensor-api.md`), which builds the same UOp graphs.
UIR stays the debugging format: `Tensor.uir()` prints the graph of any tensor as UIR.

## 9. Errors and the run contract

Your compiler is the package `uc` (in `compiler/uc`). The test harness calls:

```python
from uc.uir import run_uir
result = run_uir(source: str, backend: str | None = None)   # backend: "interp" | "c" | "cuda"
result.kernels   # int: number of kernel launches (week 4+; may be 0 before that)
```

* Raise `uc.errors.CompileError` for anything wrong at compile time (parsing, types, bounds, failed `prove`).
* Raise `uc.errors.AssertFailure` when an `assert` doesn't hold at run time.
* Any other exception counts as a crash: the test fails.
