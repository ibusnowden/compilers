# Week 1 assignment — UOps & rewrites

**Tests due:** end of lecture 1.3 · **Compiler due:** two days later · **Grade:** `python tools/grade.py 1`

## Deliverables

1. `compiler/uc/uop.py`: `Ops`, `UOp` (hash-consed, immutable, `.toposort()`), `UPat`, `PatternMatcher`,
   `graph_rewrite`. Interface: the docstring in the starter file and `tests/week01/test_uop.py`.
2. `compiler/uc/symbolic.py`: one `exec_alu`, bounds (`vmin`/`vmax`), and a `PatternMatcher` with **at least** the
   [required rules](../uir-spec.md#the-required-simplification-rules-for-prove). Every rule must be sound.
3. `compiler/uc/parse.py`: UIR → UOps, with type checking.
4. `compiler/uc/uir.py`: `run_uir(src)`. Parse, simplify, and check every `assert` / `assertclose` / `prove`.
   Raise `CompileError` or `AssertFailure` as the spec says.
5. `compiler/uc/lam.py`: the λ-calculus on your UOps (`tests/week01/test_lambda.py`).
6. **Your tests:** at least **15** new `.uir` files in `tests/week01/mine/`. Each must pass the oracle
   (`UC_ORACLE=1 pytest tests/week01/mine`). Include at least 3 `compile-error` tests, 2 `assert-fail` tests,
   and 4 `prove` tests, one of which is a *true* fact that a sound simplifier must **refuse** to prove because of overflow.

## Checklist

- [ ] `UC_ORACLE=1 pytest tests/week01` passes (sanity: your environment works)
- [ ] `pytest tests/week01/test_uop.py` passes before you start the simplifier
- [ ] `pytest tests/week01` passes, including `cse_big_dag.uir` in well under a second
- [ ] `python solutions/check_provable.py tests/week01/mine/*.uir` (optional): checks that your `prove` tests follow from the required rules
- [ ] Every rule in `symbolic.py` has a one-line comment explaining why it is sound

## Constraints

* No recursion that can exceed Python's limit on a 20,000-node chain (`test_toposort_deep`).
* Constant folding uses the same `exec_alu` as everything else.
* Target size: `uop.py` + `symbolic.py` + `parse.py` + `uir.py` ≈ 400–600 lines. If you're at 1500, something is over-engineered.

## Stretch

* Make the λ-calculus normalizer memoize normal forms per node (hash-consing makes this a dict).
* Add `UOp.render()` that prints any graph back as UIR. You'll want it for debugging every week from now on.
* Fuzz your simplifier: generate random i32 expressions over 2–3 small-range vars, simplify, and check against
  brute-force enumeration. (`hypothesis` is installed.) This finds unsound rules fast.
