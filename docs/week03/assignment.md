# Week 3 assignment — loops, rendering & runtime

**Done when:** `pytest tests/week{01..03}` passes.

## Deliverables

1. `RANGE` and `END` in the parser and the IR, with structural loop bodies and loop-carried roots.
2. The linearizer places every UOp in the innermost loop of the ranges it depends on (with loop-carried loads counted as loop-dependent).
3. `compiler/uc/renderer_c.py` and `compiler/uc/runtime/cpu.py`: render, compile (cached), and run.
   `backend="c"` becomes the default. `backend="interp"` must keep working.
4. Asserts inside loops, and the `range not closed` error.
5. **Your tests:** at least **12** new `.uir` files in `tests/week03/mine/`, passing the oracle. Include a nested-loop
   reduction, a loop-carried dependence other than a sum (e.g. a running max), a convolution with a stride, and a
   test that fails if a loop-carried load is hoisted.

## Checklist

- [ ] `UC_BACKEND=interp pytest tests/week{01..03}` and `UC_BACKEND=c pytest tests/week{01..03}` both pass
- [ ] A differential test: run every `.uir` file under both backends and compare all asserted values
- [ ] `DEBUG=2` prints the generated C

## Stretch

* Render with `#pragma clang loop vectorize(enable)` and check with `-Rpass=loop-vectorize` which loops vectorize.
* Make the cache survive across processes (`~/.cache/uc/`), keyed by source hash *and* compiler flags.
