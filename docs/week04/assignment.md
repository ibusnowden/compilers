# Week 4 assignment — rangeify, reduce, materialization

**Done when:** `pytest tests/week{01..04}` passes. From this week on, `# max-kernels` is enforced.

## Deliverables

1. Shapes on tensor UOps. List literals, elementwise ALU on tensors, `expand` (scalar → n), `reduce`, `index`, `contiguous`.
2. `compiler/uc/rangeify.py`: tensor graph → kernels (loops, loads, stores, `REDUCE`).
3. A materialization policy and a scheduler. `run_uir` returns `Result(kernels=...)`.
4. `REDUCE` rendered as a register accumulator.
5. **Your tests:** at least **12** new `.uir` files in `tests/week04/mine/`. Include a test where materialization
   matters for the kernel count, reductions with each of the four ops, and a reduce over a value that is itself a
   reduce result (a reduce of a reduce).

## Checklist

- [ ] All week 1–3 tests still pass. The low-level path (explicit loops) and the tensor path share one linearizer and renderer
- [ ] `DEBUG=2` shows one C function per kernel
- [ ] Your policy is ≤ 30 lines and written down in a comment

## Stretch

* A tiny cost model: estimate bytes moved for "fuse" vs "materialize", and pick the cheaper option instead of using a fixed rule.
