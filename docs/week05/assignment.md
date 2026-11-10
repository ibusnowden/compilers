# Week 5 assignment — movement ops

**Done when:** `pytest tests/week{01..05}` passes.

## Deliverables

1. Nested list literals, rank > 1 everywhere, and the general `expand`.
2. `reshape`, `permute`, `pad`, `shrink`, `flip` as index transformations in rangeify. No copies.
3. Pad validity (gated or clamped loads) that passes the week-2 bounds checker.
4. Simplified index expressions: no `/` or `%` that cancels should survive into the C (check with `DEBUG=2`).
5. **Your tests:** at least **12** new `.uir` files in `tests/week05/mine/`. Include a batched matmul (rank 3),
   an average pool, a 1-D convolution with stride 2 (built from reshape tricks), and a transpose-of-reshape round trip
   with `# max-kernels: 1`.

## Checklist

- [ ] All earlier weeks still pass
- [ ] `matmul.uir`, `maxpool.uir`, `conv2d_shifts.uir`, and `movement_free.uir` are each one kernel
- [ ] Generated C for `index_math_simplifies.uir` has no division or modulo

## Stretch

* Derive and implement a general `pool(x, kernel, stride)` from movement ops only. You'll need it for `conv2d`
  and `max_pool2d` next week.
