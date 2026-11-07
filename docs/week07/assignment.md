# Week 7 assignment — memory hierarchies, kernels, threading

**Grade:** `python tools/grade.py 7`, plus `pytest tests/week07 -m perf -s` on a quiet machine (paste the output into `REPORT.md`).

## Deliverables

1. `compiler/uc/opt.py`: split, reorder, unroll, upcast, parallel as UOp rewrites, with legality checks.
2. Vector dtypes, vector loads and stores (with proven contiguity), clang vector-extension rendering.
3. A thread pool (or OpenMP) and two-stage reductions. Honour `UC_THREADS`.
4. Heuristic defaults, plus `UC_OPT=0`, `UC_OPT_FUZZ=seed`, and `UC_BEAM=n` (cached by AST).
5. `stats.flops` and `stats.bytes` filled in per launch. `benchmarks/roofline.py CPU` runs.
6. **Your tests:** at least **10** in `tests/week07/mine/`: optimization fuzzing on shapes of your choice (primes,
   1×N, N×1), and one test per action that checks it is *refused* when illegal.

## Report (`REPORT.md`, 1 page)

Your roofline table, the four perf numbers, and for sgemm: the optimizations your heuristics chose and what beam search found.

## Stretch

* A packed-B layout for sgemm (copy B into tiles once, per the Goto paper) as a separate materialized kernel.
* AVX-512 / NEON-specific rendering behind a capability check.
