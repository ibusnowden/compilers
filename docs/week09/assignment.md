# Week 9 assignment — tensor cores, GEMM, conv, flash attention

**Done when:** `pytest tests/week{01..09}` and `UC_DEVICE=CUDA pytest tests/week06` pass, and `pytest tests/week09 -m perf -s` meets its targets (output in `REPORT.md`).

## Deliverables

1. `f16`, `bf16` end to end (casts, loads, stores, safetensors), with f32 accumulation.
2. A `WMMA` UOp and the `tc` optimization action (mma.sync + ldmatrix). Shared-memory staging with swizzle, and ≥ 2-stage `cp.async` pipelining.
3. Implicit-GEMM convolutions that use tensor cores.
4. Flash attention (forward), causal and GQA, one kernel. Write down which design (fusion rule or general reduce) you chose and why.
5. **Your tests:** at least **10** in `tests/week09/mine/`: a single-MMA test against numpy, edge tiles (sizes not divisible by 16), f16 overflow behavior, and attention with T=1 and T not divisible by the block size.

## Perf targets (H100, vs PyTorch)

bf16 GEMM 8192 ≥ 50% · conv ≥ 30% · flash attention ≥ 40% · T = 65536 runs.

## Stretch (the best one in the course)

* `wgmma` + TMA + warp specialization on Hopper. Target ≥ 75% of cuBLAS.
