# Week 8 quiz

**1.** Why can't blocks of one kernel synchronize with each other, and what does that imply for kernel boundaries?

<details><summary>answer</summary>Blocks may not all be resident at once, and their scheduling order is unspecified, so a grid-wide barrier could deadlock. Anything that needs all blocks done (e.g. using a full reduction's result) needs a new kernel.</details>

**2.** What is occupancy, and what limits it?

<details><summary>answer</summary>Resident warps per SM relative to the maximum. It's limited by registers per thread, shared memory per block, threads per block, and the per-SM block limit.</details>

**3.** A warp reads `a[32·t]` for t = 0…31. How many 32-byte sectors does it fetch, compared with reading `a[t]`?

<details><summary>answer</summary>32 sectors (one per thread) instead of 4 (128 contiguous bytes): 8× the traffic.</details>

**4.** How many shuffle steps does a warp-level sum take, and which intrinsic?

<details><summary>answer</summary>5 (log₂ 32), with `__shfl_xor_sync(0xffffffff, v, offset)` for offsets 16, 8, 4, 2, 1.</details>

**5.** Why is `test_big_grid` (70000 rows) a test?

<details><summary>answer</summary>gridDim.y and .z max out at 65535. Mapping 70000 to y fails to launch. Use .x (up to 2³¹−1) or loop.</details>

**6.** In your reduction, a block handles the tail of an array of length n = 1000 with 256 threads. What value do out-of-range threads contribute?

<details><summary>answer</summary>The reduction's identity: 0 for sum, -inf for max. Those threads must not skip the shuffle steps, or the warp's lanes disagree.</details>

**7.** Why does softmax need a new scheduling rule on GPUs?

<details><summary>answer</summary>Week 4's policy materializes reduce results that get expanded, which means 3 kernels and 3 HBM round trips. When a row fits in one block, the reduce can be done cooperatively inside the block, keeping the row on-chip.</details>

**8.** Name two warp stall reasons in Nsight Compute and the usual fix for each.

<details><summary>answer</summary>Long scoreboard (waiting on global memory): more ILP, prefetching, or occupancy. MIO throttle / short scoreboard (shared memory pressure): larger register tiles, fewer shared loads, vector shared loads. Barrier: less imbalance between syncs.</details>
