# Week 7 quiz

**1.** What's the arithmetic intensity of `c = a + b` in f32, and what does it imply?

<details><summary>answer</summary>1 FLOP per 12 bytes ≈ 0.083 FLOP/B. It's memory-bound: only fusion (fewer bytes) helps, not more FLOPs.</details>

**2.** Why does a dot product with one accumulator run at about ¼ of peak, even with data in L1?

<details><summary>answer</summary>Each FMA depends on the previous one through the accumulator. With ~4 cycles of FMA latency, you get one FMA per 4 cycles per lane instead of one or two per cycle. Use several independent accumulators.</details>

**3.** Express "tile i by 4" as a range rewrite. What does the simplifier need to do afterwards?

<details><summary>answer</summary>`i → i_o·4 + i_i` with `i_o ∈ [0, N/4)` and `i_i ∈ [0, 4)`. Indices like `(i_o·4 + i_i)·K + k` get re-canonicalized, and bounds stay provable.</details>

**4.** What must be proven before a load is turned into a vector load of 8 floats?

<details><summary>answer</summary>That the 8 lanes read consecutive addresses `base + lane`. For aligned loads, also that `base` is a multiple of 8 (or use unaligned loads).</details>

**5.** How do you parallelize `sum(x)` over 8 threads when the output is a scalar?

<details><summary>answer</summary>Split the reduce range: each thread reduces a chunk into partial[tid], then a second small kernel (or one thread) reduces the partials.</details>

**6.** What is false sharing?

<details><summary>answer</summary>Two threads write different variables on the same cache line, so the line bounces between cores even though there's no logical sharing.</details>

**7.** Why must optimizations be UOp rewrites rather than C-source transformations?

<details><summary>answer</summary>So they can be checked (fuzzed against UC_OPT=0), searched, cached by AST, and reused unchanged for the CUDA backend.</details>

**8.** Why can a softmax implementation beat numpy's by 2× or more without any SIMD tricks?

<details><summary>answer</summary>It's memory-bound. numpy makes separate passes for max, subtract/exp, sum, and divide, each reading and writing DRAM. A fused implementation reads x about twice and writes once.</details>
