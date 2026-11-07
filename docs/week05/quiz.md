# Week 5 quiz

**1.** `x` has shape `(2, 3)`. After `permute [1 0]` then `reshape [6]`, which element of the original buffer does output index `k = 4` read?

<details><summary>answer</summary>The (3,2) view has t[i, j] = x[j, i], and k = 4 is t[2, 0] = x[0, 2], buffer index 0·3 + 2 = 2.</details>

**2.** Why can `reshape(permute(x))` fail to be a single strided view, and why doesn't the index-expression design care?

<details><summary>answer</summary>After a permute, the memory order no longer matches the logical order, so merging axes in a reshape may need a non-constant stride. Index expressions just compose: the reshape adds `/` and `%`, which the simplifier removes when it can.</details>

**3.** Which week-1 rule makes `reshape [12]` followed by `reshape [4 3]` simplify back to direct indexing? What are its side conditions here?

<details><summary>answer</summary>Rule 5: `(i*3 + j)/3 → i + j/3` and `(i*3 + j)%3 → j%3`. It needs `i*3 + j ≥ 0` and `j ≥ 0` without overflow, which the range bounds give. Then `j ∈ [0, 2]` makes `j/3 → 0` and `j%3 → j`.</details>

**4.** The padded load index `i - 1` has bounds `[-1, 6]`. How do you get it past the bounds checker?

<details><summary>answer</summary>Gate the load with `1 ≤ i < 7` and check the bounds of the index under the gate, or clamp the index and select 0 when invalid.</details>

**5.** Write a 2×2 max-pool of a 4×4 tensor using movement ops and one reduce.

<details><summary>answer</summary>`reduce (permute (reshape x [2 2 2 2]) [0 2 1 3]) op=max pop=2`</details>

**6.** In the movement-op matmul, how much memory does `expand (reshape A [M 1 K]) [M N K]` allocate?

<details><summary>answer</summary>None. It's an index transformation: the load from A ignores the j index.</details>

**7.** Why is index math "free" for an elementwise add and not necessarily free inside a GEMM inner loop?

<details><summary>answer</summary>The add is memory-bound: a few integer ops per element hide under DRAM latency. The GEMM is compute-bound, so integer divisions and compares compete with the FMAs for issue slots.</details>

**8.** What does `flip x [0]` do to the index of axis 0 of size n?

<details><summary>answer</summary>`i → n - 1 - i`</details>
