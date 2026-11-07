# Week 4 quiz

**1.** Why does UIR forbid implicit broadcasting, even though every array library allows it?

<details><summary>answer</summary>To keep the IR small and every broadcast visible. The front end inserts explicit `expand`s, so rangeify has one case to handle and the compiler can see (and cost) every broadcast.</details>

**2.** In rangeify, what are the indices of `x` inside `reduce x pop=1` when the output index is `()`, and where does the new index come from?

<details><summary>answer</summary>`x[(r,)]`, where `r` is a fresh RANGE over x's last axis, created by the reduce and closed by it.</details>

**3.** What is the identity of an f32 `max` reduction, and why can't it be 0?

<details><summary>answer</summary>`-inf`. With 0, the max of an all-negative tensor would wrongly be 0.</details>

**4.** Fusing `x - expand(max(x))` into one kernel costs how much work for n elements? Materializing?

<details><summary>answer</summary>O(n²) fused (the max is recomputed for each of n outputs), O(n) materialized (plus one launch and a tiny buffer).</details>

**5.** Why is a kernel with two consecutive top-level loops not a valid kernel in our definition?

<details><summary>answer</summary>A kernel is one loop nest whose outer loops are independent. That's the shape that maps to a GPU grid. The second loop depends on the first finishing, which needs a global barrier, i.e. a kernel boundary.</details>

**6.** Why keep REDUCE as one UOp until rendering, instead of lowering it to a loop and an accumulator immediately?

<details><summary>answer</summary>Later rewrites (splitting, parallelizing, tensor cores, folding a reduce of a constant) need to see the whole reduction as one node with its ranges. Once it's lowered to loads and stores, that structure is gone.</details>

**7.** Rangeify takes 30 seconds on a 200-node graph. What's the likely bug?

<details><summary>answer</summary>No memoization on (UOp, index tuple), so shared subexpressions are re-lowered once per path. That's exponential in DAG depth.</details>

**8.** What does `reduce [1 2 3] op=add pop=0` mean?

<details><summary>answer</summary>A reduction over no axes: the identity transformation, `[1 2 3]`.</details>
