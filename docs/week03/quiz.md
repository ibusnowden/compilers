# Week 3 quiz

**1.** Why does every `RANGE` need a unique id, when two `add 1 2`s should be the same node?

<details><summary>answer</summary>A range is a loop, which has identity. `range 4` for rows and `range 4` for columns are different loops. If they were hash-consed into one, a 2-D loop would collapse into a 1-D diagonal.</details>

**2.** `%s = store %o 0 (add (load %o 0) 1)`, then `%o1 = end %s %i` with `%i = range 5`. What is `load %o1 0`? Why isn't the store hoisted, even though it doesn't mention `%i`?

<details><summary>answer</summary>5. The store is on the chain closed by `end`, so it runs every iteration (structural semantics). Also, its value reads the loop-carried root, which changes each iteration.</details>

**3.** Where does `(load %k 0) * 3` go in the generated C, and what classic optimization is that?

<details><summary>answer</summary>Before the loop, because it depends on no range: loop-invariant code motion.</details>

**4.** Why compile with `-fwrapv`? Give a UIR test that fails without it at `-O2`.

<details><summary>answer</summary>Signed overflow is UB in C, so clang may assume `x + 1 > x`, fold overflow checks, or vectorize assuming no wrap. `wrap_in_c.uir` (adding 2³⁰ four times) can be "optimized" incorrectly.</details>

**5.** Why might `fmaxf(a, b)` be the wrong rendering of UIR's `max` for f32?

<details><summary>answer</summary>`fmaxf` returns the non-nan argument when one input is nan. UIR's `max` propagates nan.</details>

**6.** What does `restrict` promise, and why is the promise true for uc's kernels?

<details><summary>answer</summary>That no other pointer argument aliases this one. Distinct `BUFFER`s are distinct allocations, and versions that would conflict are copied (week 2), so kernel arguments never overlap.</details>

**7.** In the matmul loop nest, which loops can run in parallel and which can't? Why?

<details><summary>answer</summary>`i` and `j` are independent (each iteration writes a different C element). `k` accumulates into the same C element, so each iteration depends on the previous one (a reduction).</details>

**8.** Why cache compiled `.so` files by a hash of the C source rather than by the UIR text?

<details><summary>answer</summary>Different UIR programs often produce identical kernels (after simplification and renaming), and the source is exactly what clang sees. The source hash is the precise key. (In week 6 the key becomes the kernel's hash-consed AST.)</details>

**9.** Your C backend gives `0.30000001` where the interpreter gives `0.3` for a float expression. Name two flags or rendering choices that could cause it.

<details><summary>answer</summary>FMA contraction (`-ffp-contract` defaulting to fused on some targets), `-ffast-math` reassociation, or rendering float literals with too few digits (print with `%.9g` or as hex floats).</details>
