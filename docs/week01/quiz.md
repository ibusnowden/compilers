# Week 1 quiz

Answer before you look. Answers are folded under each question.

**1.** Why is `x is y` a correct (not just fast) structural equality test for hash-consed UOps?

<details><summary>answer</summary>
Construction is the only way to create a UOp, and construction returns the existing live node when one with the same
(op, dtype, src, arg) exists. By induction on depth (sources are hash-consed too, so their identity is their structure),
two structurally equal live nodes are the same object.
</details>

**2.** Your cache is a plain `dict` keyed by `(op, dtype, src, arg)`. Name two problems.

<details><summary>answer</summary>
(1) Nothing is ever freed: the dict keeps every UOp alive. (2) Floats: `0.0` and `-0.0` collide (equal and same hash),
and `nan` never matches itself, so nan constants are never shared. (Keying on `src` by value instead of identity is also slow.)
</details>

**3.** Is `x + 0 → x` sound for f32? What about `x * 1 → x`?

<details><summary>answer</summary>
`x + 0.0 → x` is unsound: `-0.0 + 0.0 = +0.0`. `x * 1.0 → x` is sound in IEEE-754, including for nan, inf, and -0.0.
(`x + (-0.0) → x` is also sound.)
</details>

**4.** Fold `div -7 2`, `mod -7 2`, `mod 7 -2` with C semantics.

<details><summary>answer</summary>`-3`, `-1`, `1`. Truncate toward zero; the remainder takes the dividend's sign.</details>

**5.** Give bounds for `(x * 3) % 4` with `x ∈ [-2, 5]`, using the table from lecture 1.3.

<details><summary>answer</summary>
`x * 3 ∈ [-6, 15]`. The dividend can be negative, so the non-negative mod rule doesn't apply. A sound answer is
`[-3, 3]` (|remainder| < 4, sign follows the dividend). The full i32 range is also sound, just useless.
</details>

**6.** Why must interval arithmetic be computed exactly and *then* widened to the full range, instead of wrapping the endpoints?

<details><summary>answer</summary>
Wrapping endpoints gives an interval that doesn't contain the real values. For example, `[2³¹-2, 2³¹-1] + 1` would become
`[2³¹-1, -2³¹]`, an "empty" or reversed interval. Any rule using it would be unsound. Widening says "anything is possible",
which is always true.
</details>

**7.** `(4x + 3) / 4 → x` is true for `x ∈ [0, 100]`. Give a value of `x` for which it's false, and say which side condition of rule 5 catches it.

<details><summary>answer</summary>
`x = -1`: `(-4 + 3) / 4 = -1 / 4 = 0 ≠ -1`. The condition `vmin(4x + 3) ≥ 0` fails.
</details>

**8.** The rule `x + y → y + x` never terminates. How do you get commutativity without it?

<details><summary>answer</summary>
Orient it: rewrite only when the result is *smaller* in a fixed total order, e.g. put constants on the right and sort operands by a stable key.
Or don't rewrite at all and make the *matcher* try both source orders.
</details>

**9.** In de Bruijn notation, what is `λx. λy. x y`? What is `λx. (λy. y) x` after one β step?

<details><summary>answer</summary>`λ.λ. #1 #0`. Reducing the inner redex gives `λx. x`, which is `λ. #0`.</details>

**10.** Why does `(λx.λy. x) I Ω` terminate under normal order but not applicative order?

<details><summary>answer</summary>
Normal order reduces the outermost redex first: `(λx.λy.x) I → λy. I`, then applying that to `Ω` discards `Ω` without evaluating it.
Applicative order evaluates arguments first, and evaluating `Ω` loops forever.
</details>

**11.** The test `cse_big_dag.uir` builds `a30` from 30 doublings. Your simplifier finishes, but it takes 40 seconds. What's the most likely bug?

<details><summary>answer</summary>
Rewriting is not memoized per node (or the memo is keyed by something other than node identity), so the engine visits
every *path* (2³⁰), not every node. A rule that rebuilds its linear form recursively without caching has the same effect.
</details>

**12.** Why is "one function for ALU semantics" worth enforcing from week 1?

<details><summary>answer</summary>
Constant folding, the interpreter (week 2), the C renderer's expectations (week 3), and test oracles must agree exactly.
Separate copies drift (truncation vs. floor, f32 vs. f64 rounding), and the mismatch shows up much later as "the
optimized program gives a different answer than the unoptimized one."
</details>
