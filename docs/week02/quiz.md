# Week 2 quiz

**1.** `%b1 = store %b0 0 5` then `%x = load %b0 0`. What is `%x`, and what ordering constraint does this create?

<details><summary>answer</summary>`0` (b0 is zero-initialized). The load of b0 must run before the store that produces b1, because the store overwrites b0's storage in place (WAR).</details>

**2.** Why must two `buffer i32 4` statements not be hash-consed into one node, when two `add 1 2` should be?

<details><summary>answer</summary>A buffer has identity: it's a distinct piece of storage that can be written. `add 1 2` is a pure value. Give BUFFER a unique id in its arg.</details>

**3.** Name the three dependence kinds, and say which one our versioned IR does *not* represent as an edge.

<details><summary>answer</summary>RAW (true), WAW (output), WAR (anti). WAR is implicit: the overwriting store doesn't point at the loads of the old version.</details>

**4.** Give a two-statement pattern that has no valid in-place schedule. How do you fix it?

<details><summary>answer</summary>A fork: `x = store b0 0 1` and `y = store b0 1 2`, with both x and y read later. Copy b0 for one branch.</details>

**5.** `load %b (load %idx 0)` is rejected even though `%idx` contains only valid indices. Why, and what's the idiomatic fix?

<details><summary>answer</summary>A loaded value's bounds are the full i32 range, so the proof fails. Clamp it: `max 0 (min v (N-1))`.</details>

**6.** When is `load (store B i v) j → load B j` sound?

<details><summary>answer</summary>When `i ≠ j` holds for all values. That means the simplifier must turn `cmpne i j` into `true`. "Probably different" isn't enough.</details>

**7.** Your interpreter stores a separate Python list per *version*. All tests pass. What's wrong with this design for the course?

<details><summary>answer</summary>It hides WAR bugs. The real backends (C, CUDA) mutate one allocation, so an incorrect schedule that works in the interpreter will fail in week 3. The interpreter should alias versions to one storage, like the hardware.</details>

**8.** Why is "linearize = toposort" not enough on its own?

<details><summary>answer</summary>Toposort respects only src edges. Memory needs the extra WAR edges (load of v before the store that overwrites v). Without them, any toposort that happens to put the overwriting store first is "valid" but wrong.</details>

**9.** Is dead-store elimination always safe in week 2? What changes in week 6?

<details><summary>answer</summary>In week 2 a store whose version is never read can be removed. In week 6, tensors the user holds are outputs even if nothing in the graph reads them, so "never read in the graph" is not the same as "dead".</details>

**10.** What does differential testing compare, and why is it valuable from week 3 on?

<details><summary>answer</summary>The same program on two implementations (interpreter vs. C, later C vs. CUDA, unoptimized vs. optimized). Any difference is a bug in one of them, without anyone having to write the expected answer.</details>
