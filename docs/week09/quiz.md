# Week 9 quiz

**1.** Why accumulate bf16 matmuls in f32?

<details><summary>answer</summary>bf16 has ~3 significant digits. Adding thousands of products in bf16 loses the small terms (swamping), and the error grows with K.</details>

**2.** In `mma.m16n8k16`, how many multiply-adds does one warp instruction do?

<details><summary>answer</summary>16·8·16 = 2048 MACs (4096 FLOPs).</details>

**3.** A warp reads column 0 of a 32×32 f32 tile in shared memory, stored row-major. How many-way bank conflict? Give one fix.

<details><summary>answer</summary>32-way: every row starts in bank 0, so column 0 is bank 0 for all rows. Pad each row by one element, or XOR-swizzle the column with the row.</details>

**4.** Why is online softmax exact?

<details><summary>answer</summary>softmax is invariant to subtracting any constant. When the running max changes from m to m', multiplying the earlier partial sums by exp(m − m') converts them exactly to the new reference.</details>

**5.** What is the HBM memory complexity of naive attention vs flash attention for sequence length T and head dim d?

<details><summary>answer</summary>Naive: O(T²) for the scores. Flash: O(T·d), since only Q, K, V, O and the per-row statistics touch HBM.</details>

**6.** How does causal masking change the work in flash attention?

<details><summary>answer</summary>Key blocks strictly above the diagonal are skipped (about half the work), and only the diagonal block needs an elementwise mask.</details>

**7.** Express a 3×3, stride-1, pad-1 conv on (N, C, H, W) with K filters as a GEMM: what are M, N, K?

<details><summary>answer</summary>M = N·H·W (output pixels), N_gemm = K (filters), K_gemm = C·3·3.</details>

**8.** What does `cp.async` buy over a normal load-then-store to shared memory?

<details><summary>answer</summary>It copies global → shared without staging in registers, and it's asynchronous, so the warp can issue MMAs on the previous tile while the copy is in flight (pipelining).</details>
