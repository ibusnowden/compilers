# Week 11 quiz

**1.** Qwen2.5-1.5B in bf16 is ~3.1 GB. What's the batch-1 decode bound on an H100 SXM? On the PCIe card?

<details><summary>answer</summary>~3350/3.1 ≈ 1000 tok/s. ~2000/3.1 ≈ 650 tok/s on PCIe.</details>

**2.** Why does a KV cache grown with `cat` defeat the JIT?

<details><summary>answer</summary>The cache shape changes every step, so the kernels (which are specialized to shapes) change. Every step re-schedules and recompiles.</details>

**3.** How does `start_pos` reach the kernels without specializing them?

<details><summary>answer</summary>As a symbolic VAR with bounds [0, max_seq_len), bound to a runtime argument at launch. Index expressions use it, and bounds checks still work through its interval.</details>

**4.** What does GQA with H = 12 query heads and H_kv = 2 do to the KV cache size?

<details><summary>answer</summary>It shrinks it 6× compared with full multi-head attention.</details>

**5.** Why keep master weights in f32 when computing in bf16?

<details><summary>answer</summary>Updates like lr·g are often smaller than bf16's resolution at the weight's magnitude, so they round to zero. f32 masters accumulate the small updates.</details>

**6.** For B·T = 16k tokens and a 50k vocabulary, how big are f32 logits, and what's the fix?

<details><summary>answer</summary>16384 × 50304 × 4 B ≈ 3.3 GB. Fuse the cross-entropy (chunked or online log-softmax over the vocabulary) so the full logits are never materialized in f32.</details>

**7.** Your 20-step tiny-GPT-2 test matches torch for 3 steps, then drifts. Name the first two things to check.

<details><summary>answer</summary>The optimizer: AdamW bias correction, step count, weight-decay application, and which parameters get decay. Then `assign`/versioning (does the step read updated parameters correctly?). Steps 1–3 matching means the forward and gradients are likely fine.</details>

**8.** Write the MFU formula used in the benchmarks.

<details><summary>answer</summary>tokens/s × (6N + 12·L·T·d) / peak FLOP/s, where N excludes position embeddings, L = layers, T = sequence length, d = model width.</details>
