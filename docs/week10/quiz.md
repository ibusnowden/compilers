# Week 10 quiz

**1.** In ring all-reduce with N devices and data size D, how much does each device send? What happens as N grows?

<details><summary>answer</summary>2(N−1)/N · D, which approaches 2D and is independent of N. That's why it's bandwidth-optimal.</details>

**2.** `x` is sharded on axis 0 across 4 devices. What communication does `x.sum(axis=0)` need? `x.sum(axis=1)`?

<details><summary>answer</summary>axis=0 (the sharded axis): a partial sum per device plus an all-reduce. axis=1: none; the result stays sharded on axis 0.</details>

**3.** Megatron's MLP splits W₁ by columns and W₂ by rows. Why that pairing, and how many all-reduces does a forward pass need?

<details><summary>answer</summary>Column-split W₁ gives each device a slice of the hidden activations. The elementwise activation applies per slice with no communication, and row-split W₂ then produces partial outputs that need one all-reduce.</details>

**4.** Why does data-parallel training match single-device training exactly (up to float rounding)?

<details><summary>answer</summary>The gradient of a mean loss over the global batch is the mean of the per-shard gradients. An all-reduce of the per-shard sums (divided by the global batch size) recovers it, and every device then takes the identical step.</details>

**5.** What does ZeRO stage 3 shard, and what communication does it add?

<details><summary>answer</summary>Parameters, gradients, and optimizer state. It all-gathers parameters before each layer's forward and backward, and reduce-scatters gradients after.</details>

**6.** Why are MIG slices a poor way to test all-reduce performance?

<details><summary>answer</summary>MIG instances are isolated: no P2P and no NCCL between them. Communication has to go through the host, which doesn't resemble NVLink.</details>

**7.** Why does autodiff need no special rules for sharded tensors in uc?

<details><summary>answer</summary>Communication is expressed as ordinary graph ops (copies and adds) with ordinary gradients. The VJP of a sum all-reduce is an all-reduce, and of a copy, a copy back.</details>
