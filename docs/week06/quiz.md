# Week 6 quiz

**1.** Why reverse mode and not forward mode for training?

<details><summary>answer</summary>One scalar output and millions of inputs. Reverse mode gets every ∂L/∂param in one backward pass. Forward mode needs one pass per input.</details>

**2.** What is the gradient of `expand x [4 3]` (x has shape `(4, 1)`), as a movement or reduce op?

<details><summary>answer</summary>A sum over the expanded axis: `reduce add` over axis 1 (keepdim to (4, 1)).</details>

**3.** `y = x * x` is built as `MUL(x, x)` with both sources the same hash-consed node. What does a correct backward produce for `x`?

<details><summary>answer</summary>`g·x + g·x = 2·g·x`. Both source slots contribute, and the contributions must be summed, not overwritten.</details>

**4.** Why does torch split the gradient of `max` evenly among tied elements, and why must you match it?

<details><summary>answer</summary>max isn't differentiable at ties, so any subgradient is valid. torch chooses the even split. The tests compare with torch, and a different choice changes training trajectories.</details>

**5.** `b = a + 1` (lazy), then `a.assign(zeros)`, then `b.numpy()`. What must you get, and which week's machinery delivers it?

<details><summary>answer</summary>old a + 1. `assign` is a store that creates a new version (week 2). b references the old version, so the WAR ordering realizes b first (or copies a).</details>

**6.** What does the compile cache key on, and why must buffers be kernel parameters rather than constants in the source?

<details><summary>answer</summary>The kernel's AST (a hash-consed UOp). If buffer addresses were baked in, every new tensor would produce a new kernel source and miss the cache.</details>

**7.** How do you convert bf16 bits to f32 exactly?

<details><summary>answer</summary>Zero-extend the 16 bits to 32 and shift left by 16, then bitcast to f32. bf16 is the top half of an f32.</details>

**8.** HF GPT-2 stores `c_attn.weight` with shape `(768, 2304)`. What do you do with it for a Linear that computes `x @ W.T`?

<details><summary>answer</summary>Transpose it to `(2304, 768)`, because HF's Conv1D computes `x @ W`.</details>

**9.** Your MNIST CNN trains with the shim but gets 11% accuracy on uc. Name two likely culprits.

<details><summary>answer</summary>A wrong gradient for some op (conv, max_pool, or cross_entropy's log_softmax); grads that aren't zeroed or accumulate across steps; `assign` in the optimizer not updating the buffer the next step reads (versioning); dropout active at test time.</details>
