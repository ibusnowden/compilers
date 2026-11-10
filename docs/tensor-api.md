# The `uc.Tensor` API (weeks 6–11)

From week 6 on, tests drive your compiler through a small, PyTorch-flavoured, **lazy** tensor
library. Every method builds UOps. Nothing runs until `.realize()`, `.numpy()`, `.item()`,
or `.tolist()`. This page is the contract the tests rely on. Anything not listed here is up to you.

> **Check your tests against PyTorch.** `UC_TORCH_SHIM=1 pytest tests/week06` replaces `uc` with
> `reference/torch_shim/`, a ~350-line PyTorch implementation of this exact API.
> If a test fails under the shim, the *test* is wrong.

## Creating tensors

| call | notes |
|---|---|
| `Tensor(data, dtype=None, device=None, requires_grad=False)` | `data`: Python scalar, nested list, or `np.ndarray`. Floats default to `f32`, ints to `i32` (not i64), bools to `bool` |
| `Tensor.zeros(*shape, dtype=f32, device=None)`, `.ones`, `.full(shape, value)` | |
| `Tensor.arange(stop)` / `Tensor.arange(start, stop, step=1)` | i32 when every argument is an int |
| `Tensor.eye(n)` | |
| `Tensor.manual_seed(seed)`; `Tensor.rand(*shape)`, `.randn(*shape)`, `.uniform(*shape, low, high)` | values need only be deterministic for a given seed. They don't have to match torch |
| `Tensor.empty(*shape, dtype)` | contents undefined |

## Properties and conversion

`t.shape` (tuple), `t.dtype` (`uc.dtypes.*`), `t.device` (str), `t.ndim`, `t.numel()`,
`t.numpy()`, `t.item()`, `t.tolist()`, `t.realize()` (returns `t`), `t.to(device)`, `t.cast(dtype)`,
`t.contiguous()`, `t.detach()`, `t.assign(other)` (overwrite in place, used by optimizers and weight loading. Tensors built *before* the assign still see the old value: week 2's versioned memory), `t.uir() -> str` (the UOp graph as UIR text, for debugging).

## Elementwise (numpy broadcasting rules; Python scalars are allowed on either side)

`+ - * /` (true division; int / int is an error, so cast first), unary `-`, `t ** k` (Python number `k`),
`exp log exp2 log2 sqrt rsqrt sin cos abs neg reciprocal`,
`relu sigmoid tanh silu gelu` (gelu uses the **tanh approximation**, as in GPT-2),
`maximum(a, b) minimum(a, b)`, `t.where(a, b)` (`t` is the bool condition),
`< <= > >= == !=` (return bool tensors), `t.clip(lo, hi)`.

## Reductions (`axis`: int, tuple of ints, or None = all; negative axes allowed)

`sum max min mean (axis=None, keepdim=False)`, `var(axis=None, keepdim=False)` (population, i.e. divides by N),
`argmax(axis=-1)` → i32, `softmax(axis=-1)`, `log_softmax(axis=-1)`, `logsumexp(axis=-1)`.

## Movement (no data is copied until something forces it)

`reshape(*shape)` (one `-1` allowed), `permute(*order)`, `expand(*shape)` (`-1` means keep),
`pad(((before, after), ...))` (zero fill), `shrink(((start, end), ...))`, `flip(*axes)`,
`transpose(a=-2, b=-1)`, `flatten(start_dim=0)`, `unsqueeze(dim)`, `squeeze(dim)`,
`t[...]` with ints, `slice`s with step 1, and `None`. `Tensor.cat(*ts, dim=0)`, `Tensor.stack(ts, dim=0)`.

## Linear algebra and NN functions

| call | semantics |
|---|---|
| `a @ b`, `a.matmul(b)` | numpy matmul semantics (batched, broadcast) |
| `x.linear(w, b=None)` | `x @ w.T + b` (weights stored `(out, in)` like torch) |
| `x.conv2d(w, b=None, stride=1, padding=0)` | NCHW, `w` is `(Cout, Cin, kh, kw)`, cross-correlation like torch |
| `x.max_pool2d(kernel=2, stride=None, padding=0)`, `x.avg_pool2d(kernel, stride=None)` | `stride` defaults to `kernel`. Max-pool padding counts as `-inf` |
| `x.layernorm(axis=-1, eps=1e-5)`, `x.rmsnorm(axis=-1, eps=1e-6)` | no affine part: multiply by weights yourself |
| `logits.cross_entropy(labels)` | `logits (N, C)` f32, `labels (N,)` i32, mean over N |
| `q.scaled_dot_product_attention(k, v, is_causal=False)` | `(B, H, T, D)`; scale `1/sqrt(D)`. `k, v` may have fewer heads (GQA): heads are repeated |
| `x.dropout(p)` | only when `Tensor.training` is True |

## Autograd (week 6)

* `requires_grad=True` on leaves. `loss.backward()` on a scalar fills `.grad` (a `Tensor`) on every leaf that needs it.
  Gradients **accumulate** across calls until you zero them.
* `with Tensor.no_grad(): ...` disables graph recording for gradients.
* `Tensor.training` (class attribute, default `False`) controls dropout.

## `uc.nn`, `uc.optim`, `uc.nn.state`

```python
from uc import nn, optim
from uc.nn.state import safe_load, safe_save, get_parameters, load_state_dict, get_state_dict

nn.Linear(in, out, bias=True)    nn.Embedding(vocab, dim)     nn.Conv2d(cin, cout, k, stride=1, padding=0, bias=True)
nn.LayerNorm(dim, eps=1e-5)      nn.RMSNorm(dim, eps=1e-6)    nn.BatchNorm2d(c, eps=1e-5)   # eval mode: running stats
# each has .weight (and .bias), is callable, and exposes its parameters to get_parameters

optim.SGD(params, lr, momentum=0.0, weight_decay=0.0)
optim.AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)    # torch semantics (decoupled decay)
opt.zero_grad(); opt.step()

safe_load(path) -> dict[str, Tensor]       # safetensors, read with mmap. bf16 / f16 / f32 / i32 / i64 / u8
safe_save(dict[str, Tensor], path)
get_parameters(obj) -> list[Tensor]        # walks attributes, lists, and dicts and finds every requires_grad Tensor
get_state_dict(obj) -> dict[str, Tensor]   # dotted names, e.g. "layers.0.attn.wq.weight"
load_state_dict(obj, sd, strict=True)      # assigns by those dotted names
```

## Models (`uc.models`, written by you)

| module | class | weights | tested in |
|---|---|---|---|
| `uc.models.mnist` | `MLP()`, `CNN()` | trained by you | week 6 |
| `uc.models.gpt2` | `GPT2(config)`, `GPT2.from_hf(state_dict, config)` | HF `openai-community/gpt2` naming (an optional `transformer.` prefix is stripped) | weeks 6, 11 |
| `uc.models.resnet` | `ResNet18(num_classes=1000)` | torchvision / timm naming (`conv1`, `bn1`, `layer1.0.conv1`, …, `fc`) | week 6 |
| `uc.models.llama` | `Llama(config)`, `Llama.from_hf(state_dict, config)` | HF Llama / Qwen2 naming (`model.layers.N.self_attn.q_proj.weight`, …) | week 11 |

The config dicts use the keys in `tests/ref/configs.py`. Every model is callable:
`model(tokens)` gives `(B, T, vocab)` logits. Llama also takes `model(tokens, start_pos)`,
which uses and updates a KV cache.

## Devices (weeks 8, 10)

* `"CPU"`: your C backend (weeks 3–7). `"CPU:1"`, `"CPU:2"`, … are *separate* CPU devices with separate buffers
  (week 10 uses them to test sharding without any GPU). `"CUDA"`: week 8. `"CUDA:1"` names a specific GPU.
* `UC_DEVICE=CUDA` changes the default device.
* **Virtual GPUs:** with `UC_VIRTUAL_GPUS=4`, the names `CUDA:0`…`CUDA:3` all live on the one physical GPU.
  Each has its own allocations and its own stream. This is how you test multi-GPU code on a single H100.
* Week 10: `t.shard(devices, axis)` splits `t` evenly along `axis` across a tuple of devices.
  `t.shard(devices, None)` replicates it. `t.device` is then that tuple. Every op works on sharded tensors.
  If `t` is a leaf that requires grad, the sharded tensor is a new leaf that requires grad too (use it in place of `t`).
  `.numpy()` gathers.

## JIT (week 11)

```python
from uc.jit import jit
@jit
def step(x, y): ...      # the first calls trace and schedule; later calls with the same shapes/dtypes
                         # replay the captured kernels with new input buffers (no scheduling, no compiling)
```

## Instrumentation (used by tests)

```python
from uc import stats
stats.reset()
stats.kernels       # kernel launches since reset
stats.compiles      # kernels compiled (not served from cache) since reset
stats.flops         # estimated FLOPs and
stats.bytes         # bytes moved by those kernels (for roofline numbers)
```

## Environment variables your compiler should read

| var | meaning |
|---|---|
| `UC_DEVICE` | default device |
| `UC_BACKEND` | for UIR files: `interp`, `c`, `cuda` |
| `UC_THREADS` | CPU worker threads (week 7). Default: the number of cores |
| `UC_OPT` | `0` turns off kernel optimizations (tiling, upcasting, …). The result must not change |
| `UC_OPT_FUZZ` | `seed`: apply a *random* sequence of legal optimizations to every kernel (week 7 fuzzing) |
| `UC_BEAM` | `n`: beam-search kernel optimizations with width n (week 7+) |
| `UC_VIRTUAL_GPUS` | see above |
| `DEBUG` | `1`: print each kernel launch. `2`: plus generated source. `3`: plus the UOp graph |
