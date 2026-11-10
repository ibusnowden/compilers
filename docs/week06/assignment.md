# Week 6 assignment — autodiff & speed

**Done when:** `pytest tests/week{01..06}` passes. The new tests are Python (`tests/week06/*.py`).

## Deliverables

1. `compiler/uc/tensor.py`: the [Tensor API](../tensor-api.md) on top of your UOps (a new file).
2. `compiler/uc/autodiff.py`: reverse mode as a graph transformation, with every rule in lecture 6.2.
3. `compiler/uc/nn/`, `compiler/uc/optim.py`: modules, SGD, AdamW, and safetensors I/O (with bf16).
4. `compiler/uc/stats.py` counters wired into your runtime (kernels, compiles, flops, bytes).
5. Models: `uc/models/{mnist,gpt2,resnet}.py`. Debug them with `UC_TORCH_SHIM=1` first.
6. **Your tests:** at least **15** new tests in `tests/week06/mine/test_*.py` that pass under `UC_TORCH_SHIM=1`:
   gradients of composite ops (layernorm, softmax + cross-entropy, conv with stride), the `assign` corner cases,
   and fusion counts for two patterns of your choice.

## Checklist

- [ ] `UC_TORCH_SHIM=1 pytest tests/week06` passes with your models (proves the model code is right)
- [ ] `pytest tests/week06 -m "not slow"` passes on uc
- [ ] `python tools/download.py mnist gpt2 resnet18 && pytest tests/week06 -m slow` passes
- [ ] MNIST CNN training (1000 steps) finishes in under 2 minutes on CPU. Note your time in `REPORT.md`

## Stretch

* Fold BatchNorm into the preceding conv when loading ResNet.
* Generate text with GPT-2 (`tiktoken` for the tokenizer) and report tokens/s on CPU.
