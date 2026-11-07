# Week 11 assignment — SOTA inference and training

**Grade:** `python tools/grade.py 11`, plus the two benchmark reports. There are no tests due this week, but the final project proposal is.

## Deliverables

1. `uc/models/llama.py`: Llama / Qwen2 with a fixed-size KV cache and symbolic `start_pos`
   (`tests/week11/test_llama.py`).
2. `uc/jit.py`: capture and replay (`test_jit_replays_without_recompiling`). CUDA graphs are optional.
3. Inference: `python benchmarks/generate.py` prints coherent text at **≥ 50% of the bandwidth bound**.
4. Training: `benchmarks/train_gpt2.py` matches `train_gpt2_torch.py`'s validation loss within 1% at equal tokens
   (default budget 1B tokens; use `TOTAL_TOKENS` for a shorter run while iterating), at **≥ 50% of PyTorch-eager MFU**.
5. **Final project proposal** (see `docs/week12/final-project.md`), due at the end of this week.

## Report

The tokens/s table (and bound), both loss curves on one plot (`benchmarks/out/*.csv`), MFU for uc, torch eager, and
`torch.compile`, plus your top-5 kernels by time for one training step.
