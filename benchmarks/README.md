# Benchmarks (weeks 9–11)

These are scripts, not tests. Run them on the H100 and put the numbers in your weekly report.

| script | week | what it measures |
|---|---|---|
| `roofline.py` | 7–9 | achieved FLOP/s and GB/s for a sweep of kernels, plotted against the machine's roofline |
| `generate.py` | 11 | Qwen2.5-1.5B decode tokens/s at batch 1, as a fraction of the HBM-bandwidth bound |
| `train_gpt2.py` | 11 | GPT-2 124M on FineWeb with **your** compiler |
| `train_gpt2_torch.py` | 11 | the same run in PyTorch: the curve you must match |

Pass bars for week 11 (one H100):

* **Inference:** batch-1 decode ≥ 50% of `bandwidth / bytes_of_weights` tokens/s.
* **Training:** at equal tokens, your validation loss is within 1% of `train_gpt2_torch.py`'s, and your
  model-FLOPs utilization (MFU) is ≥ 50% of PyTorch's (with `torch.compile` off).
