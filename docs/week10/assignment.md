# Week 10 assignment — multiple GPUs and sharding

**Grade:** `python tools/grade.py 10`, and `UC_VIRTUAL_GPUS=4 UC_MULTI=CUDA pytest tests/week10`.

## Deliverables

1. Multi-device tensors (`multi.py`) with the propagation rules in lecture 10.2. `Tensor.shard`, and `CPU:N` devices.
2. Device-to-device copies (P2P where available, host-staged otherwise), and virtual GPUs (`UC_VIRTUAL_GPUS`).
3. Ring all-reduce built from your copies and add kernels, compared with a naive gather-to-one.
4. Data-parallel and tensor-parallel training that match single-device results.
5. **Your tests:** at least **8** in `tests/week10/mine/`: resharding paths, sharded reductions on every axis, gradients through a sharded matmul, and an all-reduce with N = 2, 3, 4.

## Report

All-reduce bandwidth vs message size (1 KB to 1 GB) for ring vs naive, on virtual GPUs. On a real multi-GPU node, if you rent one.

## Stretch

* Overlap gradient all-reduce with backward compute (a second stream).
* ZeRO-1: shard AdamW's moments across data-parallel ranks.
