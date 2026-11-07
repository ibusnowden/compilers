"""Week 10: multiple devices and sharding.  Runs on CPU:0..3 (no GPU needed), and on virtual GPUs with
  UC_VIRTUAL_GPUS=4 UC_MULTI=CUDA pytest tests/week10"""
import os
import numpy as np
import pytest
from uc import Tensor, nn, optim
from helpers import close

KIND = os.environ.get("UC_MULTI", "CPU")
DEVS = tuple(f"{KIND}:{i}" for i in range(4))
rng = np.random.default_rng(0)
def r(*s): return rng.standard_normal(s).astype(np.float32)

@pytest.mark.parametrize("axis", [0, 1, None])
def test_shard_roundtrip(axis):
  a = r(8, 12)
  t = Tensor(a).shard(DEVS, axis)
  assert t.device == DEVS
  close(t, a)

@pytest.mark.parametrize("axis", [0, 1])
def test_elementwise_on_shards(axis):
  a, b = r(8, 12), r(8, 12)
  close(Tensor(a).shard(DEVS, axis) * Tensor(b).shard(DEVS, axis) + 1.0, a * b + 1)

def test_mixed_sharding_reshards():
  a, b = r(8, 12), r(8, 12)
  close(Tensor(a).shard(DEVS, 0) + Tensor(b).shard(DEVS, 1), a + b)
  close(Tensor(a).shard(DEVS, 0) + Tensor(b).shard(DEVS, None), a + b)

@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_across_shards(axis):
  a = r(8, 12)
  t = Tensor(a).shard(DEVS, axis)
  close(t.sum(axis=axis), a.sum(axis), atol=1e-5)       # along the sharded axis: needs an allreduce
  close(t.sum(axis=1 - axis), a.sum(1 - axis), atol=1e-5)
  close(t.sum(), a.sum(), atol=1e-4)

def test_matmul_sharded_k():
  # contracting over the sharded axis = partial sums + allreduce
  a, b = r(6, 16), r(16, 5)
  close(Tensor(a).shard(DEVS, 1) @ Tensor(b).shard(DEVS, 0), a @ b, atol=1e-4)

def test_tensor_parallel_mlp():
  # Megatron: W1 split by columns, W2 split by rows -> one allreduce at the end
  x, w1, w2 = r(4, 16), r(16, 32), r(32, 16)
  X = Tensor(x).shard(DEVS, None)
  h = (X @ Tensor(w1).shard(DEVS, 1)).relu()
  y = h @ Tensor(w2).shard(DEVS, 0)
  close(y, np.maximum(x @ w1, 0) @ w2, atol=1e-4)

def _train(shard):
  Tensor.manual_seed(0)
  model = nn.Linear(10, 3)
  if shard:   # replicate the weights; each device sees a quarter of every batch
    model.weight, model.bias = model.weight.shard(DEVS, None), model.bias.shard(DEVS, None)
  opt = optim.SGD([model.weight, model.bias], lr=0.1)
  data = np.random.default_rng(1)
  for _ in range(5):
    x, y = data.standard_normal((16, 10)).astype(np.float32), data.integers(0, 3, 16).astype(np.int32)
    X, Y = (Tensor(x).shard(DEVS, 0), Tensor(y).shard(DEVS, 0)) if shard else (Tensor(x), Tensor(y))
    opt.zero_grad(); model(X).cross_entropy(Y).backward(); opt.step()
  return model.weight.numpy()

def test_data_parallel_equals_single_device():
  close(_train(shard=True), _train(shard=False), atol=1e-5)

def test_uneven_shard_is_an_error():
  with pytest.raises(Exception): Tensor(r(7, 3)).shard(DEVS, 0).numpy()
