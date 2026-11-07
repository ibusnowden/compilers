"""Week 6: train on MNIST.  Needs data/mnist (python tools/download.py mnist)."""
import gzip, pathlib
import numpy as np
import pytest
from uc import Tensor, optim
from uc.nn.state import get_parameters
from helpers import slow

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "mnist"

def load(name, offset):
  with gzip.open(DATA / name) as f: return np.frombuffer(f.read(), np.uint8, offset=offset)

@pytest.fixture(scope="module")
def mnist():
  if not (DATA / "train-images-idx3-ubyte.gz").exists(): pytest.skip("run: python tools/download.py mnist")
  X = load("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
  Y = load("train-labels-idx1-ubyte.gz", 8).astype(np.int32)
  Xt = load("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
  Yt = load("t10k-labels-idx1-ubyte.gz", 8).astype(np.int32)
  return X, Y, Xt, Yt

def train(model, X, Y, steps, bs=128, lr=1e-3):
  opt = optim.AdamW(get_parameters(model), lr=lr, weight_decay=0.0)
  rng = np.random.default_rng(0)
  Tensor.training = True
  for _ in range(steps):
    i = rng.integers(0, len(X), bs)
    opt.zero_grad()
    loss = model(Tensor(X[i])).cross_entropy(Tensor(Y[i]))
    loss.backward(); opt.step()
  Tensor.training = False

def accuracy(model, Xt, Yt):
  with Tensor.no_grad():
    preds = np.concatenate([model(Tensor(Xt[i:i + 1000])).argmax(-1).numpy() for i in range(0, len(Xt), 1000)])
  return (preds == Yt).mean()

@slow
def test_mlp(mnist):
  from uc.models.mnist import MLP
  X, Y, Xt, Yt = mnist
  Tensor.manual_seed(0); m = MLP()
  train(m, X, Y, steps=1000)
  assert accuracy(m, Xt, Yt) >= 0.96

@slow
def test_cnn(mnist):
  from uc.models.mnist import CNN
  X, Y, Xt, Yt = mnist
  Tensor.manual_seed(0); m = CNN()
  train(m, X, Y, steps=1000)
  assert accuracy(m, Xt, Yt) >= 0.98
