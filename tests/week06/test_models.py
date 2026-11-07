"""Week 6: inference with real architectures.  Random weights vs. the torch references in tests/ref;
real checkpoints run only if the files exist (see tools/download.py)."""
import pathlib
import numpy as np
import pytest
from uc import Tensor
from helpers import close, slow
from ref import configs

torch = pytest.importorskip("torch")
DATA = pathlib.Path(__file__).resolve().parents[2] / "data"

def to_uc(sd): return {k: Tensor(v.detach().numpy()) for k, v in sd.items()}

def test_gpt2_tiny():
  from ref import gpt2_torch
  from uc.models.gpt2 import GPT2
  cfg = configs.GPT2_TINY
  sd = gpt2_torch.init(cfg, seed=0)
  tokens = np.random.default_rng(0).integers(0, cfg["vocab_size"], (2, 11)).astype(np.int32)
  ref = gpt2_torch.forward(sd, torch.tensor(tokens).long(), cfg).detach().numpy()
  model = GPT2.from_hf(to_uc(sd), cfg)
  close(model(Tensor(tokens)), ref, atol=1e-4, rtol=1e-3)

def test_gpt2_strips_prefix():
  from ref import gpt2_torch
  from uc.models.gpt2 import GPT2
  cfg = configs.GPT2_TINY
  sd = {"transformer." + k: v for k, v in gpt2_torch.init(cfg, seed=1).items()}
  model = GPT2.from_hf(to_uc(sd), cfg)
  tokens = np.zeros((1, 3), np.int32)
  close(model(Tensor(tokens)), gpt2_torch.forward(sd, torch.tensor(tokens).long(), cfg).detach().numpy(), atol=1e-4, rtol=1e-3)

def test_resnet18_random():
  from ref import resnet_torch
  from uc.models.resnet import ResNet18
  sd = resnet_torch.init(num_classes=10, seed=0)
  x = np.random.default_rng(0).standard_normal((2, 3, 64, 64)).astype(np.float32)
  ref = resnet_torch.forward(sd, torch.tensor(x)).numpy()
  from uc.nn.state import load_state_dict
  m = ResNet18(num_classes=10)
  load_state_dict(m, to_uc(sd))      # your attribute names must match torchvision's
  close(m(Tensor(x)), ref, atol=1e-3, rtol=1e-3)

@slow
def test_gpt2_124m_real():
  path = DATA / "gpt2" / "model.safetensors"
  if not path.exists(): pytest.skip("run: python tools/download.py gpt2")
  from ref import gpt2_torch
  from uc.models.gpt2 import GPT2
  from uc.nn.state import safe_load
  cfg = configs.GPT2_124M
  model = GPT2.from_hf(safe_load(str(path)), cfg)
  tokens = np.array([[15496, 11, 314, 1101, 257, 3303, 2746, 11]], np.int32)   # "Hello, I'm a language model,"
  from ref.st_numpy import load_torch
  ref = gpt2_torch.forward(load_torch(str(path)), torch.tensor(tokens).long(), cfg).detach().numpy()
  close(model(Tensor(tokens)), ref, atol=2e-3, rtol=2e-3)

@slow
def test_resnet18_real():
  path = DATA / "resnet18" / "model.safetensors"
  if not path.exists(): pytest.skip("run: python tools/download.py resnet18")
  from ref import resnet_torch
  from ref.st_numpy import load_torch
  from uc.models.resnet import ResNet18
  from uc.nn.state import safe_load, load_state_dict
  keep = lambda k: not k.endswith("num_batches_tracked")
  m = ResNet18(); load_state_dict(m, {k: v for k, v in safe_load(str(path)).items() if keep(k)})
  x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
  ref = resnet_torch.forward({k: v for k, v in load_torch(str(path)).items() if keep(k)}, torch.tensor(x)).numpy()
  close(m(Tensor(x)), ref, atol=2e-3, rtol=2e-3)
