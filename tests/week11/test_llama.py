"""Week 11: a modern decoder (RoPE, GQA, RMSNorm, SwiGLU) with a KV cache.  Random tiny weights vs tests/ref."""
import numpy as np
import pytest
from uc import Tensor
from helpers import close, DEVICE
from ref import configs

torch = pytest.importorskip("torch")
from ref import llama_torch  # noqa: E402

def to_uc(sd): return {k: Tensor(v.detach().numpy(), device=DEVICE) for k, v in sd.items()}

@pytest.mark.parametrize("cfg_name", ["LLAMA_TINY", "QWEN_TINY"])
def test_forward(cfg_name):
  from uc.models.llama import Llama
  cfg = getattr(configs, cfg_name)
  sd = llama_torch.init(cfg, seed=0)
  tokens = np.random.default_rng(0).integers(0, cfg["vocab_size"], (2, 13)).astype(np.int32)
  ref = llama_torch.forward(sd, torch.tensor(tokens).long(), cfg).detach().numpy()
  m = Llama.from_hf(to_uc(sd), cfg)
  close(m(Tensor(tokens, device=DEVICE)), ref, atol=2e-4, rtol=1e-3)

def test_kv_cache_matches_full_forward():
  from uc.models.llama import Llama
  cfg = configs.LLAMA_TINY
  sd = llama_torch.init(cfg, seed=1)
  tokens = np.random.default_rng(1).integers(0, cfg["vocab_size"], (1, 12)).astype(np.int32)
  full = llama_torch.forward(sd, torch.tensor(tokens).long(), cfg).detach().numpy()
  m = Llama.from_hf(to_uc(sd), cfg)
  out = m(Tensor(tokens[:, :8], device=DEVICE), start_pos=0)          # prefill
  close(out, full[:, :8], atol=2e-4, rtol=1e-3)
  for t in range(8, 12):                                              # decode one token at a time
    step = m(Tensor(tokens[:, t:t + 1], device=DEVICE), start_pos=t)
    close(step, full[:, t:t + 1], atol=2e-4, rtol=1e-3)

def test_greedy_generate_is_deterministic():
  from uc.models.llama import Llama
  cfg = configs.QWEN_TINY
  m = Llama.from_hf(to_uc(llama_torch.init(cfg, seed=2)), cfg)
  prompt = Tensor(np.array([[1, 2, 3]], np.int32), device=DEVICE)
  def gen():
    m.reset_cache() if hasattr(m, "reset_cache") else None
    toks, logits = [], m(prompt, start_pos=0)
    for i in range(5):
      nxt = int(logits[0, -1].argmax().item()); toks.append(nxt)
      logits = m(Tensor(np.array([[nxt]], np.int32), device=DEVICE), start_pos=3 + i)
    return toks
  # the reference: argmax of a full forward at every step
  sd = llama_torch.init(cfg, seed=2); seq = [1, 2, 3]
  for _ in range(5):
    seq.append(int(llama_torch.forward(sd, torch.tensor([seq]), cfg)[0, -1].argmax()))
  assert gen() == seq[3:] == gen()
