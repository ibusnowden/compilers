"""Reference GPT-2 in plain PyTorch, functional over a HF-named state dict."""
import math, torch
import torch.nn.functional as F

def init(cfg, seed=0):
  g = torch.Generator().manual_seed(seed)
  C, V, T, L = cfg["n_embd"], cfg["vocab_size"], cfg["n_ctx"], cfg["n_layer"]
  def n(*s, std=0.02): return torch.randn(*s, generator=g) * std
  sd = {"wte.weight": n(V, C), "wpe.weight": n(T, C), "ln_f.weight": torch.ones(C), "ln_f.bias": torch.zeros(C)}
  for i in range(L):
    p = f"h.{i}."
    sd.update({p + "ln_1.weight": torch.ones(C) + n(C), p + "ln_1.bias": n(C), p + "ln_2.weight": torch.ones(C) + n(C), p + "ln_2.bias": n(C),
               p + "attn.c_attn.weight": n(C, 3 * C), p + "attn.c_attn.bias": n(3 * C),
               p + "attn.c_proj.weight": n(C, C, std=0.02 / math.sqrt(2 * L)), p + "attn.c_proj.bias": n(C),
               p + "mlp.c_fc.weight": n(C, 4 * C), p + "mlp.c_fc.bias": n(4 * C),
               p + "mlp.c_proj.weight": n(4 * C, C, std=0.02 / math.sqrt(2 * L)), p + "mlp.c_proj.bias": n(C)})
  return sd

def forward(sd, tokens, cfg):
  sd = {k.removeprefix("transformer."): v for k, v in sd.items()}
  B, T = tokens.shape
  C, H, eps = cfg["n_embd"], cfg["n_head"], cfg["layer_norm_eps"]
  x = sd["wte.weight"][tokens] + sd["wpe.weight"][:T]
  for i in range(cfg["n_layer"]):
    p = f"h.{i}."
    h = F.layer_norm(x, (C,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"], eps)
    qkv = h @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]
    q, k, v = (t.reshape(B, T, H, C // H).transpose(1, 2) for t in qkv.split(C, dim=-1))
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    x = x + a @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
    h = F.layer_norm(x, (C,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"], eps)
    h = F.gelu(h @ sd[p + "mlp.c_fc.weight"] + sd[p + "mlp.c_fc.bias"], approximate="tanh")
    x = x + h @ sd[p + "mlp.c_proj.weight"] + sd[p + "mlp.c_proj.bias"]
  x = F.layer_norm(x, (C,), sd["ln_f.weight"], sd["ln_f.bias"], eps)
  return x @ sd["wte.weight"].T
