"""Reference Llama / Qwen2 in plain PyTorch, functional over a HF-named state dict.
RoPE uses the HF "rotate_half" convention (not the interleaved Meta one)."""
import torch
import torch.nn.functional as F

def init(cfg, seed=0):
  g = torch.Generator().manual_seed(seed)
  D, Hd = cfg["dim"], cfg["dim"] // cfg["n_heads"]
  def n(*s, std=0.05): return torch.randn(*s, generator=g) * std
  sd = {"model.embed_tokens.weight": n(cfg["vocab_size"], D, std=0.5), "model.norm.weight": 1 + n(D)}
  if not cfg["tie_embeddings"]: sd["lm_head.weight"] = n(cfg["vocab_size"], D)
  for i in range(cfg["n_layers"]):
    p = f"model.layers.{i}."
    sd.update({p + "input_layernorm.weight": 1 + n(D), p + "post_attention_layernorm.weight": 1 + n(D),
               p + "self_attn.q_proj.weight": n(cfg["n_heads"] * Hd, D), p + "self_attn.k_proj.weight": n(cfg["n_kv_heads"] * Hd, D),
               p + "self_attn.v_proj.weight": n(cfg["n_kv_heads"] * Hd, D), p + "self_attn.o_proj.weight": n(D, cfg["n_heads"] * Hd),
               p + "mlp.gate_proj.weight": n(cfg["hidden_dim"], D), p + "mlp.up_proj.weight": n(cfg["hidden_dim"], D),
               p + "mlp.down_proj.weight": n(D, cfg["hidden_dim"])})
    if cfg["qkv_bias"]:
      sd.update({p + "self_attn.q_proj.bias": n(cfg["n_heads"] * Hd), p + "self_attn.k_proj.bias": n(cfg["n_kv_heads"] * Hd),
                 p + "self_attn.v_proj.bias": n(cfg["n_kv_heads"] * Hd)})
  return sd

def rope(x, pos, theta):
  d = x.shape[-1]
  inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
  f = pos[:, None].float() * inv[None, :]
  cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
  x1, x2 = x[..., : d // 2], x[..., d // 2:]
  return x * cos + torch.cat([-x2, x1], -1) * sin

def rms(x, w, eps): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

def forward(sd, tokens, cfg):
  B, T = tokens.shape
  H, KV, D = cfg["n_heads"], cfg["n_kv_heads"], cfg["dim"]
  Hd = D // H
  x = sd["model.embed_tokens.weight"][tokens]
  pos = torch.arange(T)
  for i in range(cfg["n_layers"]):
    p = f"model.layers.{i}."
    h = rms(x, sd[p + "input_layernorm.weight"], cfg["norm_eps"])
    def proj(name): return F.linear(h, sd[p + f"self_attn.{name}.weight"], sd.get(p + f"self_attn.{name}.bias"))
    q = proj("q_proj").reshape(B, T, H, Hd).transpose(1, 2)
    k = proj("k_proj").reshape(B, T, KV, Hd).transpose(1, 2)
    v = proj("v_proj").reshape(B, T, KV, Hd).transpose(1, 2)
    q, k = rope(q, pos, cfg["rope_theta"]), rope(k, pos, cfg["rope_theta"])
    k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
    x = x + F.linear(a, sd[p + "self_attn.o_proj.weight"])
    h = rms(x, sd[p + "post_attention_layernorm.weight"], cfg["norm_eps"])
    x = x + F.linear(F.silu(F.linear(h, sd[p + "mlp.gate_proj.weight"])) * F.linear(h, sd[p + "mlp.up_proj.weight"]), sd[p + "mlp.down_proj.weight"])
  x = rms(x, sd["model.norm.weight"], cfg["norm_eps"])
  return F.linear(x, sd["lm_head.weight"] if not cfg["tie_embeddings"] else sd["model.embed_tokens.weight"])
