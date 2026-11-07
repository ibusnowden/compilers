"""Model configs used by the tests.  For real checkpoints, build the config from the
downloaded config.json with the helpers below instead of trusting these by hand."""

GPT2_TINY = dict(vocab_size=96, n_ctx=32, n_embd=48, n_head=4, n_layer=2, layer_norm_eps=1e-5)
GPT2_124M = dict(vocab_size=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, layer_norm_eps=1e-5)

LLAMA_TINY = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=128, hidden_dim=160, norm_eps=1e-6,
                  rope_theta=10000.0, max_seq_len=64, qkv_bias=False, tie_embeddings=False)
QWEN_TINY = dict(LLAMA_TINY, qkv_bias=True, tie_embeddings=True, rope_theta=1_000_000.0)

def llama_config_from_hf(cfg: dict) -> dict:
  """config.json of a HF Llama / Qwen2 checkpoint -> our config dict"""
  return dict(dim=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"], n_heads=cfg["num_attention_heads"],
              n_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]), vocab_size=cfg["vocab_size"],
              hidden_dim=cfg["intermediate_size"], norm_eps=cfg["rms_norm_eps"], rope_theta=float(cfg.get("rope_theta", 10000.0)),
              max_seq_len=min(cfg.get("max_position_embeddings", 4096), 4096),
              qkv_bias=cfg.get("model_type") == "qwen2" or cfg.get("attention_bias", False),
              tie_embeddings=cfg.get("tie_word_embeddings", False), rope_scaling=cfg.get("rope_scaling"))
