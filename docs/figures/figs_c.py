"""Weeks 9-14 figures."""
from svg import SVG, C, MUTED, INK, LINE
FIGS = {}
def fig(path):
  def deco(fn): FIGS[path] = fn; return fn
  return deco

# ============================================================ week 9
@fig("week09/figures/gemm_hierarchy.svg")
def gemm_hierarchy():
  s = SVG(980, 455, "A fast GEMM is tiles inside tiles", "each level stages data one step closer to the tensor cores; every level is a split range in your IR")
  # level boxes
  s.box(30, 100, 280, 280, "", "orange")
  s.text(170, 120, "C, tiled for the grid", size=13, anchor="middle", weight=700, color=C["orange"][2])
  for r in range(4):
    for c in range(4):
      s.rect(55 + c * 58, 140 + r * 56, 58, 56, "#2a9d9d" if (r, c) == (1, 2) else "#fdf0e1", "#e08a2c", 0, sw=1)
  s.text(170, 398, "block tile 128×128 → one CTA", size=12, anchor="middle", color=MUTED)
  s.line(55 + 3 * 58, 140 + 56, 360, 110, LINE, dash="3 3"); s.line(55 + 3 * 58, 140 + 112, 360, 380, LINE, dash="3 3")
  s.box(360, 100, 280, 280, "", "teal")
  s.text(500, 120, "block tile, split across warps", size=13, anchor="middle", weight=700, color=C["teal"][2])
  for r in range(2):
    for c in range(4):
      s.rect(385 + c * 58, 140 + r * 112, 58, 112, "#2e9e6a" if (r, c) == (0, 1) else "#e2f4f4", "#2a9d9d", 0, sw=1)
  s.text(500, 398, "warp tile 64×32 (8 warps)", size=12, anchor="middle", color=MUTED)
  s.line(385 + 2 * 58, 140, 690, 110, LINE, dash="3 3"); s.line(385 + 2 * 58, 252, 690, 380, LINE, dash="3 3")
  s.box(690, 100, 260, 280, "", "green")
  s.text(820, 120, "warp tile in mma fragments", size=13, anchor="middle", weight=700, color=C["green"][2])
  for r in range(4):
    for c in range(4):
      s.rect(715 + c * 52, 140 + r * 52, 52, 52, "#2e9e6a" if (r, c) == (0, 0) else "#e5f5ec", "#2e9e6a", 0, sw=1)
  s.text(820, 398, "mma.m16n8k16: 16×8 per instruction", size=12, anchor="middle", color=MUTED)
  s.text(490, 430, "global → (cp.async / TMA) → shared memory → (ldmatrix) → registers → (mma) → accumulators in registers", size=12, anchor="middle", color=INK)
  return s

@fig("week09/figures/pipeline_smem.svg")
def pipeline_smem():
  s = SVG(980, 330, "Double buffering hides the loads", "while the tensor cores chew on k-tile t, the copy engine fetches k-tile t+1 into the other buffer")
  rows = [("load (cp.async)", "green", 0), ("compute (mma)", "teal", 1)]
  for i, (name, col, off) in enumerate(rows):
    y = 120 + i * 70
    s.text(40, y + 20, name, size=12, weight=700, color=C[col][2])
    for t in range(6):
      x = 190 + (t + off) * 110
      if x + 100 > 960: continue
      s.box(x, y, 100, 40, f"tile {t}", col, size=12, sub=f"buffer {t % 2}" if i == 0 else None)
  s.text(490, 280, "without it: load, wait, compute, load, wait, compute — the SM idles for the full memory latency every k-step", size=12, anchor="middle", color=MUTED)
  return s

@fig("week09/figures/flash.svg")
def flash():
  s = SVG(980, 470, "Flash attention: never write the T×T matrix", "for one block of queries, stream over blocks of keys/values, keeping a running max m, sum ℓ and output O in registers")
  # Q block and K/V blocks
  s.box(40, 110, 120, 90, "Q block", "blue", sub="Br × d")
  for j in range(4):
    s.box(220 + j * 90, 110, 80, 90, f"K{j}, V{j}", "green" if j != 2 else "teal", size=12, sub="Bc × d")
  s.arrow(160, 155, 218, 155)
  s.text(400, 225, "stream j = 0, 1, 2, … (only one K/V block in SRAM at a time)", size=11, anchor="middle", color=MUTED)
  s.code(40, 250, 560, [
    "S   = Q · Kjᵀ · (1/√d)          # Br × Bc, registers only",
    "m'  = max(m, rowmax(S))",
    "P   = exp(S − m')",
    "ℓ   = ℓ · exp(m − m') + rowsum(P)",
    "O   = O · exp(m − m') + P · Vj",
    "m   = m'",
    "# after the last block:  O = O / ℓ   (one store)",
  ], size=12, title="online softmax, per key block", color="teal", hl={1: "purple", 3: "purple", 4: "purple"})
  s.rect(640, 110, 310, 330, "#fbfbfd", "#e3e6eb", 10)
  s.text(660, 135, "why it's exact", size=13, weight=700)
  s.text(660, 162, "softmax(x)ᵢ = exp(xᵢ − m) / Σ exp(xⱼ − m)", size=11, mono=True)
  s.text(660, 186, "holds for ANY m. When a new block raises", size=11, color=MUTED)
  s.text(660, 204, "the max from m to m', rescale what you", size=11, color=MUTED)
  s.text(660, 222, "already summed by exp(m − m').", size=11, color=MUTED)
  s.text(660, 258, "memory:  O(T·d)  instead of  O(T²)", size=12, weight=600, color=C["teal"][2])
  s.text(660, 284, "HBM traffic: Q, K, V read once per", size=11, color=MUTED)
  s.text(660, 302, "query block; S and P never leave the SM", size=11, color=MUTED)
  s.text(660, 338, "causal: skip key blocks entirely above", size=11, color=MUTED)
  s.text(660, 356, "the diagonal (≈ 2× fewer FLOPs)", size=11, color=MUTED)
  s.text(660, 392, "in uc: it's a reduce whose accumulator is", size=11, color=C["purple"][2])
  s.text(660, 410, "the tuple (m, ℓ, O) — a custom combine.", size=11, color=C["purple"][2])
  return s

# ============================================================ week 10
@fig("week10/figures/sharding.svg")
def sharding():
  s = SVG(980, 445, "Two ways to split a layer y = relu(x W₁) W₂ over 4 devices", "data parallel splits the batch; tensor parallel (Megatron) splits the weights")
  cols = ["blue", "green", "orange", "purple"]
  s.text(240, 95, "data parallel", size=14, anchor="middle", weight=700)
  for d in range(4):
    y = 115 + d * 62
    s.box(40, y, 90, 48, f"x[{d}::4]", cols[d], size=12, mono=True, sub="¼ of batch")
    s.box(160, y, 120, 48, "W₁, W₂", "gray", size=12, sub="full copy")
    s.box(310, y, 120, 48, f"∇W on dev {d}", cols[d], size=12)
  s.box(250, 380, 240, 40, "allreduce(∇W), then step", "red", size=12)
  s.arrow(370, 351, 370, 379, color="#d64545")
  s.text(740, 95, "tensor parallel", size=14, anchor="middle", weight=700)
  for d in range(4):
    y = 115 + d * 62
    s.box(530, y, 90, 48, "x", "gray", size=12, mono=True, sub="replicated")
    s.box(640, y, 120, 48, f"W₁ cols {d}", cols[d], size=12, sub="column shard")
    s.box(780, y, 150, 48, f"W₂ rows {d}", cols[d], size=12, sub="row shard → partial y")
  s.box(720, 380, 240, 40, "allreduce(partial y): one per MLP", "red", size=12)
  s.arrow(855, 351, 855, 379, color="#d64545")
  s.line(505, 100, 505, 425, color="#e3e6eb")
  return s

@fig("week10/figures/ring.svg")
def ring():
  s = SVG(980, 420, "Ring allreduce", "reduce-scatter (N−1 steps) then all-gather (N−1 steps): each device sends 2(N−1)/N of the data, independent of N")
  import math
  cx, cy, R, gap = 220, 252, 112, 0.36          # gap: where an arc stops short of a device (radians)
  cols = ["blue", "green", "orange", "purple"]
  for d in range(4):
    a = -math.pi / 2 + d * math.pi / 2
    s.arc(cx, cy, R, a + gap, a + math.pi / 2 - gap)
    s.circle(cx + R * math.cos(a), cy + R * math.sin(a), 34, cols[d], f"dev {d}", size=12)
  s.text(cx, cy, "each device sends\nto its neighbour", size=11, anchor="middle", color=MUTED)
  # chunk table
  x0, y0, cw = 470, 140, 100
  for c in range(4): s.text(x0 + 95 + c * cw, y0 - 14, f"chunk {c}", size=11, anchor="middle", color=MUTED, weight=600)
  after = [["a0", "Σ1", "a2", "a3"], ["b0", "b1", "Σ2", "b3"], ["c0", "c1", "c2", "Σ3"], ["Σ0", "d1", "d2", "d3"]]
  for d in range(4):
    s.text(x0, y0 + 16 + d * 34, f"dev {d}", size=11, color=C[cols[d]][2], weight=600)
    for c in range(4):
      v = after[d][c]
      s.box(x0 + 50 + c * cw, y0 + d * 34, 90, 30, v, "red" if v.startswith("Σ") else "white", size=12, mono=True, rx=4)
  s.text(x0 + 225, y0 + 160, "after reduce-scatter: each device owns one fully reduced chunk", size=11, anchor="middle", color=MUTED)
  s.text(x0 + 225, y0 + 190, "all-gather then circulates the Σ chunks the same way", size=11, anchor="middle", color=MUTED)
  s.text(x0 + 225, y0 + 240, "on one H100: virtual devices make this testable; a real 8-GPU node makes it fast", size=11, anchor="middle", color=C["purple"][2])
  return s

# ============================================================ week 11
@fig("week11/figures/llama_block.svg")
def llama_block():
  s = SVG(980, 540, "One decoder block (Llama / Qwen2)", "pre-norm residual stream · RoPE on q and k · grouped-query attention · SwiGLU MLP   (data flows upward)")
  L, R = 200, 640                      # column centres
  # ---- attention half
  s.text(L, 92, "attention", size=13, anchor="middle", weight=700, color=MUTED)
  s.box(L - 110, 480, 220, 34, "x   (B, T, D)", "gray", mono=True, size=12)
  s.box(L - 110, 424, 220, 34, "RMSNorm", "purple")
  s.box(L - 140, 364, 280, 40, "q, k, v = h·W_q, h·W_k, h·W_v", "blue", mono=True, size=12)
  s.box(L - 110, 308, 220, 34, "RoPE(q), RoPE(k)", "orange")
  s.box(L - 110, 244, 220, 44, "SDPA  causal, GQA", "teal", size=12, sub="flash attention (week 9)")
  s.box(L - 110, 190, 220, 34, "attn · W_o", "blue", mono=True, size=12)
  s.circle(L, 145, 13, "gray", "+")
  for y1, y2 in [(480, 459), (424, 405), (364, 343), (308, 289), (244, 225), (190, 159)]: s.arrow(L, y1, L, y2)
  s.path([(L + 110, 497), (L + 170, 497), (L + 170, 145), (L + 14, 145)], color="#5b6474")
  s.text(L + 182, 320, "residual", size=11, color=MUTED)
  # ---- connector
  s.path([(L, 132), (L, 112), (440, 112), (440, 497), (R - 111, 497)], color="#5b6474")
  # ---- MLP half
  s.text(R, 92, "MLP (SwiGLU)", size=13, anchor="middle", weight=700, color=MUTED)
  s.box(R - 110, 480, 220, 34, "RMSNorm", "purple")
  s.box(R - 150, 410, 140, 40, "gate = h·W_g", "blue", mono=True, size=12)
  s.box(R + 10, 410, 140, 40, "up = h·W_u", "blue", mono=True, size=12)
  s.box(R - 110, 340, 220, 38, "silu(gate) · up", "orange", mono=True, size=12)
  s.box(R - 110, 280, 220, 36, "· W_d", "blue", mono=True, size=12)
  s.circle(R, 230, 13, "gray", "+")
  s.arrow(R - 40, 480, R - 80, 451); s.arrow(R + 40, 480, R + 80, 451)
  s.arrow(R - 80, 410, R - 40, 379); s.arrow(R + 80, 410, R + 40, 379)
  s.arrow(R, 340, R, 317); s.arrow(R, 280, R, 244)
  s.path([(440, 230), (R - 14, 230)], color="#5b6474")
  s.text(470, 220, "residual", size=11, color=MUTED)
  s.arrow(R, 216, R, 120, label="next block", loff=(46, 0))
  s.code(800, 300, 160, ["decode, batch 1:", "every weight is", "read once per", "token, so it is", "HBM-bandwidth", "bound"], size=11, color="yellow")
  return s

@fig("week11/figures/kv_cache.svg")
def kv_cache():
  s = SVG(980, 340, "The KV cache turns O(T²) recompute into O(T) reads", "prefill the prompt once; each new token appends one k, v row and attends over the cache")
  for t in range(10):
    filled = t < 6
    s.rect(80 + t * 50, 120, 50, 40, "#e5f5ec" if filled else ("#e8effc" if t == 6 else "#fafbfc"), "#9aa3b2", 0, sw=1)
    s.rect(80 + t * 50, 170, 50, 40, "#e5f5ec" if filled else ("#e8effc" if t == 6 else "#fafbfc"), "#9aa3b2", 0, sw=1)
  s.text(60, 140, "K", size=13, anchor="end", weight=700); s.text(60, 190, "V", size=13, anchor="end", weight=700)
  s.text(230, 228, "prompt (prefill)", size=11, anchor="middle", color=C["green"][2])
  s.text(405, 228, "new", size=11, anchor="middle", color=C["blue"][2])
  s.text(505, 228, "preallocated, max_seq_len", size=11, anchor="middle", color=MUTED)
  s.code(620, 100, 330, ["# decode step at position p", "q, k, v = proj(x)        # T = 1", "K[:, p] = k;  V[:, p] = v", "attn over K[:, :p+1]", "", "fixed-size cache + a mask", "→ same shapes every step", "→ the JIT can replay it"], size=11, color="blue", hl={6: "blue", 7: "blue"})
  s.text(490, 300, "growing the cache with cat() changes shapes every step, so every step recompiles: fine for tests, fatal for speed", size=12, anchor="middle", color=C["red"][2])
  return s

@fig("week11/figures/train_step.svg")
def train_step():
  s = SVG(980, 300, "One training step, captured once and replayed", "the JIT records the kernel launches of a step; later steps rebind the input buffers and launch the same list")
  segs = [("load batch", 100, "gray"), ("forward (bf16)", 240, "blue"), ("backward", 340, "red"), ("clip + AdamW", 200, "purple")]
  x = 40
  for name, w, col in segs:
    s.box(x, 120, w, 50, name, col, size=12)
    x += w + 6
  s.text(490, 205, "first call: trace → schedule → compile → run    ·    later calls: run only", size=12, anchor="middle")
  s.text(490, 240, "MFU = (6·N·tokens/s + attention FLOPs) / peak FLOP/s.  Report yours next to PyTorch's on the same GPU.", size=12, anchor="middle", color=MUTED)
  return s

# ============================================================ weeks 12-14
@fig("week12/figures/timeline.svg")
def timeline():
  s = SVG(980, 240, "Final project, three weeks", "propose · build · measure · write it up")
  ms = [("proposal", "1 page: claim,\nbaseline, metric", "yellow"), ("week 12", "reproduce the baseline\nend to end", "blue"),
        ("week 13", "the idea, behind a flag;\ntests that pin it", "orange"), ("week 14", "ablations, then\nthe report", "green")]
  for i, (a, b, col) in enumerate(ms):
    x = 40 + i * 235
    s.box(x, 110, 210, 50, a, col, size=14, weight=700)
    s.text(x + 105, 192, b, size=11, anchor="middle", color=MUTED)
    if i: s.arrow(x - 24, 135, x - 2, 135)
  return s
