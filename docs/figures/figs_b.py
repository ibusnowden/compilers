"""Weeks 4-8 figures."""
from svg import SVG, C, MUTED, INK, LINE
FIGS = {}
def fig(path):
  def deco(fn): FIGS[path] = fn; return fn
  return deco

# ============================================================ week 4
@fig("week04/figures/rangeify.svg")
def rangeify():
  s = SVG(980, 420, "Rangeify: shapes become loops", "sum(x · y) over a (4,) tensor: give every op the ranges of its output, then close reduces")
  s.text(150, 92, "tensor graph (shapes)", size=13, anchor="middle", weight=700, color=MUTED)
  s.node(150, 130, "REDUCE add  ()", "orange", w=150); s.node(150, 200, "MUL  (4,)", "blue", w=110)
  s.node(90, 270, "x  (4,)", "green", w=80); s.node(210, 270, "y  (4,)", "green", w=80)
  s.line(150, 145, 150, 185); s.line(150, 215, 90, 255); s.line(150, 215, 210, 255)
  s.arrow(285, 200, 365, 200, label="rangeify", color="#e08a2c")
  s.text(560, 92, "kernel graph (ranges)", size=13, anchor="middle", weight=700, color=MUTED)
  s.node(560, 125, "STORE out[0]", "green", w=130); s.node(560, 180, "REDUCE add over r", "orange", w=170)
  s.node(560, 235, "MUL", "blue", w=60); s.node(480, 295, "LOAD x[r]", "green", w=100); s.node(640, 295, "LOAD y[r]", "green", w=100)
  s.node(560, 360, "r = RANGE 4", "orange", w=120)
  for a, b in [((560, 140), (560, 165)), ((560, 195), (560, 220)), ((560, 250), (480, 280)), ((560, 250), (640, 280)), ((480, 310), (545, 345)), ((640, 310), (575, 345))]:
    s.line(*a, *b)
  s.curve(645, 180, 800, 300, 620, 360, color="#e08a2c", dash="4 3")
  s.text(722, 300, "creates r,\nand closes it", size=11, color=C["orange"][2])
  s.code(760, 120, 200, ["float acc = 0;", "for (r = 0; r < 4; r++)", "  acc += x[r]*y[r];", "out[0] = acc;"], size=11, title="rendered", color="orange")
  return s

@fig("week04/figures/materialize.svg")
def materialize():
  s = SVG(980, 400, "To materialize or to recompute?", "y = x − max(x):  the scalar max(x) is needed by every element")
  s.text(250, 92, "fuse: 1 kernel, O(n²) work", size=14, anchor="middle", weight=700, color=C["orange"][2])
  s.code(40, 108, 420, ["for i in range(n):            # output", "  m = -inf", "  for r in range(n):          # reduce, again", "    m = max(m, x[r])", "  y[i] = x[i] - m"], size=12, color="orange", hl={2: "red", 3: "red"})
  s.text(730, 92, "materialize: 2 kernels, O(n) work", size=14, anchor="middle", weight=700, color=C["green"][2])
  s.code(520, 108, 420, ["# kernel 1", "m = -inf", "for r in range(n): m = max(m, x[r])", "M[0] = m                      # buffer", "# kernel 2", "for i in range(n): y[i] = x[i] - M[0]"], size=12, color="green", hl={3: "green"})
  s.rect(40, 256, 900, 120, "#fbfbfd", "#e3e6eb", 10)
  s.text(60, 281, "A workable default policy (week 4):", size=13, weight=700)
  s.text(60, 308, "• materialize a REDUCE whose result is EXPANDed (used by more outputs than it has elements)", size=12, color=MUTED)
  s.text(60, 330, "• materialize anything marked contiguous, and every program output", size=12, color=MUTED)
  s.text(60, 352, "• fuse everything else: elementwise ops and movement ops are always free to recompute", size=12, color=MUTED)
  return s

@fig("week04/figures/reduce_acc.svg")
def reduce_acc():
  s = SVG(980, 330, "A reduce is a register, a loop, and an identity", "REDUCE(op, src, ranges)  →  acc = identity(op);  for ranges: acc = op(acc, src);  use acc")
  rows = [("add", "0", "acc + v"), ("mul", "1", "acc * v"), ("max", "-inf  /  INT_MIN", "max(acc, v)"), ("min", "+inf  /  INT_MAX", "min(acc, v)")]
  x0 = 60
  for j, hdr in enumerate(["op", "identity", "update"]):
    s.box(x0 + j * 170, 100, 170, 34, hdr, "orange", size=12, weight=700, rx=0)
  for i, row in enumerate(rows):
    for j, v in enumerate(row):
      s.box(x0 + j * 170, 134 + i * 34, 170, 34, v, "white", size=12, mono=True, rx=0, weight=400)
  s.code(600, 100, 340, ["float acc = -INFINITY;     // identity", "for (int r = 0; r < N; r++) {", "  float v = x[r];", "  acc = fmaxf(acc, v);", "}", "out[0] = acc;"], size=12, title="max, rendered", color="orange", hl={0: "orange", 3: "orange"})
  s.text(320, 300, "the loop must be sequential (or tree-shaped) — week 7 and 8 parallelize it", size=12, anchor="middle", color=MUTED)
  return s

# ============================================================ week 5
@fig("week05/figures/views.svg")
def views():
  s = SVG(980, 450, "Movement ops are index arithmetic", "nothing moves: each op rewrites the index expression used to LOAD from the original buffer")
  # buffer
  s.text(60, 100, "buffer (6 elements)", size=12, weight=700, color=MUTED)
  s.grid(60, 112, 1, 6, 40, "#e5f5ec", [[0, 1, 2, 3, 4, 5]])
  s.text(60, 175, "reshape [2 3]", size=12, weight=700, color=MUTED)
  s.grid(60, 187, 2, 3, 40, "#e8effc", [[0, 1, 2], [3, 4, 5]])
  s.text(60, 290, "permute [1 0]", size=12, weight=700, color=MUTED)
  s.grid(60, 302, 3, 2, 40, "#efeafa", [[0, 3], [1, 4], [2, 5]])
  s.code(330, 100, 620, [
    "out[i]     = buf[i]",
    "",
    "reshape:   v[i, j]   = buf[i*3 + j]",
    "permute:   t[i, j]   = v[j, i]          = buf[j*3 + i]",
    "reshape:   f[k]      = t[k / 2, k % 2]  = buf[(k % 2)*3 + k / 2]",
    "",
    "pad [[1 1]]:     p[i] = valid(i) ? buf[i - 1] : 0,",
    "                 valid(i) = 1 ≤ i  &&  i < 7",
    "shrink [[2 5]]:  s[i] = buf[i + 2]",
    "flip [0]:        r[i] = buf[5 - i]",
    "expand [6 4]:    e[i, j] = buf[i]          (j is ignored)",
  ], size=12, title="the index expression each view produces", color="blue", hl={3: "purple", 4: "purple"})
  s.text(640, 372, "(i*3 + j)/3 → i  and  (i*3 + j)%3 → j  are week-1 rule 5:", size=12, anchor="middle", color=C["purple"][2])
  s.text(640, 392, "a reshape round-trip simplifies away, so it costs nothing", size=12, anchor="middle", color=C["purple"][2])
  return s

@fig("week05/figures/matmul_expand.svg")
def matmul_expand():
  s = SVG(980, 420, "Matmul from movement ops", "A (M,K) → reshape (M,1,K) → expand (M,N,K);   Bᵀ (N,K) → reshape (1,N,K) → expand (M,N,K);   multiply; reduce pop=1")
  def slab(x, y, rows, cols, color, label, sub):
    f, st, t = C[color]
    for r in range(rows):
      for c in range(cols):
        s.rect(x + c * 26, y + r * 26, 26, 26, f, st, 0, sw=1)
    s.text(x + cols * 13, y - 12, label, size=12, anchor="middle", weight=700, color=t)
    s.text(x + cols * 13, y + rows * 26 + 16, sub, size=11, anchor="middle", color=MUTED)
  slab(50, 130, 2, 3, "blue", "A (2,3)", "rows i, cols k")
  slab(50, 270, 2, 3, "green", "Bᵀ (2,3)", "rows j, cols k")
  s.arrow(140, 156, 236, 176, label="expand over j", loff=(20, -14))
  s.arrow(140, 296, 236, 256, label="expand over i", loff=(28, 18))
  # cube of products (i,j) x k
  for i in range(2):
    for j in range(2):
      x, y = 250 + j * 120, 150 + i * 90
      for k in range(3):
        s.rect(x + k * 26, y, 26, 26, "#fdf0e1", "#e08a2c", 0, sw=1)
      s.text(x + 39, y + 40, f"(i={i}, j={j})", size=10, anchor="middle", color=MUTED)
  s.text(349, 128, "a · b   shape (2, 2, 3): one strip of k per (i, j)", size=12, anchor="middle", weight=700, color=C["orange"][2])
  s.arrow(468, 210, 572, 210, label="reduce pop=1", color="#e08a2c")
  s.grid(590, 174, 2, 2, 36, "#fdf0e1", [["Σk", "Σk"], ["Σk", "Σk"]])
  s.text(626, 158, "C (2, 2)", size=12, anchor="middle", weight=700, color=C["orange"][2])
  s.code(700, 120, 260, ["for i in range(2):", "  for j in range(2):", "    acc = 0", "    for k in range(3):", "      acc += A[i*3+k]", "           * B[k*2+j]", "    C[i*2+j] = acc"], size=11, title="after rangeify: 1 kernel", color="orange")
  s.text(490, 390, "expand costs nothing: it just ignores an index. Nothing of size M·N·K is ever allocated.", size=12, anchor="middle", color=MUTED)
  return s

@fig("week05/figures/pad_mask.svg")
def pad_mask():
  s = SVG(980, 300, "pad is a guarded load", "pad [1 2 3] [[1 2]]  →  [0 1 2 3 0 0]")
  s.grid(70, 120, 1, 6, 46, lambda r, c: "#e5f5ec" if 1 <= c <= 3 else "#f1f3f6", [["0", "1", "2", "3", "0", "0"]], size=13)
  for c in range(6): s.text(70 + c * 46 + 23, 185, f"i={c}", size=11, anchor="middle", color=MUTED, mono=True)
  s.code(420, 100, 520, ["valid = (1 <= i) & (i < 4)", "out[i] = valid ? buf[i - 1] : 0", "", "# never load buf[-1]: a gated load, or", "# clamp the index:  buf[max(0, min(i-1, 2))]"], size=12, color="green", title="rendered", hl={1: "green"})
  s.text(490, 270, "the index i−1 has bounds [−1, 4]: week-2 bounds checking only accepts it under the gate", size=12, anchor="middle", color=C["red"][2])
  return s

# ============================================================ week 6
@fig("week06/figures/autodiff.svg")
def autodiff():
  s = SVG(980, 440, "Reverse mode: one backward pass gives every gradient", "L = sum(relu(x · w)) — the backward graph is just more UOps, built by rewrite rules, fused like any other")
  # forward
  s.text(190, 95, "forward", size=13, anchor="middle", weight=700, color=C["blue"][2])
  fw = [("x", 90, 360, "green"), ("w", 290, 360, "green"), ("MUL", 190, 290, "blue"), ("RELU", 190, 220, "blue"), ("SUM", 190, 150, "blue")]
  for name, x, y, col in fw: s.node(x, y, name, col, w=70)
  s.arrow(100, 345, 175, 305); s.arrow(280, 345, 205, 305); s.arrow(190, 275, 190, 236); s.arrow(190, 205, 190, 166)
  # backward
  s.text(690, 95, "backward (gradients flow down)", size=13, anchor="middle", weight=700, color=C["red"][2])
  bw = [("dL = 1", 690, 150), ("dS = expand(dL)", 690, 220), ("dR = dS · (x·w > 0)", 690, 290), ("dx = dR · w", 580, 365), ("dw = dR · x", 800, 365)]
  for name, x, y in bw: s.node(x, y, name, "red", w=190 if "(" in name else 150)
  s.arrow(690, 166, 690, 204); s.arrow(690, 236, 690, 274); s.arrow(670, 306, 600, 349); s.arrow(710, 306, 780, 349)
  s.code(330, 140, 220, ["SUM    → expand", "EXPAND → sum", "MUL    → swap & mul", "RELU   → mask", "RESHAPE→ reshape back", "PERMUTE→ inverse perm", "PAD    → shrink", "SHRINK → pad"], size=11, title="one rule per op", color="purple")
  s.text(490, 420, "every movement op's gradient is another movement op: they stay free", size=12, anchor="middle", color=MUTED)
  return s

@fig("week06/figures/safetensors.svg")
def safetensors():
  s = SVG(980, 280, "A safetensors file", "trivial to parse, and mmap-able: the tensor bytes need no copying")
  s.box(40, 110, 120, 60, "u64 LE", "gray", mono=True, sub="header size N")
  s.box(160, 110, 360, 60, '{"h.0.attn.c_attn.weight": {"dtype":"F32",', "yellow", mono=True, size=11, sub='"shape":[768,2304], "data_offsets":[0, 7077888]}, ...}')
  s.box(520, 110, 140, 60, "tensor 0", "green", sub="raw bytes")
  s.box(660, 110, 140, 60, "tensor 1", "green", sub="raw bytes")
  s.box(800, 110, 140, 60, "…", "green")
  s.text(100, 195, "8 bytes", size=11, anchor="middle", color=MUTED); s.text(340, 195, "N bytes of JSON", size=11, anchor="middle", color=MUTED)
  s.text(730, 195, "offsets are relative to 8 + N", size=11, anchor="middle", color=MUTED)
  s.text(490, 240, "bf16 has no numpy dtype: load the bits as u16, and let your compiler cast (x as u32) << 16 to f32", size=12, anchor="middle", color=C["purple"][2])
  return s

@fig("week06/figures/schedule.svg")
def schedule():
  s = SVG(980, 350, "realize(): from a lazy graph to launches", "the scheduler cuts the graph at materialized UOps; each piece is one kernel; the cache is keyed by the kernel's AST")
  steps = [("lazy UOp graph", "Tensor ops", "purple"), ("schedule", "cut at buffers →\nlist of kernels", "orange"), ("kernel AST", "rangeified,\nsimplified", "orange"),
           ("cache lookup", "hash-consed AST\nis the key", "gray"), ("compile", "only on a miss", "gray"), ("launch", "bind buffers,\nrun", "green")]
  for i, (a, b, col) in enumerate(steps):
    x = 30 + i * 158
    s.box(x, 120, 138, 70, a, col, size=13, weight=600)
    s.text(x + 69, 220, b, size=11, anchor="middle", color=MUTED)
    if i: s.arrow(x - 19, 155, x - 1, 155)
  hit, launch = 30 + 3 * 158 + 69, 30 + 5 * 158 + 69
  s.path([(hit + 60, 190), (hit + 60, 258), (launch - 60, 258), (launch - 60, 192)], color="#2e9e6a", label="cache hit: skip compile", lpos=(30 + 4 * 158 + 69, 274))
  s.text(490, 318, "hash-consing pays again: identical kernels are the identical UOp, so the cache key is free", size=12, anchor="middle", color=C["purple"][2])
  return s

# ============================================================ week 7
@fig("week07/figures/hierarchy.svg")
def hierarchy():
  s = SVG(980, 440, "The memory hierarchy (one modern x86 core, rough numbers)", "every level is ~10× bigger and ~3–10× slower than the one above it")
  levels = [("registers", "~2 KB", "0 cycles", "~1 KB/cycle"), ("L1d", "48 KB", "~5 cycles", "~128 B/cycle"),
            ("L2", "1–2 MB", "~14 cycles", "~64 B/cycle"), ("L3 (shared)", "32+ MB", "~50 cycles", "~32 B/cycle"),
            ("DRAM", "GBs", "~300 cycles", "~10 B/cycle/core")]
  cols = ["blue", "teal", "teal", "orange", "red"]
  for i, ((name, size, lat, bw), col) in enumerate(zip(levels, cols)):
    w = 150 + i * 100; x = 315 - w / 2; y = 100 + i * 62
    f, st, t = C[col]
    s.poly([(x + 18, y), (x + w - 18, y), (x + w, y + 52), (x, y + 52)], f, st, 1.5)
    s.text(315, y + 26, name, size=13, anchor="middle", weight=700, color=t)
    s.text(640, y + 26, f"{size}", size=12, mono=True); s.text(740, y + 26, lat, size=12, mono=True); s.text(850, y + 26, bw, size=12, mono=True)
  s.text(640, 88, "size", size=11, color=MUTED, weight=700); s.text(740, 88, "latency", size=11, color=MUTED, weight=700); s.text(850, 88, "bandwidth", size=11, color=MUTED, weight=700)
  s.text(490, 425, "a kernel is fast when each byte fetched from a slow level is reused many times from a fast one", size=12, anchor="middle", color=MUTED)
  return s

@fig("week07/figures/roofline.svg")
def roofline():
  s = SVG(980, 440, "The roofline", "attainable FLOP/s = min(peak compute, bandwidth × arithmetic intensity)")
  import math
  x0, y0, W, H = 110, 380, 560, 280
  def X(ai): return x0 + (math.log10(ai) + 2) / 6 * W          # AI from 0.01 to 10^4
  def Y(f): return y0 - (math.log10(f) - 9) / 6 * H             # 1 GFLOP/s to 1 PFLOP/s
  s.line(x0, y0, x0 + W, y0, INK); s.line(x0, y0, x0, y0 - H, INK)
  for ai in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
    s.text(X(ai), y0 + 16, f"{ai:g}", size=10, anchor="middle", color=MUTED, mono=True)
  for p, lab in [(9, "1G"), (10, "10G"), (11, "100G"), (12, "1T"), (13, "10T"), (14, "100T"), (15, "1P")]:
    s.text(x0 - 8, Y(10 ** p), lab, size=10, anchor="end", color=MUTED, mono=True)
  s.text(x0 + W / 2, y0 + 38, "arithmetic intensity (FLOP per byte of DRAM traffic)", size=12, anchor="middle", color=MUTED)
  # H100 roof: 3.35 TB/s, 989 TF bf16 tensor, 67 TF fp32
  bw, peak_tc, peak_fp32 = 3.35e12, 989e12, 67e12
  for peak, col, lab, dy in [(peak_tc, "#2a9d9d", "bf16 tensor cores ≈ 989 TFLOP/s", -11), (peak_fp32, "#e08a2c", "fp32 CUDA cores ≈ 67 TFLOP/s", 13)]:
    ridge = peak / bw
    s.line(X(0.01), Y(bw * 0.01), X(ridge), Y(peak), col, 2.2)
    s.line(X(ridge), Y(peak), X(10000), Y(peak), col, 2.2)
    s.text(X(10000) - 4, Y(peak) + dy, lab, size=11, anchor="end", color=col, weight=600)
  s.text(X(0.012), Y(bw * 6), "slope: HBM3 bandwidth ≈ 3.35 TB/s", size=11, color=MUTED)
  s.text(x0 - 46, y0 - H - 14, "FLOP/s", size=11, color=MUTED)
  # (AI, fraction of the roof achieved, label, anchor, dx, dy): example kernels, drawn below their roof
  for ai, frac, lab, anchor, dx, dy in [(1 / 12, 0.85, "a + b  (f32)", "start", 10, 8), (0.6, 0.8, "softmax", "start", 10, 8),
                                        (1365, 0.55, "gemm 4096 bf16", "start", 10, 4), (200, 0.3, "flash attention", "start", 10, 4)]:
    f = min(bw * ai, peak_tc) * frac
    s.circle(X(ai), Y(f), 5, "red")
    s.text(X(ai) + dx, Y(f) + dy, lab, size=11, color=C["red"][2], anchor=anchor)
  s.code(700, 110, 260, ["memory-bound: left of the ridge", "  → fuse, don't move bytes", "", "compute-bound: right of it", "  → tile, use tensor cores", "", "ridge = peak / bandwidth", "  ≈ 295 FLOP/B (bf16 TC)"], size=11, title="reading it", color="teal")
  return s

@fig("week07/figures/tiling.svg")
def tiling():
  s = SVG(980, 420, "Tiling: make the working set fit", "compute a T×T block of C from T-wide strips of A and B, so each loaded element is reused T times")
  def mat(x, y, n, cell, color, hl, label):
    f, st, t = C[color]
    for r in range(n):
      for c in range(n):
        on = hl(r, c)
        s.rect(x + c * cell, y + r * cell, cell, cell, f if on else "#fafbfc", st if on else "#d5dae2", 0, sw=1)
    s.text(x + n * cell / 2, y - 12, label, size=13, anchor="middle", weight=700, color=t)
  mat(60, 150, 8, 22, "blue", lambda r, c: 2 <= r < 4, "A")
  mat(300, 150, 8, 22, "green", lambda r, c: 4 <= c < 6, "B")
  mat(540, 150, 8, 22, "orange", lambda r, c: 2 <= r < 4 and 4 <= c < 6, "C")
  s.text(250, 238, "×", size=22, anchor="middle"); s.text(490, 238, "=", size=22, anchor="middle")
  s.code(760, 110, 200, ["for i0 in 0..N step T:", "  for j0 in 0..N step T:", "    for k in 0..N:", "      for i in i0..i0+T:", "        for j in j0..j0+T:", "          C[i,j] +=", "            A[i,k]*B[k,j]"], size=11, title="split + reorder", color="teal")
  s.text(450, 360, "in uc these are rewrites on ranges: split r → (r_outer·T + r_inner), reorder, then", size=12, anchor="middle", color=MUTED)
  s.text(450, 382, "upcast the innermost to registers / SIMD lanes. The symbolic simplifier cleans up the indices.", size=12, anchor="middle", color=MUTED)
  return s

# ============================================================ week 8
@fig("week08/figures/h100.svg")
def h100():
  s = SVG(980, 470, "H100 (SXM) from 10,000 feet", "132 SMs · 50 MB L2 · 80 GB HBM3 at ~3.35 TB/s.  (PCIe: 114 SMs, ~2 TB/s)")
  s.box(30, 100, 560, 300, "", "gray")
  s.text(50, 120, "GPU", size=12, weight=700, color=MUTED)
  for r in range(3):
    for c in range(6):
      s.box(50 + c * 88, 135 + r * 56, 78, 46, "SM", "teal", size=11)
  s.text(310, 318, "… 132 SMs", size=12, anchor="middle", color=MUTED)
  s.box(50, 335, 520, 30, "L2 cache  50 MB  (shared by all SMs)", "orange", size=12)
  s.box(30, 415, 560, 36, "HBM3  80 GB  ·  ~3.35 TB/s", "red", size=12)
  # zoom SM
  s.box(630, 100, 320, 350, "", "teal")
  s.text(650, 120, "one SM", size=12, weight=700, color=C["teal"][2])
  for q in range(4):
    x = 645 + (q % 2) * 150; y = 135 + (q // 2) * 105
    s.box(x, y, 140, 95, "", "white")
    s.text(x + 70, y + 16, f"sub-partition {q}", size=10, anchor="middle", color=MUTED)
    s.text(x + 70, y + 36, "warp scheduler", size=10, anchor="middle")
    s.text(x + 70, y + 54, "32 fp32 lanes", size=10, anchor="middle")
    s.text(x + 70, y + 72, "tensor core", size=10, anchor="middle", color=C["teal"][2], weight=600)
  s.box(645, 350, 290, 40, "registers 256 KB", "blue", size=12)
  s.box(645, 398, 290, 40, "shared memory / L1  228 KB", "green", size=12)
  s.rect(490, 135, 78, 46, "none", C["teal"][2], 8, sw=2.2)                       # the SM being magnified
  s.line(568, 135, 630, 100, LINE, dash="3 3"); s.line(568, 181, 630, 450, LINE, dash="3 3")
  return s

@fig("week08/figures/axis_map.svg")
def axis_map():
  s = SVG(980, 420, "Mapping ranges onto the grid", "out[i, j] for a 1024×1024 output: split each range, bind the pieces to hardware indices")
  s.code(30, 100, 400, ["%i = range 1024", "%j = range 1024", "", "# opt actions (week 7 style):", "split i → i_blk·16 + i_thr", "split j → j_blk·16 + j_thr", "bind  i_blk → blockIdx.y", "bind  j_blk → blockIdx.x", "bind  i_thr → threadIdx.y", "bind  j_thr → threadIdx.x"], size=12, title="kernel ranges", color="orange", hl={6: "teal", 7: "teal", 8: "blue", 9: "blue"})
  # grid
  for r in range(4):
    for c in range(4):
      s.rect(480 + c * 52, 110 + r * 52, 52, 52, "#e2f4f4" if (r, c) != (1, 2) else "#2a9d9d", "#2a9d9d", 0, sw=1)
  s.text(584, 98, "grid: 64 × 64 blocks", size=12, anchor="middle", weight=700, color=C["teal"][2])
  s.line(480 + 3 * 52, 110 + 52, 760, 110, LINE, dash="3 3"); s.line(480 + 3 * 52, 110 + 104, 760, 300, LINE, dash="3 3")
  for r in range(8):
    for c in range(8):
      s.rect(760 + c * 23.75, 110 + r * 23.75, 23.75, 23.75, "#e8effc", "#3b6fd8", 0, sw=0.8)
  s.text(855, 98, "block: 16 × 16 threads", size=12, anchor="middle", weight=700, color=C["blue"][2])
  s.code(480, 330, 470, ["int i = blockIdx.y*16 + threadIdx.y;", "int j = blockIdx.x*16 + threadIdx.x;", "out[i*1024 + j] = ...;"], size=12, color="blue")
  return s

@fig("week08/figures/coalescing.svg")
def coalescing():
  s = SVG(980, 400, "Coalescing: a warp's loads should hit consecutive addresses", "threadIdx.x should index the contiguous (stride-1) axis   (8 of a warp's 32 threads shown)")
  def warp(y, stride, ok, title, note):
    s.text(40, y - 16, title, size=13, weight=700, color=C["green" if ok else "red"][2])
    cx = lambda a: 60 + a * 34 + 17
    for a in range(24):
      hit = a % stride == 0 and a // stride < 8
      s.rect(60 + a * 34, y + 56, 34, 26, ("#e5f5ec" if ok else "#fbe9e9") if hit else "#fafbfc", "#c9ced6", 0, sw=1)
      if a % 8 == 0: s.text(60 + a * 34 + 2, y + 94, f"{a * 4} B", size=9, color=MUTED, mono=True)
    for t in range(8):
      s.circle(cx(t), y + 14, 11, "blue", f"t{t}", size=9)
      s.arrow(cx(t), y + 26, cx(t * stride), y + 55, color="#7b8494", sw=1.2)
    s.text(940, y + 14, note, size=11, anchor="end", color=MUTED)
  warp(110, 1, True, "coalesced: thread t reads a[t]", "8 × 4 B = one 32 B sector")
  warp(250, 3, False, "strided: thread t reads a[3t]  (threadIdx.x bound to the wrong axis)", "3 sectors fetched, ⅔ wasted")
  s.text(490, 380, "fix by choosing the axis binding (week 8) or by staging the tile through shared memory (week 9)", size=12, anchor="middle", color=MUTED)
  return s
