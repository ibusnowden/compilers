"""Course overview + weeks 1-3 figures."""
from svg import SVG, C, MUTED, INK
FIGS = {}
def fig(path):
  def deco(fn): FIGS[path] = fn; return fn
  return deco

# ============================================================ course
@fig("figures/roadmap.svg")
def roadmap():
  s = SVG(980, 670, "The semester as a stack", "each week is built on top of everything below it; nothing is thrown away")
  weeks = [
    ("1", "UOps & rewrites", "hash-consing · pattern matching · symbolic · bounds · λ-calculus", "purple"),
    ("2", "Loads, stores & memory", "versioned buffers · write-after-read · linearize", "green"),
    ("3", "Loops, rendering & runtime", "range / end · C codegen · clang + ctypes", "orange"),
    ("4", "Rangeify, reduce, materialize", "shapes → loops · accumulators · kernel boundaries", "orange"),
    ("5", "Movement ops", "reshape · permute · pad · shrink · flip as index math", "orange"),
    ("6", "Autodiff & speed", "Tensor API · reverse mode · safetensors · MNIST, GPT-2, ResNet", "blue"),
    ("7", "CPU memory hierarchy", "tiling · upcasting · threads · roofline · search", "teal"),
    ("8", "GPUs", "SIMT · axis mapping · coalescing · NVRTC + driver API", "teal"),
    ("9", "GPUs part 2", "tensor cores · fast GEMM · implicit-GEMM conv · flash attention", "teal"),
    ("10", "Multiple GPUs", "sharding · allreduce · data & tensor parallel", "yellow"),
    ("11", "SOTA inference & training", "Llama/Qwen decode · GPT-2 124M training · JIT", "yellow"),
    ("12–14", "Final project", "implement a paper · port to new hardware · your idea", "red"),
  ]
  y0, h, gap = 80, 40, 6
  for i, (w, name, sub, col) in enumerate(reversed(weeks)):
    y = y0 + i * (h + gap)
    indent = (len(weeks) - 1 - i) * 10
    f, st, t = C[col]
    s.rect(60 + indent, y, 860 - 2 * indent, h, f, st, 8)
    s.text(82 + indent, y + h / 2, f"week {w}", size=12, weight=700, color=t)
    s.text(165 + indent, y + h / 2, name, size=14, weight=600, color=INK)
    s.text(480, y + h / 2, sub, size=12, color=MUTED)
  s.text(490, 650, "ground floor: a hash-consed graph and a rewrite engine", size=12, anchor="middle", color=MUTED, italic=True)
  return s

@fig("figures/pipeline.svg")
def pipeline():
  s = SVG(1000, 370, "What uc looks like by week 11", "every arrow is a graph_rewrite or a lowering you write")
  stages = [("Tensor API", "x @ w + b", "blue", "6"), ("UOp graph", "lazy, hash-consed", "purple", "1"),
            ("autodiff", "grad rewrites", "blue", "6"), ("rangeify", "shapes → ranges", "orange", "4–5"),
            ("kernel opts", "tile · upcast · map", "teal", "7–9"), ("linearize", "graph → list", "green", "2–3"),
            ("render", "C · CUDA", "orange", "3, 8"), ("runtime", "clang · NVRTC", "gray", "3, 8")]
  x, y, w, h = 30, 112, 105, 74
  for i, (name, sub, col, wk) in enumerate(stages):
    xx = x + i * 120
    s.box(xx, y, w, h, name, col, size=13, weight=600, sub=sub)
    s.text(xx + w / 2, y - 12, f"week {wk}", size=11, anchor="middle", color=MUTED)
    if i: s.arrow(xx - 15, y + h / 2, xx - 1, y + h / 2)
  # the simplifier sits underneath: one dashed arrow per stage that calls it
  s.box(150, 232, 700, 44, "symbolic simplifier + bounds (week 1): every graph stage calls it", "purple", size=13)
  for i in range(1, 6):
    cx = x + i * 120 + w / 2
    s.arrow(cx, 232, cx, y + h + 2, dash="4 3", color="#7c5cc4")
  s.box(150, 296, 700, 40, "tests: .uir files (weeks 1–5) · Tensor-API pytest (weeks 6–11) · cumulative every week", "gray", size=12)
  return s

# ============================================================ week 1
@fig("week01/figures/tree_vs_dag.svg")
def tree_vs_dag():
  s = SVG(980, 400, "Trees duplicate, DAGs share", "a = 1;  a1 = a + a;  a2 = a1 + a1;  a3 = a2 + a2")
  # tree
  s.text(230, 92, "as a tree: 2ⁿ nodes", size=13, anchor="middle", weight=600, color=MUTED)
  def tree(cx, cy, depth, spread):
    if depth == 0: s.node(cx, cy, "1", "gray", w=30); return
    s.node(cx, cy, "+", "blue", w=34)
    for d in (-1, 1):
      nx, ny = cx + d * spread, cy + 70
      s.line(cx, cy + 15, nx, ny - 15)
      tree(nx, ny, depth - 1, spread / 2)
  tree(230, 125, 3, 104)
  # dag
  s.text(610, 92, "hash-consed DAG: n + 1 nodes", size=13, anchor="middle", weight=600, color=MUTED)
  ys = [125, 195, 265, 335]
  labels = ["a3 = +", "a2 = +", "a1 = +", "a = 1"]
  for i, (yy, lab) in enumerate(zip(ys, labels)):
    if i < 3:     # both operands are the same node: two edges to one child
      s.curve(592, yy + 15, 570, (yy + ys[i + 1]) / 2, 592, ys[i + 1] - 15, head=False)
      s.curve(628, yy + 15, 650, (yy + ys[i + 1]) / 2, 628, ys[i + 1] - 15, head=False)
    s.node(610, yy, lab, "blue" if i < 3 else "gray", w=90)
  s.text(700, 205, "UOp(ADD, i32, (a1, a1))\n  is UOp(ADD, i32, (a1, a1))", size=12, mono=True, color=C["purple"][2])
  s.text(700, 256, "building the same node twice\nreturns the same object", size=12, color=MUTED)
  s.line(470, 100, 470, 370, color="#e3e6eb")
  return s

@fig("week01/figures/rewrite.svg")
def rewrite():
  s = SVG(980, 380, "graph_rewrite: bottom-up to a fixed point", "x·2 + x·3   →   x·5      (sound mod 2³², so no bounds are needed)")
  # before
  def g(cx, top, lab):
    s.text(cx, top - 4, lab, size=12, anchor="middle", color=MUTED, weight=600)
  g(150, 110, "input")
  s.node(150, 140, "ADD", "blue"); s.node(90, 210, "MUL", "blue"); s.node(210, 210, "MUL", "blue")
  s.node(50, 290, "2", "gray", w=36); s.node(150, 290, "x", "purple", w=36); s.node(250, 290, "3", "gray", w=36)
  for a, b in [((150, 155), (90, 195)), ((150, 155), (210, 195)), ((90, 225), (50, 275)), ((90, 225), (142, 275)), ((210, 225), (158, 275)), ((210, 225), (250, 275))]:
    s.line(*a, *b)
  s.text(150, 330, "x is ONE node, used by both MULs\n(hash-consing)", size=11, anchor="middle", color=MUTED)
  s.arrow(285, 210, 345, 210, label="match", color="#7c5cc4")
  # pattern
  s.box(355, 120, 260, 180, "", "purple")
  s.text(485, 140, "pattern  →  rewrite", size=12, anchor="middle", weight=700, color=C["purple"][2])
  s.text(485, 180, "linear form of the ADD:", size=11, anchor="middle", color=MUTED)
  s.text(485, 202, "{x: 2} + {x: 3}", size=13, anchor="middle", mono=True)
  s.text(485, 228, "collect like terms", size=11, anchor="middle", color=MUTED)
  s.text(485, 250, "{x: 5}", size=13, anchor="middle", mono=True)
  s.text(485, 278, "rebuild canonically", size=11, anchor="middle", color=MUTED)
  s.arrow(625, 210, 685, 210, label="rebuild", color="#7c5cc4")
  g(800, 110, "output (a fixed point)")
  s.node(800, 175, "MUL", "blue"); s.node(760, 250, "x", "purple", w=36); s.node(840, 250, "5", "gray", w=36)
  s.line(800, 190, 760, 235); s.line(800, 190, 840, 235)
  s.text(800, 310, "rewriting MUL(x,5) again\nproduces MUL(x,5): stop", size=11, anchor="middle", color=MUTED)
  return s

@fig("week01/figures/intervals.svg")
def intervals():
  s = SVG(980, 390, "Bounds flow up the graph", "x ∈ [0, 100], y ∈ [0, 3]:   (4x + y) / 4   simplifies to   x")
  s.node(170, 330, "x", "purple", w=40); s.node(330, 330, "y", "purple", w=40); s.node(80, 330, "4", "gray", w=36)
  s.node(125, 250, "MUL", "blue"); s.node(230, 175, "ADD", "blue"); s.node(330, 100, "DIV", "blue"); s.node(430, 175, "4", "gray", w=36)
  for a, b in [((125, 265), (80, 315)), ((125, 265), (170, 315)), ((230, 190), (125, 235)), ((230, 190), (330, 315)), ((330, 115), (230, 160)), ((330, 115), (430, 160))]:
    s.line(*a, *b)
  tag = lambda x, y, t, c="purple": s.text(x, y, t, size=12, mono=True, color=C[c][2], weight=600)
  tag(196, 330, "[0, 100]"); tag(356, 330, "[0, 3]"); tag(150, 250, "[0, 400]"); tag(255, 175, "[0, 403]"); tag(355, 100, "[0, 100]")
  s.code(520, 92, 430, [
    "rule 5:  e = a·c + b,  c = 4",
    "  a = x,   b = y",
    "  vmin(e) = 0 ≥ 0  ✓   vmin(b) = 0 ≥ 0  ✓",
    "  e cannot overflow:  403 < 2³¹  ✓",
    "  e / 4  →  x + y / 4",
    "",
    "interval rule:  y ∈ [0, 3],  3 < 4",
    "  y / 4  →  0",
    "  x + 0  →  x",
  ], size=12, title="why it is allowed", color="purple", hl={4: "purple", 8: "purple"})
  s.text(735, 326, "intervals alone give DIV ∈ [0, 100];  rule 5 proves it is exactly x", size=12, anchor="middle", color=C["purple"][2])
  s.text(490, 366, "if x could be negative, truncating division breaks rule 5:  x = −1 gives (−4 + 3) / 4 = 0, not −1", size=12, anchor="middle", color=C["red"][2])
  return s

@fig("week01/figures/beta.svg")
def beta():
  s = SVG(980, 340, "λ-calculus on UOps (de Bruijn indices)", "names disappear, so α-equivalent terms are the same node:  λx.x  is  λy.y")
  s.code(30, 92, 410, [
    "K = λx. λy. x        →   λ. λ. #1",
    "I = λx. x            →   λ. #0",
    "Ω = (λx. x x)(λx. x x)",
    "",
    "normal order: reduce the leftmost-",
    "outermost redex first",
    "",
    "K I Ω  →  (λ. I) Ω  →  I        ✓",
    "applicative order evaluates Ω first: ∞",
  ], size=12, title="terms", color="purple", hl={7: "green", 8: "red"})
  # tree for (λ.λ.#1) I
  s.node(560, 120, "APP", "blue"); s.node(500, 190, "LAM", "purple"); s.node(500, 255, "LAM", "purple"); s.node(500, 320 - 10, "#1", "gray", w=40)
  s.node(625, 190, "LAM", "purple"); s.node(625, 255, "#0", "gray", w=40)
  for a, b in [((560, 135), (500, 175)), ((560, 135), (625, 175)), ((500, 205), (500, 240)), ((500, 270), (500, 295)), ((625, 205), (625, 240))]:
    s.line(*a, *b)
  s.arrow(680, 200, 760, 200, label="β (shift, subst, shift)", color="#7c5cc4", loff=(0, -12))
  s.node(840, 160, "LAM", "purple"); s.node(840, 225, "LAM", "purple"); s.node(840, 290, "#0", "gray", w=40)
  s.line(840, 175, 840, 210); s.line(840, 240, 840, 275)
  s.text(910, 225, "λ. I", size=13, mono=True, color=C["purple"][2])
  return s

# ============================================================ week 2
@fig("week02/figures/versions.svg")
def versions():
  s = SVG(980, 360, "Memory is versioned", "store returns a NEW version; loads name the version they read")
  xs = [90, 300, 510, 720]
  labs = ["b0 = buffer i32 4", "b1 = store b0 0 1", "b2 = store b1 0 2", "b3 = store b2 1 7"]
  for i, (x, lab) in enumerate(zip(xs, labs)):
    s.box(x - 85, 110, 170, 44, lab, "green", size=12, mono=True)
    if i: s.arrow(xs[i - 1] + 85, 132, x - 85, 132)
  loads = [(300, "load b1 0  → 1"), (510, "load b2 0  → 2"), (90, "load b0 0  → 0")]
  for x, lab in loads:
    s.box(x - 80, 215, 160, 36, lab, "blue", size=12, mono=True)
    s.arrow(x, 210, x, 156, color="#2e9e6a")
  s.rect(60, 290, 860, 50, "#fbfbfd", "#e3e6eb", 8)
  s.text(490, 305, "Physically there is ONE allocation, mutated in place.", size=13, anchor="middle", weight=600)
  s.text(490, 326, "So every load of version v must run before the store that overwrites v (write-after-read).", size=12, anchor="middle", color=MUTED)
  return s

@fig("week02/figures/war.svg")
def war():
  s = SVG(980, 420, "Scheduling around write-after-read", "b1 = store b0 0 1;   b2 = store b1 0 2;   x = load b1 0")
  s.text(240, 92, "valid order", size=13, anchor="middle", weight=700, color=C["green"][2])
  s.text(740, 92, "invalid order", size=13, anchor="middle", weight=700, color=C["red"][2])
  ok = ["1  store [0] ← 1     (b1)", "2  x = load [0]      (reads b1 = 1)", "3  store [0] ← 2     (b2)"]
  bad = ["1  store [0] ← 1     (b1)", "2  store [0] ← 2     (b2)", "3  x = load [0]      (reads 2 ✗)"]
  s.code(60, 106, 360, ok, size=12, color="green", hl={1: "green"})
  s.code(560, 106, 360, bad, size=12, color="red", hl={2: "red"})
  s.text(490, 245, "When no valid order exists, copy.", size=14, anchor="middle", weight=700)
  s.box(80, 275, 190, 40, "b0", "green", mono=True)
  s.box(330, 260, 220, 36, "x = store b0 0 100", "green", mono=True, size=12)
  s.box(330, 310, 220, 36, "y = store b0 1 200", "green", mono=True, size=12)
  s.arrow(270, 290, 330, 278); s.arrow(270, 300, 330, 328)
  s.text(640, 285, "two futures of one version:", size=12, color=MUTED)
  s.text(640, 305, "one in-place buffer cannot hold both", size=12, color=MUTED)
  s.box(640, 330, 290, 36, "fix: y = store (copy b0) 1 200", "purple", mono=True, size=12)
  return s

@fig("week02/figures/linearize.svg")
def linearize():
  s = SVG(980, 380, "Linearize: graph → ordered list", "b1 = store b0 0 1;  t = load b1 0;  b2 = store b1 0 2;  assert t 1")
  s.node(120, 310, "BUFFER b0", "green", w=110); s.node(120, 235, "STORE b1", "green", w=100)
  s.node(60, 160, "LOAD b1", "blue", w=90); s.node(190, 160, "STORE b2", "green", w=100); s.node(120, 90, "ASSERT", "gray", w=90)
  s.arrow(120, 295, 120, 251); s.arrow(110, 220, 70, 176); s.arrow(135, 220, 180, 176); s.arrow(65, 145, 105, 106); s.arrow(185, 145, 140, 106)
  s.arrow(105, 160, 140, 160, color="#d64545", dash="4 3")
  s.text(122, 140, "WAR", size=10, color=C["red"][2], anchor="middle", weight=700)
  s.arrow(265, 200, 340, 200, label="toposort", color="#2e9e6a")
  s.code(350, 100, 330, ["0  b0 = BUFFER i32 2", "1  STORE b0[0] ← 1      # b1", "2  t = LOAD b0[0]       # reads b1", "3  STORE b0[0] ← 2      # b2", "4  ASSERT t == 1"], size=12, title="linear program", color="green")
  s.code(700, 100, 250, ["mem = [0, 0]", "mem = [1, 0]", "t = 1", "mem = [2, 0]", "1 == 1  ✓"], size=12, title="interpreter state", color="gray")
  s.text(490, 300, "the red edge is not data: it says “read b1 before b1 is overwritten”", size=12, anchor="middle", color=C["red"][2])
  return s

# ============================================================ week 3
@fig("week03/figures/range_end.svg")
def range_end():
  s = SVG(980, 340, "range and end become a loop", "the chain closed by `end` is the loop body; everything else is placed as far out as it can go")
  s.code(30, 92, 370, ["%k = buffer i32 [7]", "%o = buffer i32 16", "%i = range 16", "%v = add %i (mul (load %k 0) 3)", "%s = store %o %i %v", "%o1 = end %s %i"], size=12, title="UIR", color="orange", hl={2: "orange", 5: "orange"})
  s.arrow(410, 180, 470, 180, label="linearize\n+ render", color="#e08a2c", lsize=10, loff=(0, -22))
  s.code(480, 92, 470, ["void kernel(int* restrict k, int* restrict o) {", "  int c = k[0] * 3;            // hoisted: no %i", "  for (int i = 0; i < 16; i++) {", "    o[i] = i + c;", "  }", "}"], size=12, title="C", color="orange", hl={1: "purple", 2: "orange", 3: "green"})
  s.legend(480, 262, [("purple", "loop-invariant: depends on no range → outside"), ("orange", "the range opens a loop; end closes it"), ("green", "the store chain: runs every iteration")])
  return s

@fig("week03/figures/matmul_space.svg")
def matmul_space():
  s = SVG(980, 460, "Matmul is a 3-D iteration space", "C[i,j] += A[i,k] · B[k,j]:  i, j index the output; k is reduced")
  # oblique projection: j → right, k → up, i → depth (up-right)
  ox, oy, a, d = 110, 375, 48, 26
  def P(i, j, k): return (ox + j * a + i * d, oy - k * a - i * d * 0.8)
  n = 4
  for i in range(n):          # top face (k = n)
    for j in range(n):
      s.poly([P(i, j, n), P(i, j + 1, n), P(i + 1, j + 1, n), P(i + 1, j, n)], "#fbe8d2", "#e08a2c", 0.9)
  for i in range(n):          # right face (j = n)
    for k in range(n):
      s.poly([P(i, n, k), P(i + 1, n, k), P(i + 1, n, k + 1), P(i, n, k + 1)], "#f5dcbd", "#e08a2c", 0.9)
  for j in range(n):          # front face (i = 0): one (j, k) slab per output column
    for k in range(n):
      s.poly([P(0, j, k), P(0, j + 1, k), P(0, j + 1, k + 1), P(0, j, k + 1)], "#fdf0e1", "#e08a2c", 0.9)
  s.poly([P(0, 1, 0), P(0, 2, 0), P(0, 2, n), P(0, 1, n)], "#e8effc", "#3b6fd8", 1.6, opacity=0.85)
  s.text(P(0, 2, 0)[0], P(0, 2, 0)[1] + 20, "j  (output)", size=12, anchor="middle", color=C["orange"][2], weight=600)
  s.text(P(0, 0, 2)[0] - 14, P(0, 0, 2)[1], "k  (reduce)", size=12, anchor="end", color=C["orange"][2], weight=600)
  s.text(P(2, 4, 0)[0] + 14, P(2, 4, 0)[1] + 4, "i  (output)", size=12, color=C["orange"][2], weight=600)
  s.text(P(0, 2, 0)[0], P(0, 2, 0)[1] + 42, "blue column: all k for one (i, j) = one C[i, j]", size=11, anchor="middle", color=C["blue"][2])
  s.code(530, 92, 420, ["for i in range(M):        # output", "  for j in range(N):      # output", "    for k in range(K):    # reduce", "      C[i*N+j] += A[i*K+k] * B[k*N+j]"], size=12, title="the week-3 loop nest", color="orange")
  s.text(740, 250, "M·N·K multiply-adds,  M·N outputs", size=13, anchor="middle", weight=600)
  s.text(740, 275, "every (i, j) column is independent → parallel", size=12, anchor="middle", color=MUTED)
  s.text(740, 297, "every k step depends on the last → sequential", size=12, anchor="middle", color=MUTED)
  s.text(740, 330, "week 4 moves the += into a register;", size=12, anchor="middle", color=MUTED)
  s.text(740, 350, "weeks 7–9 tile all three axes for caches and tensor cores", size=12, anchor="middle", color=MUTED)
  return s

@fig("week03/figures/runtime.svg")
def runtime():
  s = SVG(980, 300, "From UOps to running code", "the whole CPU runtime is ~60 lines of Python")
  st = [("UOps", "linearized", "purple"), ("C source", "render", "orange"), ("clang", "-O2 -fwrapv\n-shared -fPIC", "gray"),
        (".so", "cached by\nhash(source)", "gray"), ("ctypes", "CDLL + argtypes", "blue"), ("numpy buffers", "pointers in,\nresults out", "green")]
  for i, (a, b, c) in enumerate(st):
    x = 30 + i * 158
    s.box(x, 110, 138, 80, a, c, size=14, weight=600)
    s.text(x + 69, 215, b, size=11, anchor="middle", color=MUTED, mono=i in (2,))
    if i: s.arrow(x - 19, 150, x - 1, 150)
  s.text(490, 265, "asserts compile to `if (!(cond)) fail[0] = line;` and are checked by the host after the call", size=12, anchor="middle", color=MUTED)
  return s
