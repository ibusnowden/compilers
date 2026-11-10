# Course map

Where everything is, what you write each week, and how to work through it.

## The map

Each row is one week. Read left to right: lecture, quiz, then the assignment (your tests first, then the compiler).

| week | read | quiz | assignment | you write (in `compiler/uc/`) | your tests | done when this passes |
|---|---|---|---|---|---|---|
| 1 | [UOps & rewrites](docs/week01/lecture.md) | [quiz](docs/week01/quiz.md) | [hw](docs/week01/assignment.md) | `uop.py` `symbolic.py` `parse.py` `uir.py` `lam.py` | 15 `.uir` | `pytest tests/week01` |
| 2 | [Loads, stores & memory](docs/week02/lecture.md) | [quiz](docs/week02/quiz.md) | [hw](docs/week02/assignment.md) | `linearize.py` `interp.py` | 12 `.uir` | `pytest tests/week{01..02}` |
| 3 | [Loops, rendering & runtime](docs/week03/lecture.md) | [quiz](docs/week03/quiz.md) | [hw](docs/week03/assignment.md) | `renderer_c.py` `runtime/cpu.py` | 12 `.uir` | `pytest tests/week{01..03}` |
| 4 | [Rangeify, reduce, materialization](docs/week04/lecture.md) | [quiz](docs/week04/quiz.md) | [hw](docs/week04/assignment.md) | `rangeify.py` | 12 `.uir` | `pytest tests/week{01..04}` |
| 5 | [Movement ops](docs/week05/lecture.md) | [quiz](docs/week05/quiz.md) | [hw](docs/week05/assignment.md) | `rangeify.py` (movement ops) | 12 `.uir` | `pytest tests/week{01..05}` |
| 6 | [Autodiff & speed](docs/week06/lecture.md) | [quiz](docs/week06/quiz.md) | [hw](docs/week06/assignment.md) | `tensor.py` (new) `autodiff.py` `nn/` `optim.py` `stats.py` `models/{mnist,gpt2,resnet}.py` | 15 `test_*.py` | `pytest tests/week{01..06}` |
| 7 | [Memory hierarchies, threading](docs/week07/lecture.md) | [quiz](docs/week07/quiz.md) | [hw](docs/week07/assignment.md) | `opt.py` | 10 `test_*.py` | `pytest tests/week{01..07}` + perf |
| 8 | [GPUs](docs/week08/lecture.md) | [quiz](docs/week08/quiz.md) | [hw](docs/week08/assignment.md) | `runtime/cuda.py` `renderer_cuda.py` | 10 `test_*.py` | `pytest tests/week{01..08}` + week 6 on GPU + perf |
| 9 | [Tensor cores, GEMM, flash attention](docs/week09/lecture.md) | [quiz](docs/week09/quiz.md) | [hw](docs/week09/assignment.md) | `opt.py` `renderer_cuda.py` (f16, bf16, MMA) | 10 `test_*.py` | `pytest tests/week{01..09}` + perf |
| 10 | [Multiple GPUs & sharding](docs/week10/lecture.md) | [quiz](docs/week10/quiz.md) | [hw](docs/week10/assignment.md) | `multi.py` | 8 `test_*.py` | `pytest tests/week{01..10}` |
| 11 | [SOTA inference & training](docs/week11/lecture.md) | [quiz](docs/week11/quiz.md) | [hw](docs/week11/assignment.md) | `models/llama.py` `jit.py` | optional | `pytest tests/week{01..11}` + [benchmarks](benchmarks/README.md) |
| 12–14 | [Final project guide](docs/week12/final-project.md) | | proposal, code, report | your extension, behind a flag | your own | every earlier week still passes |

Your tests go in `tests/weekNN/mine/`. The exact "done when" command, including the perf and GPU parts, is the first
line of each assignment.

## Where things are

| you need | look in |
|---|---|
| what a UIR op means | [docs/uir-spec.md](docs/uir-spec.md) |
| what a `Tensor` method must do | [docs/tensor-api.md](docs/tensor-api.md) |
| to check a `.uir` test you wrote (weeks 1–5) | `UC_ORACLE=1 pytest tests/weekNN/mine` |
| to check a Python test or your model code (weeks 6+) | `UC_TORCH_SHIM=1 pytest tests/weekNN/mine` |
| data and weights | `python tools/download.py mnist gpt2 resnet18 qwen fineweb` |
| reference models the tests compare against | `tests/ref/` |
| to change a figure | `docs/figures/figs_*.py`, then `python docs/figures/make_figures.py` |
| the reading list, ground rules, hardware per week | [SYLLABUS.md](SYLLABUS.md) |

What you need installed: Python only for weeks 1–2, `clang` from week 3, an NVIDIA GPU from week 8.

## How to work through each week

1. **Read the three lectures actively.** Type out the code sketches rather than copying them. Predict what each
   figure shows before you read its caption.
2. **Take the quiz closed-book.** Write your answer down before you unfold the given one. Re-read the section for
   anything you miss, now: the course is cumulative, so a gap comes back weeks later as a bug.
3. **Write your own tests first**, in `tests/weekNN/mine/`, and check them with the oracle (weeks 1–5) or the shim
   (weeks 6+). This is where you find out whether you understood the semantics. The oracle only checks that a `prove`
   is *true* and an index is *in bounds*, so check by hand that the
   [required rules](docs/uir-spec.md#the-required-simplification-rules-for-prove) can derive them.
4. **Implement until the week's command passes.** Always rerun every earlier week, not just the current one.
5. **Refactor before moving on.** Later weeks build directly on this code, so mess compounds.

## Getting the most out of it

* **Struggle before asking.** Give a bug a real attempt (30–45 minutes) before you ask for help. Then ask for a hint
  or a review, not the code.
* **Don't read tinygrad's source until your version works.** The lectures point at it as the production version of the
  same ideas. It's most useful as a comparison afterwards.
* **Keep `REPORT.md` as a lab notebook:** what broke, why, and what you'd do differently. It becomes your final
  report, and your map when you extend the project.
* **Give weeks 1–2 extra time.** Every later week uses the hash-consed graph, the rewrite engine and the simplifier.
* **Do week 1's "fuzz your simplifier" stretch.** It's the cheapest way to catch unsound rules, which otherwise show
  up as silent wrong answers in week 5.
* **Build the debugging tools when the lectures suggest them** (`UOp.render()`, `DEBUG=1`, `DEBUG=2`). You'll use
  them every week.
* **Suspect yourself first, then the test.** The public tests pass the oracle and the shim. If one still looks
  impossible to pass soundly, work out by hand which required rule would have to derive it.
