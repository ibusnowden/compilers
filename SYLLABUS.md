# Syllabus

## Format

Eleven weeks of building one compiler, then a three-week project. It's paced like a semester and built for self-study.

* **Lectures:** three per week (`docs/weekNN/lecture.md`, sections N.1–N.3). Each takes 60–90 minutes to read
  and work through.
* **Quiz:** one per week (`docs/weekNN/quiz.md`). Take it *after* the lectures and *before* the assignment.
  Answers are folded under each question. Anything you miss tends to resurface later as a bug.
* **Assignment** (`docs/weekNN/assignment.md`), always in two steps:
  1. **Tests first:** write your own tests for the week's features in `tests/weekNN/mine/` and check them against
     the oracle or the shim. This forces you to understand the semantics before you commit to an implementation.
  2. **Then the compiler:** make the tests pass, **cumulatively**. In week N, run every test from weeks 1 to N
     (`pytest tests/week{01..NN}`), including the tests you wrote in earlier weeks.
* **Final project** (weeks 12–14): extend `uc`. Implement a paper, port it to unusual hardware, or pursue your
  own idea. See [the guide](docs/week12/final-project.md).

### A suggested week (≈ 10–14 hours)

| day | work |
|---|---|
| 1 | Lecture 1. Read the assignment. Sketch the data structures |
| 2 | Lecture 2. Write tests (`tests/weekNN/mine/`) and run them under the oracle or shim |
| 3 | Lecture 3. Take the quiz. Finish your tests. Start the implementation |
| 4–5 | Implement. Rerun all the weeks so far, often. Refactor before moving on |

## When is a week done?

* `pytest tests/week{01..NN}` passes: every public test and every test of yours, for weeks 1 to N.
* Your own tests pass the oracle or the shim, and would catch a wrong compiler (not just a crashing one).
* You can answer the quiz without looking.
* Weeks 7–9 only: the **performance tests** (`-m perf`) meet their targets on quiet hardware.

Keep a `REPORT.md` at the repo root as a lab notebook (it's gitignored): perf numbers, timings, and the design
decisions the assignments ask you to write down. The final report draws on it.

## Ground rules

* **Simplicity is a requirement.** If one week's diff adds more than ~1500 lines, stop and ask what can be deleted.
  The whole compiler should stay in the low thousands of lines.
* **Pure Python + C/CUDA source.** `uc` may import numpy (for host-side arrays only) and `ctypes`.
  It must not import torch, triton, cupy, jax, or tvm. You call clang and NVRTC yourself.
  PyTorch appears only in the tests, as a reference.
* **Regressions count.** A week isn't done while an earlier week's test fails.
* **AI assistants** are for explanations and reviews, not for writing your compiler. Every line you commit should be
  one you can explain. The quizzes are how you find out.

## Hardware

| weeks | needs |
|---|---|
| 1–2 | Python only |
| 3–7 | Python + clang (any modern x86-64 or ARM64 machine) |
| 8–11 | an NVIDIA GPU with a driver. The text and perf targets assume an **H100** (SXM numbers, with PCIe noted) |
| 10 | one GPU is enough (virtual devices). Rent a multi-GPU node for a few hours if you want real scaling numbers |

## Reading list (by week)

1. Baader & Nipkow, *Term Rewriting and All That*, ch. 1–2 · Filliâtre & Conchon, *Type-Safe Modular Hash-Consing* (2006) ·
   Pierce, *TAPL* ch. 5–6 (λ-calculus, de Bruijn) · Willsey et al., *egg: Fast and Extensible Equality Saturation* (POPL 2021)
2. Click & Paleczny, *A Simple Graph-Based Intermediate Representation* (1995, "sea of nodes") · Novillo, *Memory SSA* (GCC Summit 2007)
3. Kernighan & Ritchie ch. 5 (pointers, arrays) · the clang `-fwrapv` and `restrict` docs · Python `ctypes` docs
4. Ragan-Kelley et al., *Halide* (PLDI 2013) · Chen et al., *TVM* (OSDI 2018) · tinygrad's `rangeify` and scheduler source
5. tinygrad `ShapeTracker` / view merging · NumPy's strided array model (van der Walt et al., 2011)
6. Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018) · the safetensors format spec ·
   Radford et al., *GPT-2* (2019) · He et al., *ResNet* (2016)
7. Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication* (TOMS 2008) · Williams et al., *Roofline* (CACM 2009) ·
   Drepper, *What Every Programmer Should Know About Memory* (2007)
8. NVIDIA CUDA C++ Programming Guide (ch. 5–7) · NVRTC and Driver API docs · NVIDIA H100 architecture whitepaper
9. NVIDIA PTX ISA (mma, ldmatrix, cp.async, wgmma) · Dao et al., *FlashAttention* 1/2/3 (2022/2023/2024) · CUTLASS docs
10. Shoeybi et al., *Megatron-LM* (2019) · Rajbhandari et al., *ZeRO* (2020) · Liu et al., *Ring Attention* (2023) · NCCL docs
11. Touvron et al., *Llama* (2023) / Llama 3 report (2024) · Su et al., *RoFormer (RoPE)* (2021) · Shazeer, *GLU Variants* (2020) ·
    Ainslie et al., *GQA* (2023) · Micikevicius et al., *Mixed Precision Training* (2018) · Karpathy, *llm.c* and *nanoGPT*
