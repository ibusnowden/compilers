# Syllabus

## Format

The course follows a university semester, but it's built for self-study.

* **Lectures:** three per week (`docs/weekNN/lecture.md`, sections N.1–N.3). Each takes 60–90 minutes to read
  and work through, with diagrams, code, and pointers to primary sources.
* **Quizzes:** one per week (`docs/weekNN/quiz.md`). Take it *after* the lectures and *before* the assignment.
  Answers are folded under each question. Score yourself honestly. Anything you miss tends to resurface later as a bug.
* **Semester-long compiler project:** each week has two deadlines:
  1. **Tests due** (end of lecture 3): write your own tests for the week's features in `tests/weekNN/mine/`
     and check them against the oracle or the shim. Writing tests first forces you to understand the semantics
     before you commit to an implementation.
  2. **Compiler due** (two days later): `python tools/grade.py N` passes, **cumulatively**. Week N runs every test
     from weeks 1 to N, including your own tests from earlier weeks.
* **Final project** (weeks 12–14): extend `uc`. Implement a paper, port it to unusual hardware, or pursue your
  own idea. See [the guide](docs/week12/final-project.md).

### A suggested week (≈ 10–14 hours)

| day | work |
|---|---|
| 1 | Lecture 1. Read the assignment. Sketch the data structures |
| 2 | Lecture 2. Write tests (`tests/weekNN/mine/`) and run them under the oracle or shim |
| 3 | Lecture 3. Take the quiz. **Tests due.** Start the implementation |
| 4–5 | Implement. Run the cumulative grader often. Refactor before moving on. **Compiler due** |

## Grading (if you want numbers)

| component | weight |
|---|---|
| Weekly compiler (11 × cumulative public tests + your tests) | 50% |
| Weekly tests you wrote (coverage of the spec; would they catch a wrong compiler?) | 10% |
| Quizzes | 10% |
| Final project: code, results, report | 30% |

A week's compiler grade is the fraction of tests passing on weeks 1..N at the deadline. **Performance tests**
(`-m perf`, weeks 7–9) count when you run them on quiet hardware and paste the output into `REPORT.md`.

## Policies

* **Simplicity is a requirement.** Line counts are a soft signal. If one week's diff adds more than ~1500 lines,
  stop and ask what can be deleted. The reference design's whole compiler stays in the low thousands of lines.
* **Pure Python + C/CUDA source.** `uc` may import numpy (for host-side arrays and tests only) and `ctypes`.
  It must not import torch, triton, cupy, jax, or tvm. You call clang and NVRTC yourself.
  PyTorch appears only in the tests, as a reference.
* **Regressions count.** The grader is cumulative on purpose.
* **Using AI assistants.** Allowed for explanations and reviews. Every line you commit must be a line you can
  explain. The quizzes and final report will find out if you can't.

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
