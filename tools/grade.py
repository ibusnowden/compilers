#!/usr/bin/env python3
"""Cumulative grading: week N must pass weeks 1..N.  Mistakes compound, so regressions count.

  python tools/grade.py 3              # weeks 1-3 against your compiler
  python tools/grade.py 3 --oracle     # check the .uir tests themselves
  python tools/grade.py 8 --perf       # include performance thresholds
"""
import os, sys, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
week = int(args[0]) if args else 1
env = dict(os.environ)
if "--oracle" in sys.argv: env["UC_ORACLE"] = "1"
dirs = [str(ROOT / "tests" / f"week{w:02d}") for w in range(1, week + 1) if (ROOT / "tests" / f"week{w:02d}").exists()]
mark = [] if "--perf" in sys.argv else ["-m", "not perf"]
if week >= 8 and "--oracle" not in sys.argv:   # the GPU must pass everything the CPU passes
  r1 = subprocess.call([sys.executable, "-m", "pytest", "-q", *mark, *dirs], env=env)
  env2 = dict(env, UC_DEVICE="CUDA")
  r2 = subprocess.call([sys.executable, "-m", "pytest", "-q", *mark, str(ROOT / "tests" / "week06")], env=env2)
  sys.exit(r1 or r2)
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", *mark, *dirs], env=env))
