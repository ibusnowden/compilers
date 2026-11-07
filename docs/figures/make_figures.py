#!/usr/bin/env python3
"""Regenerate every course figure:   python docs/figures/make_figures.py   [--png]   [name-filter]
Figures are plain functions in figs_*.py returning an SVG; this writes them under docs/."""
import sys, subprocess, importlib, pathlib
HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent
sys.path.insert(0, str(HERE))

def main():
  png = "--png" in sys.argv
  filt = [a for a in sys.argv[1:] if not a.startswith("--")]
  figs = {}
  for mod in sorted(p.stem for p in HERE.glob("figs_*.py")):
    figs.update(importlib.import_module(mod).FIGS)
  for path, fn in figs.items():
    if filt and not any(f in path for f in filt): continue
    out = DOCS / path
    out.parent.mkdir(parents=True, exist_ok=True)
    fn().save(out)
    if png: subprocess.run(["rsvg-convert", "-z", "1.5", str(out), "-o", str(out.with_suffix(".png"))], check=True)
    print("wrote", out.relative_to(DOCS))

if __name__ == "__main__":
  main()
