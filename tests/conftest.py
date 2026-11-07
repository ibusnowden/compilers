"""
pytest plugin: every `*.uir` file under tests/ is one test.

    pytest tests/week01                 # run week 1 against YOUR compiler (compiler/uc)
    UC_ORACLE=1 pytest tests/week01     # run the .uir files against reference/oracle.py
                                        # (use this to check the tests you write!)
    UC_BACKEND=interp pytest ...        # force a backend (passed to run_uir)
    UC_TORCH_SHIM=1 pytest tests/week06 # run the Tensor-API tests against PyTorch instead of uc
                                        # (checks your tests AND your model code in uc/models)

Directives understood in .uir files (see docs/uir-spec.md):
    # expect: pass | compile-error | assert-fail
    # max-kernels: N
"""
import os, re, sys, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "compiler"))
if os.environ.get("UC_TORCH_SHIM") == "1":
  sys.path.insert(0, str(ROOT / "reference" / "torch_shim"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "tests"))

USE_ORACLE = os.environ.get("UC_ORACLE") == "1"

def pytest_collect_file(parent, file_path):
  if file_path.suffix == ".uir":
    return UirFile.from_parent(parent, path=file_path)

# the oracle only understands .uir files: don't even import the Python tests
collect_ignore_glob = ["*/test_*.py"] if USE_ORACLE else []

class UirFile(pytest.File):
  def collect(self):
    yield UirItem.from_parent(self, name=self.path.name)

def directives(src: str) -> dict:
  out = {}
  for line in src.splitlines():
    m = re.match(r"\s*#\s*([\w-]+)\s*:\s*(\S+)", line)
    if m: out[m.group(1)] = m.group(2)
  return out

class UirFailure(Exception): pass

class UirItem(pytest.Item):
  def runtest(self):
    src = self.path.read_text()
    d = directives(src)
    expect = d.get("expect", "pass")
    if USE_ORACLE:
      import oracle
      CompileError, AssertFailure = oracle.CompileError, oracle.AssertFailure
      run = lambda s: oracle.run(s)
    else:
      from uc.uir import run_uir
      from uc.errors import CompileError, AssertFailure
      run = lambda s: run_uir(s, backend=os.environ.get("UC_BACKEND") or None)
    result, msg = None, ""
    try:
      result = run(src); got = "pass"
    except CompileError as e: got, msg = "compile-error", str(e)
    except AssertFailure as e: got, msg = "assert-fail", str(e)
    if got != expect:
      raise UirFailure(f"expected {expect}, got {got}" + (f": {msg}" if msg else ""))
    if not USE_ORACLE and "max-kernels" in d and got == "pass":
      k = getattr(result, "kernels", None)
      if k is None or k > int(d["max-kernels"]):
        raise UirFailure(f"used {k} kernels, the limit is {d['max-kernels']}")

  def repr_failure(self, excinfo):
    if isinstance(excinfo.value, UirFailure):
      return f"{self.path.name}: {excinfo.value}\n\n" + self.path.read_text()
    return super().repr_failure(excinfo)

  def reportinfo(self):
    return self.path, 0, f"uir: {self.name}"
