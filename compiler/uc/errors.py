"""The two exceptions the test harness understands. Do not rename them."""

class CompileError(Exception):
  """Anything rejected before running: parse errors, type/shape errors, failed bounds proofs, failed `prove`."""

class AssertFailure(Exception):
  """An `assert` in the program evaluated to false when the program ran."""
