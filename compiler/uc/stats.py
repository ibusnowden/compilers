"""Counters the tests read (docs/tensor-api.md).  Increment them from your runtime."""
kernels = compiles = flops = bytes = 0
def reset():
  global kernels, compiles, flops, bytes
  kernels = compiles = flops = bytes = 0
