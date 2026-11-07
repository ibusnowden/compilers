"""The shim can't count kernels; tests that need these skip when SHIM is True."""
SHIM = True
kernels = compiles = flops = bytes = 0
def reset():
  global kernels, compiles, flops, bytes
  kernels = compiles = flops = bytes = 0
