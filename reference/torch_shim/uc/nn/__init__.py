import math
from uc.tensor import Tensor

def _uniform(shape, bound): return Tensor.uniform(*shape, low=-bound, high=bound, requires_grad=True)

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    b = 1 / math.sqrt(in_features)
    self.weight = _uniform((out_features, in_features), b)
    self.bias = _uniform((out_features,), b) if bias else None
  def __call__(self, x): return x.linear(self.weight, self.bias)

class Embedding:
  def __init__(self, vocab_size, dim):
    self.vocab_size, self.weight = vocab_size, Tensor.randn(vocab_size, dim, requires_grad=True)
  def __call__(self, idx): return self.weight[idx]

class Conv2d:
  def __init__(self, cin, cout, k, stride=1, padding=0, bias=True):
    b = 1 / math.sqrt(cin * k * k)
    self.stride, self.padding = stride, padding
    self.weight = _uniform((cout, cin, k, k), b)
    self.bias = _uniform((cout,), b) if bias else None
  def __call__(self, x): return x.conv2d(self.weight, self.bias, self.stride, self.padding)

class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps, self.weight, self.bias = eps, Tensor.ones(dim, requires_grad=True), Tensor.zeros(dim, requires_grad=True)
  def __call__(self, x): return x.layernorm(eps=self.eps) * self.weight + self.bias

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps, self.weight = eps, Tensor.ones(dim, requires_grad=True)
  def __call__(self, x): return x.rmsnorm(eps=self.eps) * self.weight

class BatchNorm2d:
  def __init__(self, c, eps=1e-5):
    self.eps = eps
    self.weight, self.bias = Tensor.ones(c, requires_grad=True), Tensor.zeros(c, requires_grad=True)
    self.running_mean, self.running_var = Tensor.zeros(c), Tensor.ones(c)
  def __call__(self, x):
    sh = (1, -1, 1, 1)
    inv = (self.running_var.reshape(*sh) + self.eps).rsqrt()
    return (x - self.running_mean.reshape(*sh)) * inv * self.weight.reshape(*sh) + self.bias.reshape(*sh)
