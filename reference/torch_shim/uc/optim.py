import torch

class Optimizer:
  def __init__(self, params):
    self.params = [p for p in params if p.requires_grad]
  def zero_grad(self):
    for p in self.params: p.t.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
    super().__init__(params); self.o = torch.optim.SGD([p.t for p in self.params], lr=lr, momentum=momentum, weight_decay=weight_decay)
  def step(self): self.o.step()

class AdamW(Optimizer):
  def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    super().__init__(params); self.o = torch.optim.AdamW([p.t for p in self.params], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
  def step(self): self.o.step()
