"""Reference ResNet-18 (eval mode) in plain PyTorch, functional over a torchvision-named state dict."""
import torch
import torch.nn.functional as F

def init(num_classes=1000, seed=0):
  g = torch.Generator().manual_seed(seed)
  sd = {}
  def conv(name, cout, cin, k): sd[name + ".weight"] = torch.randn(cout, cin, k, k, generator=g) * (2.0 / (cin * k * k)) ** 0.5
  def bn(name, c):
    sd[name + ".weight"] = 1 + 0.1 * torch.randn(c, generator=g); sd[name + ".bias"] = 0.1 * torch.randn(c, generator=g)
    sd[name + ".running_mean"] = 0.1 * torch.randn(c, generator=g); sd[name + ".running_var"] = 1 + 0.1 * torch.rand(c, generator=g)
  conv("conv1", 64, 3, 7); bn("bn1", 64)
  cin = 64
  for li, c in enumerate([64, 128, 256, 512], 1):
    for b in range(2):
      p = f"layer{li}.{b}."
      conv(p + "conv1", c, cin if b == 0 else c, 3); bn(p + "bn1", c); conv(p + "conv2", c, c, 3); bn(p + "bn2", c)
      if b == 0 and li > 1: conv(p + "downsample.0", c, cin, 1); bn(p + "downsample.1", c)
    cin = c
  sd["fc.weight"] = torch.randn(num_classes, 512, generator=g) * 0.02; sd["fc.bias"] = torch.zeros(num_classes)
  return sd

def forward(sd, x):
  def bn(x, n): return F.batch_norm(x, sd[n + ".running_mean"], sd[n + ".running_var"], sd[n + ".weight"], sd[n + ".bias"], False, 0.0, 1e-5)
  x = F.relu(bn(F.conv2d(x, sd["conv1.weight"], stride=2, padding=3), "bn1"))
  x = F.max_pool2d(x, 3, 2, 1)
  for li in range(1, 5):
    for b in range(2):
      p = f"layer{li}.{b}."
      stride = 2 if b == 0 and li > 1 else 1
      out = F.relu(bn(F.conv2d(x, sd[p + "conv1.weight"], stride=stride, padding=1), p + "bn1"))
      out = bn(F.conv2d(out, sd[p + "conv2.weight"], padding=1), p + "bn2")
      sc = bn(F.conv2d(x, sd[p + "downsample.0.weight"], stride=2), p + "downsample.1") if p + "downsample.0.weight" in sd else x
      x = F.relu(out + sc)
  return F.linear(x.mean((2, 3)), sd["fc.weight"], sd["fc.bias"])
