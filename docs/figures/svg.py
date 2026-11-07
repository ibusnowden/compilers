"""A tiny SVG drawing library for the course figures.  One look for every diagram:
white card, rounded boxes, one accent colour per concept.

  ALU / values   blue       memory / buffers   green     loops / ranges   orange
  symbolic       purple     errors / hazards   red       neutral          gray
"""
from __future__ import annotations
from html import escape

FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"
INK, MUTED, LINE, BG = "#1f2430", "#6b7280", "#9aa3b2", "#ffffff"
C = {  # (fill, stroke, text)
  "blue": ("#e8effc", "#3b6fd8", "#1d3f8f"), "green": ("#e5f5ec", "#2e9e6a", "#165c3b"),
  "orange": ("#fdf0e1", "#e08a2c", "#8a4a0b"), "purple": ("#efeafa", "#7c5cc4", "#45297f"),
  "red": ("#fbe9e9", "#d64545", "#8c1f1f"), "gray": ("#f1f3f6", "#9aa3b2", "#384152"),
  "yellow": ("#fdf7dc", "#c9a227", "#6d5608"), "teal": ("#e2f4f4", "#2a9d9d", "#145757"),
  "white": ("#ffffff", "#9aa3b2", "#1f2430"),
}

class SVG:
  def __init__(self, w, h, title=None, subtitle=None):
    self.w, self.h, self.parts = w, h, []
    self.top = 0
    if title:
      self.text(24, 34, title, size=19, weight=700)
      self.top = 44
    if subtitle:
      self.text(24, self.top + 18, subtitle, size=13, color=MUTED)
      self.top += 26

  # ---- primitives
  def raw(self, s): self.parts.append(s)
  def text(self, x, y, s, size=13, anchor="start", weight=400, color=INK, mono=False, italic=False):
    lines = str(s).split("\n")
    fam = MONO if mono else FONT
    sty = ' font-style="italic"' if italic else ""
    dy0 = -(len(lines) - 1) * size * 0.6
    for i, ln in enumerate(lines):
      yy = y + dy0 + i * size * 1.2 + size * 0.35     # vertically centred without dominant-baseline
      self.raw(f'<text x="{x:.1f}" y="{yy:.1f}" font-family="{fam}" font-size="{size}" xml:space="preserve" '
               f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{sty}>{escape(ln)}</text>')
  def rect(self, x, y, w, h, fill="#fff", stroke=LINE, rx=8, sw=1.5, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')
  def box(self, x, y, w, h, label="", color="gray", size=13, mono=False, rx=8, weight=500, dash=None, sub=None):
    f, s, t = C[color]
    self.rect(x, y, w, h, f, s, rx, dash=dash)
    if sub:
      self.text(x + w / 2, y + h / 2 - 8, label, size=size, anchor="middle", weight=weight, color=t, mono=mono)
      self.text(x + w / 2, y + h / 2 + 10, sub, size=size - 2, anchor="middle", color=MUTED, mono=mono)
    elif label != "":
      self.text(x + w / 2, y + h / 2, label, size=size, anchor="middle", weight=weight, color=t, mono=mono)
    return (x, y, w, h)
  def node(self, cx, cy, label, color="blue", w=None, h=30, size=12, mono=True):
    w = w or max(44, 8.2 * len(label) + 18)
    return self.box(cx - w / 2, cy - h / 2, w, h, label, color, size=size, mono=mono, rx=h / 2)
  def circle(self, cx, cy, r, color="blue", label="", size=12):
    f, s, t = C[color]
    self.raw(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
    if label: self.text(cx, cy, label, size=size, anchor="middle", color=t, weight=600)
  def line(self, x1, y1, x2, y2, color=LINE, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')
  def arrow(self, x1, y1, x2, y2, color="#5b6474", sw=1.6, dash=None, label=None, lsize=11, lcolor=MUTED, loff=(0, -9), head=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mid = self._marker(color) if head else ""
    self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}{mid}/>')
    if label: self.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label, size=lsize, anchor="middle", color=lcolor)
  def path(self, pts, color="#5b6474", sw=1.6, dash=None, head=True, label=None, lpos=None, lsize=11):
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mid = self._marker(color) if head else ""
    self.raw(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{ds}{mid}/>')
    if label and lpos: self.text(lpos[0], lpos[1], label, size=lsize, anchor="middle", color=MUTED)
  def curve(self, x1, y1, cx, cy, x2, y2, color="#5b6474", sw=1.6, dash=None, head=True):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mid = self._marker(color) if head else ""
    self.raw(f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{mid}/>')
  def poly(self, pts, fill, stroke=LINE, sw=1.2, opacity=1):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    self.raw(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')
  def code(self, x, y, w, lines, size=12, title=None, color="gray", hl=None, lh=None):
    """a code panel; hl = {line_index: color}"""
    lh = lh or size * 1.5
    h = len(lines) * lh + 20 + (22 if title else 0)
    f, s, t = C[color]
    self.rect(x, y, w, h, "#fbfbfd", s, 8)
    yy = y + 12
    if title:
      self.text(x + 12, yy + 6, title, size=11, color=MUTED, weight=600); yy += 22
    for i, ln in enumerate(lines):
      if hl and i in hl:
        hf = C[hl[i]][0]
        self.rect(x + 4, yy, w - 8, lh, hf, hf, 3, sw=0)
      self.text(x + 12, yy + lh / 2, ln, size=size, mono=True)
      yy += lh
    return h
  def legend(self, x, y, items, size=11):
    for i, (color, label) in enumerate(items):
      f, s, _ = C[color]
      self.rect(x, y + i * 20, 14, 14, f, s, 3)
      self.text(x + 22, y + i * 20 + 7, label, size=size, color=MUTED)
  def grid(self, x, y, rows, cols, cell, fills=None, labels=None, size=11, stroke="#b8c0cc", mono=True):
    for r in range(rows):
      for c in range(cols):
        f = fills(r, c) if callable(fills) else (fills or "#fff")
        self.rect(x + c * cell, y + r * cell, cell, cell, f, stroke, 0, sw=1)
        if labels is not None:
          lab = labels(r, c) if callable(labels) else labels[r][c]
          if lab != "": self.text(x + c * cell + cell / 2, y + r * cell + cell / 2, str(lab), size=size, anchor="middle", mono=mono)

  def _marker(self, color):
    mid = "m" + color.strip("#")
    if not any(f'id="{mid}"' in p for p in self.parts):
      self.parts.insert(0, f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                           f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>')
    return f' marker-end="url(#{mid})"'

  def save(self, path):
    body = "\n".join(self.parts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n'
           f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="14" fill="{BG}" stroke="#e3e6eb"/>\n{body}\n</svg>\n')
    with open(path, "w") as f: f.write(svg)
