"""Week 1: the untyped lambda calculus as a term-rewriting system on your UOps.
Required API (uc/lam.py): parse, normalize, church, unchurch, app, NonTerminating."""
import pytest
from uc import lam

PLUS = r"\m n f x. m f (n f x)"
MULT = r"\m n f. m (n f)"
EXP  = r"\m n. n m"
PRED = r"\n f x. n (\g h. h (g f)) (\u. x) (\u. u)"
TRUE, FALSE = r"\x y. x", r"\x y. y"

def ap(*ts):
  out = ts[0]
  for t in ts[1:]: out = lam.app(out, t)
  return out

def test_alpha_equivalence_is_identity():
  assert lam.parse(r"\x. x") is lam.parse(r"\y. y")
  assert lam.parse(r"\x y. x") is lam.parse(r"\a b. a")
  assert lam.parse(r"\x y. x") is not lam.parse(r"\x y. y")

def test_church_roundtrip():
  for n in range(10): assert lam.unchurch(lam.church(n)) == n

def test_arithmetic():
  P, M, E = (lam.parse(s) for s in (PLUS, MULT, EXP))
  assert lam.unchurch(ap(P, lam.church(2), lam.church(3))) == 5
  assert lam.unchurch(ap(M, lam.church(3), lam.church(4))) == 12
  assert lam.unchurch(ap(E, lam.church(2), lam.church(5))) == 32
  assert lam.normalize(ap(P, lam.church(2), lam.church(3))) is lam.church(5)

def test_predecessor():
  assert lam.unchurch(ap(lam.parse(PRED), lam.church(7))) == 6

def test_capture_avoiding():
  # (\x. \y. x) y  must not capture the free y: test with a closed version
  t = lam.parse(r"\y. (\x. \y. x) y")
  assert lam.normalize(t) is lam.parse(r"\y. \z. y")

def test_normal_order_finds_normal_forms():
  K, I = lam.parse(TRUE), lam.parse(r"\x. x")
  omega = lam.parse(r"(\x. x x) (\x. x x)")
  assert lam.normalize(ap(K, I, omega)) is I   # applicative order would loop forever

def test_divergence_detected():
  with pytest.raises(lam.NonTerminating):
    lam.normalize(lam.parse(r"(\x. x x) (\x. x x)"), max_steps=1000)
