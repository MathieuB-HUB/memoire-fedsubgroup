"""Sanity checks de base."""
import torch
from src.utils import set_seed, get_device


def test_seed_reproductible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_device_disponible():
    d = get_device()
    assert d.type in ("cuda", "cpu")
