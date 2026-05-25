"""Test : dirichlet_repartition garantit K_effectif == K_demande pour
plusieurs valeurs de alpha et de graines.

Bug initial : avec alpha=0.1 et garde-fou min=20, 12/18 runs avaient
K_effectif=2 (fusion des clients quasi-vides). Le fix force min=50 + rejet
et retire jusqu'a max_tries.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from experiments.stress_dirichlet import dirichlet_repartition, MIN_SAMPLES_PER_CLIENT
from src.data import ClientData


def _synthetic_pool(n_total: int = 920, seed: int = 0) -> list[ClientData]:
    """Fabrique un pool similaire en taille a Fed-Heart pool (~920 patients).

    On expose 4 'sites' comme dans Fed-Heart pour que la signature soit
    realiste; le contenu importe peu, dirichlet_repartition pool tout.
    """
    rng = np.random.default_rng(seed)
    sizes = [n_total // 4] * 4
    sizes[-1] += n_total - sum(sizes)
    out = []
    for k, n in enumerate(sizes):
        out.append(ClientData(
            name=f"site_{k}",
            X=rng.standard_normal((n, 13)).astype(np.float32),
            y=rng.integers(0, 2, size=n).astype(np.float32),
            sex=rng.integers(0, 2, size=n).astype(np.int64),
            age_bin=rng.integers(0, 2, size=n).astype(np.int64),
        ))
    return out


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 5.0])
def test_dirichlet_keeps_K_for_all_seeds(alpha):
    base = _synthetic_pool()
    for seed in range(10):
        rng = np.random.default_rng(seed)
        clients = dirichlet_repartition(base, alpha=alpha,
                                        n_new_clients=4, rng=rng)
        assert len(clients) == 4, (
            f"alpha={alpha} seed={seed}: K_effectif={len(clients)} != 4"
        )
        sizes = [len(c) for c in clients]
        assert min(sizes) >= MIN_SAMPLES_PER_CLIENT, (
            f"alpha={alpha} seed={seed}: min(sizes)={min(sizes)} < "
            f"{MIN_SAMPLES_PER_CLIENT}"
        )


def test_dirichlet_raises_when_impossible():
    """Avec min_samples enorme, la fonction doit lever une exception explicite."""
    base = _synthetic_pool()
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="impossible de tirer"):
        dirichlet_repartition(base, alpha=0.01, n_new_clients=4, rng=rng,
                              min_samples=10_000, max_tries=5)
