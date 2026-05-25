"""Stress test : repartitionnement Dirichlet de Fed-Heart sur l'attribut sexe.

Question : a quel niveau de non-IID demographique FedSubgroup commence-t-il
a battre la perso par client ?

On pool les 4 sites Fed-Heart en un seul corpus (920 patients), puis on
repartitionne en K=4 clients synthetiques selon une Dirichlet(alpha) sur
l'attribut sensible (sexe). Petit alpha = clients extremes (un client peut
etre 95% femmes, un autre 95% hommes). Grand alpha = clients proches d'IID.

Pour chaque alpha dans {0.1, 0.5, 1.0, 5.0} et chaque graine, on execute
les 6 methodes et on sauvegarde un JSON par config dans results/stress/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ClientData, load_fed_heart, standardize_clients
from src.federated import METHODS, make_splits
from src.metrics import fairness_summary
from src.models import build_model
from src.utils import get_device, load_config, set_seed


MIN_SAMPLES_PER_CLIENT = 50


def dirichlet_repartition(clients: list[ClientData], alpha: float,
                          n_new_clients: int, rng,
                          min_samples: int = MIN_SAMPLES_PER_CLIENT,
                          max_tries: int = 100) -> list[ClientData]:
    """Pool tous les clients, puis re-partitionne en n_new_clients selon
    Dir(alpha) sur l'attribut sexe.

    Garde-fou : on **rejette et retire** tout tirage produisant un client avec
    moins de `min_samples` echantillons, jusqu'a `max_tries` essais. Cela
    garantit K_effectif == n_new_clients (pas de fusion qui ferait passer
    alpha=0.1 a 2 clients effectifs).
    """
    X = np.concatenate([c.X for c in clients])
    y = np.concatenate([c.y for c in clients])
    sex = np.concatenate([c.sex for c in clients])
    age = np.concatenate([c.age_bin for c in clients])

    last_sizes = None
    for attempt in range(max_tries):
        new_clients_idx: list[list[int]] = [[] for _ in range(n_new_clients)]
        for s in (0, 1):
            idx_s = np.where(sex == s)[0].copy()
            rng.shuffle(idx_s)
            props = rng.dirichlet([alpha] * n_new_clients)
            cuts = (np.cumsum(props) * len(idx_s)).astype(int)
            prev = 0
            for k, cut in enumerate(cuts):
                new_clients_idx[k].extend(idx_s[prev:cut].tolist())
                prev = cut

        sizes = [len(idx) for idx in new_clients_idx]
        last_sizes = sizes
        if min(sizes) >= min_samples:
            break
    else:
        raise RuntimeError(
            f"dirichlet_repartition: impossible de tirer une partition avec "
            f">= {min_samples} samples/client en {max_tries} essais "
            f"(alpha={alpha}, dernieres tailles={last_sizes})."
        )

    out = []
    for k, idx in enumerate(new_clients_idx):
        idx_arr = np.array(idx)
        out.append(ClientData(
            name=f"synth_{k}",
            X=X[idx_arr], y=y[idx_arr],
            sex=sex[idx_arr], age_bin=age[idx_arr],
        ))
    # Verification post-tirage : K_effectif == K_demande.
    assert len(out) == n_new_clients, (
        f"K_effectif={len(out)} != K_demande={n_new_clients}"
    )
    print(f"  [dirichlet] alpha={alpha} tries={attempt + 1} sizes={last_sizes}")
    return out


def run_one(method: str, alpha: float, seed: int, cfg: dict,
            base_clients: list[ClientData], device, out_dir: Path):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    clients = dirichlet_repartition(base_clients, alpha, n_new_clients=4, rng=rng)
    clients = standardize_clients(clients)
    splits = make_splits(clients, test_frac=0.2, seed=seed, device=str(device))

    def factory(multi_head: bool = False):
        m = build_model(cfg, clients[0].X.shape[1], 1, multi_head=multi_head)
        return m.to(device)

    t0 = time.time()
    y_true, y_score, sex, age = METHODS[method](
        splits, factory,
        n_rounds=cfg["n_rounds"], lr=cfg["lr"],
        local_epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
    )
    dur = time.time() - t0
    summary = fairness_summary(y_true, y_score, sex)

    out = {
        "method": method, "alpha": alpha, "seed": seed,
        "duration_s": dur,
        "client_sizes": [int(len(c)) for c in clients],
        "client_sex_means": [float(c.sex.mean()) for c in clients],
        **summary,
    }
    fname = out_dir / f"stress_{method}_alpha{alpha}_seed{seed}.json"
    fname.write_text(json.dumps(out, indent=2))
    print(f"  {method:12s} alpha={alpha:>4}  seed={seed}  "
          f"AUC={summary['auc_global']:.3f}  "
          f"WG={summary['worst_group_auc']:.3f}  ({dur:.1f}s)")


def main():
    cfg = load_config("configs/fed_heart.yaml")
    device = get_device()
    out_dir = Path("results/stress")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_clients = load_fed_heart("data/fed_heart")
    print(f"[stress] {sum(len(c) for c in base_clients)} patients pooles.")

    methods = ["fedavg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup"]
    alphas = [0.1, 0.5, 1.0, 5.0]
    seeds = list(range(10))

    total = len(methods) * len(alphas) * len(seeds)
    i = 0
    t0 = time.time()
    for alpha in alphas:
        for seed in seeds:
            for method in methods:
                i += 1
                print(f"[{i}/{total}]", end=" ")
                run_one(method, alpha, seed, cfg, base_clients, device, out_dir)
    print(f"\n=== {total} runs termines en {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
