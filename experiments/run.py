"""Point d'entree d'une experience FL.

Usage:
    python experiments/run.py --config configs/fed_heart.yaml --method fedavg --seed 0 --sensitive sex

Le chargement du dataset (load_dataset) et l'execution d'un run (run_one) sont
exposes separement pour que les grilles (grid_*.py) chargent les donnees une
seule fois et bouclent en interne -- crucial pour PTB-XL ou la lecture des 21k
signaux coute ~6 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Permet 'python experiments/run.py' depuis la racine code/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data import load_fed_heart, standardize_clients, load_ptbxl
from src.federated import METHODS, make_splits
from src.metrics import fairness_summary, bootstrap_metric, safe_auc
from src.models import build_model
from src.utils import get_device, load_config, set_seed


def build_factory(cfg, in_dim_or_ch, n_classes, device):
    """Renvoie un closure build_fn(multi_head=False) pour le runner."""
    def factory(multi_head: bool = False):
        m = build_model(cfg, in_dim_or_ch, n_classes, multi_head=multi_head)
        return m.to(device)
    return factory


def load_dataset(cfg):
    """Charge le dataset une fois. Renvoie (clients, in_dim, n_classes)."""
    t0 = time.time()
    if cfg["dataset"] == "fed_heart":
        clients = load_fed_heart("data/fed_heart")
        clients = standardize_clients(clients)
        in_dim, n_classes = clients[0].X.shape[1], 1
    elif cfg["dataset"] == "ptbxl":
        clients = load_ptbxl(cfg["data_path"], sampling_rate=cfg["sampling_rate"])
        in_dim, n_classes = 12, clients[0].y.shape[1]
    else:
        raise ValueError(cfg["dataset"])
    print(f"[load] {len(clients)} clients charges en {time.time() - t0:.1f}s")
    for c in clients:
        print(f"       {c.name:20s}  n={len(c):5d}  pos_rate={c.y.mean():.2f}  "
              f"sex_mean={c.sex.mean():.2f}  age_mean={c.age_bin.mean():.2f}")
    return clients, in_dim, n_classes


def run_one(cfg, clients, in_dim, n_classes, method, seed, sensitive,
            out="results", device=None):
    """Execute un run FL et sauvegarde le JSON. Renvoie le payload."""
    if device is None:
        device = get_device()
    set_seed(seed)
    print(f"[run] dataset={cfg['dataset']} method={method} seed={seed} "
          f"sensitive={sensitive} device={device}")

    splits = make_splits(clients, test_frac=0.2, seed=seed, device=str(device))

    in_arg = in_dim if cfg["model"] == "mlp" else 12
    if method == "fedavg_sg":
        in_arg = in_arg + 4
    factory = build_factory(cfg, in_arg, n_classes, device)

    runner = METHODS[method]
    t0 = time.time()
    y_true, y_score, sex, age = runner(
        splits, factory,
        n_rounds=cfg["n_rounds"], lr=cfg["lr"],
        local_epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
    )
    dur = time.time() - t0
    print(f"[run] entrainement: {dur:.1f}s")

    sens_arr = sex if sensitive == "sex" else age
    summary = fairness_summary(y_true, y_score, sens_arr)

    # IC bootstrap sur worst-group AUC : on bootstrap au niveau sample.
    def wga(y_true, y_score, sensitive):
        s = fairness_summary(y_true, y_score, sensitive)
        return s["worst_group_auc"]
    wga_lo, wga_hi = bootstrap_metric(
        wga, seed=seed, n_boot=500,
        y_true=y_true, y_score=y_score, sensitive=sens_arr,
    )

    print(f"[run] AUC={summary['auc_global']:.3f}  WGA={summary['worst_group_auc']:.3f}  "
          f"gap={summary['auc_gap']:.3f}  DP={summary['dp_gap']:.3f}  EO={summary['eo_gap']:.3f}  "
          f"IC95=[{wga_lo:.3f},{wga_hi:.3f}]")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['dataset']}_{method}_{sensitive}_seed{seed}.json"
    payload = {
        "config_file": cfg.get("_config_file", ""),
        "dataset": cfg["dataset"],
        "method": method,
        "seed": seed,
        "sensitive": sensitive,
        "duration_s": dur,
        **summary,
        "wga_ci95": [wga_lo, wga_hi],
        "n_test": int(len(y_true)),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[run] resultats -> {out_path}")

    # Libere la VRAM entre runs (utile quand run_one est appele en boucle).
    del splits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensitive", default="sex", choices=["sex", "age"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_file"] = args.config
    device = get_device()

    clients, in_dim, n_classes = load_dataset(cfg)
    run_one(cfg, clients, in_dim, n_classes, args.method, args.seed,
            args.sensitive, out=args.out, device=device)


if __name__ == "__main__":
    main()
