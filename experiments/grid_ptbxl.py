"""Grille complete PTB-XL : 7 methodes x 2 attrs sensibles x N seeds.

Contrairement a grid_fed_heart (un subprocess par run), on charge PTB-XL une
seule fois (~8 min de lecture wfdb) puis on boucle en interne -- sinon chaque
run rechargerait les 21k signaux.

Usage:
    uv run python experiments/grid_ptbxl.py                 # 5 seeds, 50 rounds
    uv run python experiments/grid_ptbxl.py --seeds 3
    uv run python experiments/grid_ptbxl.py --rounds 2      # smoke : valide les 7 methodes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run import load_dataset, run_one
from src.utils import get_device, load_config

METHODS = ["fedavg", "fedavg_sg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup"]
SENS = ["sex", "age"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ptbxl.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=None,
                        help="override n_rounds (smoke : ex. 2)")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--out", default="results/ptbxl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_file"] = args.config
    if args.rounds is not None:
        cfg["n_rounds"] = args.rounds
        print(f"[grid] OVERRIDE n_rounds={args.rounds} (smoke)")
    device = get_device()
    seeds = list(range(args.seeds))

    # Chargement unique.
    clients, in_dim, n_classes = load_dataset(cfg)

    n_total = len(args.methods) * len(SENS) * len(seeds)
    print(f"[grid] {n_total} runs : {args.methods} x {SENS} x seeds{seeds} "
          f"sur device={device}")

    t0 = time.time()
    fails = []
    i = 0
    out_dir = Path(args.out)
    skipped = 0
    for m in args.methods:
        for s in SENS:
            for seed in seeds:
                i += 1
                # Reprise : si le JSON existe deja, on saute (relance idempotente).
                done_path = out_dir / f"{cfg['dataset']}_{m}_{s}_seed{seed}.json"
                if done_path.exists():
                    skipped += 1
                    print(f"[{i}/{n_total}] SKIP (deja fait) {m}/{s}/seed{seed}")
                    continue
                el = time.time() - t0
                print(f"\n[{i}/{n_total}] method={m} sensitive={s} seed={seed} "
                      f"(ecoule {el/60:.1f} min)")
                try:
                    run_one(cfg, clients, in_dim, n_classes, m, seed, s,
                            out=args.out, device=device)
                except Exception as e:
                    print(f"[grid] ECHEC {m}/{s}/seed{seed}: {type(e).__name__}: {e}")
                    fails.append((m, s, seed, repr(e)))

    if skipped:
        print(f"[grid] {skipped} runs deja presents, sautes (reprise)")

    print(f"\n=== grille terminee en {(time.time()-t0)/60:.1f} min, "
          f"{len(fails)} echecs / {n_total} ===")
    for f in fails:
        print(" ", f)


if __name__ == "__main__":
    main()
