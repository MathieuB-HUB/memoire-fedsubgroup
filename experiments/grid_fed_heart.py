"""Lance la grille complete sur Fed-Heart : 7 methodes x 2 attrs x 10 seeds."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

METHODS = ["fedavg", "fedavg_sg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup"]
SENS = ["sex", "age"]
SEEDS = list(range(10))

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

t0 = time.time()
fails = []
n_total = len(METHODS) * len(SENS) * len(SEEDS)
i = 0
for m in METHODS:
    for s in SENS:
        for seed in SEEDS:
            i += 1
            print(f"\n[{i}/{n_total}] method={m} sensitive={s} seed={seed}")
            r = subprocess.run(
                [PY, "experiments/run.py", "--config", "configs/fed_heart.yaml",
                 "--method", m, "--seed", str(seed), "--sensitive", s],
                cwd=ROOT,
            )
            if r.returncode != 0:
                fails.append((m, s, seed))

print(f"\n=== termine en {time.time() - t0:.1f}s, {len(fails)} echecs ===")
for f in fails:
    print(" ", f)
