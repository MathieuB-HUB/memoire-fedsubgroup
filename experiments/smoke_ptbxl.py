"""Smoke-test PTB-XL : valide les parties jamais executees de load_ptbxl
sans charger les 21k signaux. Verifie colonnes CSV, agregation superclasses,
lecture wfdb, formes, et un forward du ResNet1D.

Usage: uv run python experiments/smoke_ptbxl.py
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import wfdb

from src.data import _PTBXL_SUPERCLASSES, _aggregate_superclass
from src.models import build_model

DATA = Path("data/ptbxl")

print("=== 1. Chargement CSV + colonnes ===")
db = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
needed = ["site", "sex", "age", "scp_codes", "filename_lr", "filename_hr"]
missing = [c for c in needed if c not in db.columns]
print(f"  lignes={len(db)}  colonnes manquantes={missing or 'AUCUNE'}")
assert not missing, f"colonnes manquantes: {missing}"

print("=== 2. site value_counts (clients federes) ===")
sc = db["site"].value_counts()
big = sc[sc >= 200]
print(f"  sites totaux={len(sc)}  sites>=200rec={len(big)}  "
      f"-> {len(big)} clients + 1 'small_sites'")
print(f"  top sites: {list(sc.head(8).items())}")

print("=== 3. Agregation superclasses ===")
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
scp_df = pd.read_csv(DATA / "scp_statements.csv", index_col=0)
scp_df = scp_df[scp_df["diagnostic"] == 1]
supers = db["scp_codes"].apply(lambda d: _aggregate_superclass(d, scp_df))
n_with_label = (supers.apply(len) > 0).sum()
print(f"  superclasses={_PTBXL_SUPERCLASSES}")
print(f"  ECG avec >=1 label diagnostique: {n_with_label}/{len(db)} "
      f"({100*n_with_label/len(db):.1f}%)")
# Distribution
from collections import Counter
cnt = Counter()
for s in supers:
    for c in s:
        cnt[c] += 1
print(f"  distribution: {dict(cnt)}")

print("=== 4. Lecture wfdb (5 signaux) + timing ===")
sample = db.index[:5]
t0 = time.time()
sigs = []
for i in sample:
    arr, meta = wfdb.rdsamp(str(DATA / db.loc[i, "filename_lr"]))
    sigs.append(arr)
dt = (time.time() - t0) / len(sample)
arr0 = np.stack(sigs).transpose(0, 2, 1)  # (n, 12, T)
print(f"  forme apres transpose: {arr0.shape}  (attendu (5,12,1000))")
print(f"  fs={meta['fs']}  temps/lecture={dt*1000:.0f}ms  "
      f"-> 21k ECG ~= {dt*len(db)/60:.1f} min de chargement")
assert arr0.shape[1] == 12, "channels != 12"

print("=== 5. Forward ResNet1D ===")
cfg = {"model": "resnet1d"}
model = build_model(cfg, in_dim_or_channels=12, n_classes=len(_PTBXL_SUPERCLASSES))
x = torch.from_numpy(arr0.astype(np.float32))
with torch.no_grad():
    out = model(x)
n_params = sum(p.numel() for p in model.parameters())
print(f"  input={tuple(x.shape)}  output={tuple(out.shape)}  params={n_params:,}")
assert out.shape == (5, len(_PTBXL_SUPERCLASSES)), f"output shape {out.shape}"

print("\n=== SMOKE TEST OK : load_ptbxl est exploitable ===")
