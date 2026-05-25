"""Agrege les JSON de results/ en tables + tests Wilcoxon vs FedAvg.

Produit :
  - results/summary_<dataset>.csv  : table large mean +/- std par (methode, attr)
  - results/wilcoxon_<dataset>.csv : test apparie de chaque methode vs FedAvg
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import paired_wilcoxon


def load_results(folder: Path) -> pd.DataFrame:
    rows = []
    for p in folder.glob("*.json"):
        d = json.loads(p.read_text())
        rows.append({
            "dataset": d["dataset"],
            "method": d["method"],
            "sensitive": d["sensitive"],
            "seed": d["seed"],
            "auc_global": d["auc_global"],
            "worst_group_auc": d["worst_group_auc"],
            "auc_gap": d["auc_gap"],
            "dp_gap": d["dp_gap"],
            "eo_gap": d["eo_gap"],
            "duration_s": d.get("duration_s", float("nan")),
        })
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "sensitive"])
    cols = ["auc_global", "worst_group_auc", "auc_gap", "dp_gap", "eo_gap"]
    mean = g[cols].mean()
    std = g[cols].std()
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = mean[c].map(lambda v: f"{v:.3f}") + " +/- " + std[c].map(lambda v: f"{v:.3f}")
    out["n_seeds"] = g.size()
    return out.reset_index()


def wilcoxon_vs(df: pd.DataFrame, ref: str = "fedavg",
                metric: str = "worst_group_auc") -> pd.DataFrame:
    rows = []
    for sens in df["sensitive"].unique():
        sub = df[df["sensitive"] == sens]
        ref_scores = sub[sub["method"] == ref].sort_values("seed")[metric].values
        for m in sub["method"].unique():
            if m == ref:
                continue
            other = sub[sub["method"] == m].sort_values("seed")[metric].values
            if len(other) != len(ref_scores) or len(ref_scores) < 2:
                continue
            res = paired_wilcoxon(list(other), list(ref_scores))
            rows.append({
                "metric": metric,
                "sensitive": sens,
                "method": m,
                "vs": ref,
                "mean_diff": res["mean_diff"],
                "wilcoxon_p": res["p"],
                "n_pairs": len(ref_scores),
            })
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    df = load_results(Path(args.results))
    df = df[df["dataset"] == args.dataset]
    if df.empty:
        print(f"aucun resultat pour dataset={args.dataset}")
        return

    print(f"\n=== {args.dataset} : {len(df)} runs, "
          f"{df['method'].nunique()} methodes, {df['seed'].nunique()} seeds ===\n")

    summary = summary_table(df)
    print(summary.to_string(index=False))
    summary.to_csv(Path(args.results) / f"summary_{args.dataset}.csv", index=False)

    print("\nWilcoxon worst-group AUC vs FedAvg :")
    wga = wilcoxon_vs(df, ref="fedavg", metric="worst_group_auc")
    print(wga.to_string(index=False))
    wga.to_csv(Path(args.results) / f"wilcoxon_{args.dataset}.csv", index=False)

    print("\nWilcoxon worst-group AUC vs meilleur de FedPer/FedRep/Ditto/FedBN :")
    # Pour H1 : FedSubgroup vs best of philosophy B per (sensitive, seed).
    philo_b = ["fedbn", "fedper", "fedrep", "ditto"]
    rows_h1 = []
    for sens in df["sensitive"].unique():
        sub = df[df["sensitive"] == sens]
        fs = sub[sub["method"] == "fedsubgroup"].sort_values("seed")
        if fs.empty:
            continue
        best_b_per_seed = []
        for seed in fs["seed"]:
            cand = sub[(sub["method"].isin(philo_b)) & (sub["seed"] == seed)]
            if cand.empty:
                continue
            best_b_per_seed.append(cand["worst_group_auc"].max())
        if len(best_b_per_seed) == len(fs):
            res = paired_wilcoxon(fs["worst_group_auc"].tolist(), best_b_per_seed)
            rows_h1.append({
                "sensitive": sens,
                "fedsubgroup_mean": fs["worst_group_auc"].mean(),
                "best_philo_b_mean": float(np.mean(best_b_per_seed)),
                "mean_diff": res["mean_diff"],
                "wilcoxon_p": res["p"],
            })
    h1 = pd.DataFrame(rows_h1)
    print(h1.to_string(index=False))
    h1.to_csv(Path(args.results) / f"h1_{args.dataset}.csv", index=False)


if __name__ == "__main__":
    main()
