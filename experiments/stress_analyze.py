"""Analyse le stress test Dirichlet : figure WG-AUC vs alpha par methode."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["fedavg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup"]
COLORS = {"fedavg": "#1f77b4", "fedbn": "#2ca02c", "fedper": "#ff7f0e",
          "fedrep": "#9467bd", "ditto": "#8c564b", "fedsubgroup": "#d62728"}


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in folder.glob("*.json"):
        d = json.loads(p.read_text())
        rows.append({
            "method": d["method"], "alpha": d["alpha"], "seed": d["seed"],
            "auc_global": d["auc_global"],
            "worst_group_auc": d["worst_group_auc"],
            "auc_gap": d["auc_gap"],
        })
    return pd.DataFrame(rows)


def lineplot(df, metric, ylabel, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHOD_ORDER:
        sub = df[df["method"] == m].groupby("alpha")[metric].agg(["mean", "std"])
        if sub.empty: continue
        lw = 2.5 if m == "fedsubgroup" else 1.3
        ls = "-" if m == "fedsubgroup" else "--"
        ax.errorbar(sub.index, sub["mean"], yerr=sub["std"],
                    label=m, color=COLORS[m], linewidth=lw, linestyle=ls,
                    marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$ Dirichlet (petit = clients extrêmes, grand = IID)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"Stress test non-IID : {ylabel} vs $\\alpha$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    df = load(Path("results/stress"))
    print(f"{len(df)} runs charges")

    # Pivot mean +/- std
    g = df.groupby(["method", "alpha"])
    cols = ["auc_global", "worst_group_auc", "auc_gap"]
    mean = g[cols].mean()
    std = g[cols].std()
    print("\n=== Stress Dirichlet : moyenne sur 3 graines ===\n")
    summary = pd.DataFrame(index=mean.index)
    for c in cols:
        summary[c] = mean[c].map(lambda v: f"{v:.3f}") + " +/- " + std[c].map(lambda v: f"{v:.3f}")
    print(summary.to_string())

    # Quel(s) alpha(s) FedSubgroup gagne-t-il ?
    print("\n=== FedSubgroup vs meilleure perso par client (FedPer/FedRep/Ditto/FedBN) ===\n")
    rows = []
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha]
        fs = sub[sub["method"] == "fedsubgroup"]
        b = sub[sub["method"].isin(["fedbn", "fedper", "fedrep", "ditto"])]
        # per seed: best of B
        best_b = b.groupby("seed")["worst_group_auc"].max()
        diff = fs.set_index("seed")["worst_group_auc"] - best_b
        rows.append({
            "alpha": alpha,
            "fs_mean": fs["worst_group_auc"].mean(),
            "bestB_mean": best_b.mean(),
            "diff_mean": diff.mean(),
            "diff_std": diff.std(),
            "fs_wins": int((diff > 0).sum()),
            "fs_loses": int((diff < 0).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    lineplot(df, "worst_group_auc", "Worst-group AUC",
             fig_dir / "stress_dirichlet_wg.png")
    lineplot(df, "auc_global", "AUC global",
             fig_dir / "stress_dirichlet_auc.png")
    lineplot(df, "auc_gap", "AUC gap (max - min entre groupes)",
             fig_dir / "stress_dirichlet_gap.png")
    print(f"\nFigures -> {fig_dir}")


if __name__ == "__main__":
    main()
