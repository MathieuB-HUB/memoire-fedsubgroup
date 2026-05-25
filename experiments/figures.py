"""Genere les figures du memoire a partir des JSON de results/.

Figures produites (dans figures/) :
  - perf_vs_fairness_<dataset>_<sens>.png : scatter AUC global vs worst-group AUC
  - bars_worst_group_<dataset>_<sens>.png : barres worst-group AUC par methode
  - gaps_<dataset>_<sens>.png             : barres auc_gap / dp_gap / eo_gap
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["fedavg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup"]


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in folder.glob("*.json"):
        d = json.loads(p.read_text())
        rows.append({k: d[k] for k in [
            "dataset", "method", "sensitive", "seed",
            "auc_global", "worst_group_auc", "auc_gap", "dp_gap", "eo_gap"
        ]})
    return pd.DataFrame(rows)


def _agg(df, metric):
    g = df.groupby("method")[metric]
    mean = g.mean().reindex(METHOD_ORDER)
    std = g.std().reindex(METHOD_ORDER).fillna(0)
    return mean, std


def plot_perf_vs_fair(df, out: Path):
    mean_auc = df.groupby("method")["auc_global"].mean().reindex(METHOD_ORDER)
    mean_wga = df.groupby("method")["worst_group_auc"].mean().reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in METHOD_ORDER:
        if m not in mean_auc.index or np.isnan(mean_auc[m]):
            continue
        marker = "*" if m == "fedsubgroup" else "o"
        size = 200 if m == "fedsubgroup" else 80
        ax.scatter(mean_auc[m], mean_wga[m], s=size, marker=marker, label=m)
        ax.annotate(m, (mean_auc[m], mean_wga[m]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("AUC global (moyenne)")
    ax.set_ylabel("Worst-group AUC (moyenne)")
    ax.grid(alpha=0.3)
    ax.set_title("Performance globale vs equite (worst-group)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bars(df, metric, out: Path, ylabel: str, title: str):
    mean, std = _agg(df, metric)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHOD_ORDER))
    colors = ["#888"] * 5 + ["#d62728"]   # fedsubgroup en rouge
    ax.bar(x, mean.values, yerr=std.values, color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gaps(df, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    for i, (col, label) in enumerate([("auc_gap", "AUC gap"),
                                       ("dp_gap", "DP gap"),
                                       ("eo_gap", "EO gap")]):
        mean = df.groupby("method")[col].mean().reindex(METHOD_ORDER)
        ax.bar(x + (i - 1) * width, mean.values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=20)
    ax.set_ylabel("Gap (plus bas = plus equitable)")
    ax.set_title("Gaps d'equite par methode")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.results))
    if df.empty:
        print("aucun resultat")
        return

    for ds in df["dataset"].unique():
        for sens in df["sensitive"].unique():
            sub = df[(df["dataset"] == ds) & (df["sensitive"] == sens)]
            if sub.empty:
                continue
            tag = f"{ds}_{sens}"
            plot_perf_vs_fair(sub, out / f"perf_vs_fairness_{tag}.png")
            plot_bars(sub, "worst_group_auc",
                      out / f"bars_worst_group_{tag}.png",
                      "Worst-group AUC", f"Worst-group AUC ({ds}, attr={sens})")
            plot_gaps(sub, out / f"gaps_{tag}.png")
            print(f"  -> figures generees pour {tag}")


if __name__ == "__main__":
    main()
