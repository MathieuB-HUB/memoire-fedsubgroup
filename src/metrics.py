"""Metriques de performance et d'equite.

Convention : tous les calculs prennent en entree des arrays numpy
(y_true, y_score, sensitive) deja concatenes sur tous les clients de test.

- AUC global (macro pour multi-label)
- AUC par sous-groupe sur attribut sensible (sex ou age_bin)
- worst-group AUC = min sur les groupes
- gap min-max d'AUC
- Demographic parity gap : ecart de taux predit positif entre groupes
- Equalized odds gap : moyenne des ecarts de TPR et FPR
- IC bootstrap (1000 resamples) sur n'importe quelle metrique scalaire
- Wilcoxon apparie pour comparer 2 methodes sur k seeds
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC tolerant aux cas degeneres (un seul label dans y_true)."""
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    # multi-label : AUC macro en ignorant les colonnes degenerees
    aucs = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) >= 2:
            aucs.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_group_auc(y_true, y_score, sensitive) -> dict[int, float]:
    """AUC pour chaque valeur unique de l'attribut sensible."""
    out = {}
    for g in np.unique(sensitive):
        mask = sensitive == g
        if mask.sum() < 10:   # trop peu d'echantillons -> NaN
            out[int(g)] = float("nan")
            continue
        out[int(g)] = safe_auc(y_true[mask], y_score[mask])
    return out


def fairness_summary(y_true, y_score, sensitive) -> dict:
    """Bundle des metriques d'equite. y_score = proba (apres sigmoid)."""
    aucs = per_group_auc(y_true, y_score, sensitive)
    valid = [v for v in aucs.values() if not np.isnan(v)]
    worst = float(min(valid)) if valid else float("nan")
    gap = float(max(valid) - min(valid)) if len(valid) >= 2 else float("nan")

    # Pour DP / EO, on binarise au seuil 0.5 (multi-label : moyenne sur classes).
    if y_true.ndim == 1:
        y_pred = (y_score >= 0.5).astype(int)
        dp = _dp_gap(y_pred, sensitive)
        eo = _eo_gap(y_true, y_pred, sensitive)
    else:
        y_pred = (y_score >= 0.5).astype(int)
        dps, eos = [], []
        for j in range(y_true.shape[1]):
            dps.append(_dp_gap(y_pred[:, j], sensitive))
            eos.append(_eo_gap(y_true[:, j], y_pred[:, j], sensitive))
        dp = float(np.nanmean(dps))
        eo = float(np.nanmean(eos))

    return {
        "auc_global": safe_auc(y_true, y_score),
        "auc_per_group": aucs,
        "worst_group_auc": worst,
        "auc_gap": gap,
        "dp_gap": dp,
        "eo_gap": eo,
    }


def _dp_gap(y_pred, sensitive) -> float:
    rates = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        rates.append(y_pred[mask].mean())
    return float(max(rates) - min(rates)) if len(rates) >= 2 else float("nan")


def _eo_gap(y_true, y_pred, sensitive) -> float:
    tprs, fprs = [], []
    for g in np.unique(sensitive):
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        pos, neg = yt == 1, yt == 0
        if pos.sum() > 0:
            tprs.append(yp[pos].mean())
        if neg.sum() > 0:
            fprs.append(yp[neg].mean())
    tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    fpr_gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
    return float((tpr_gap + fpr_gap) / 2)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """IC bootstrap percentile sur une liste de valeurs (par ex. AUC par sample
    apres tirage avec remise). Pour AUC, on fait du resampling au niveau sample :
    appeler avec values = vecteur de scores -- ou plutot utiliser bootstrap_metric."""
    rng = rng or np.random.default_rng(0)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = values[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_metric(metric_fn, n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0, **arrays) -> tuple[float, float]:
    """IC bootstrap sur une metrique calculee a partir d'arrays paralleles.

    Exemple :
        lo, hi = bootstrap_metric(safe_auc, y_true=yt, y_score=ys)
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays.values())))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = {k: v[idx] for k, v in arrays.items()}
        val = metric_fn(**sample)
        if not np.isnan(val):
            boots.append(val)
    if not boots:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_wilcoxon(scores_a: list[float], scores_b: list[float]) -> dict:
    """Wilcoxon signe apparie. Retourne stat, p-value, et diff moyenne."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    diffs = a - b
    if np.all(diffs == 0):
        return {"stat": 0.0, "p": 1.0, "mean_diff": 0.0}
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        # trop peu de points non-nuls
        return {"stat": float("nan"), "p": float("nan"),
                "mean_diff": float(diffs.mean())}
    return {"stat": float(stat), "p": float(p), "mean_diff": float(diffs.mean())}
