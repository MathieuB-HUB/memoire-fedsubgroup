"""Ajouts PTB-XL post-audit : (a) bornes centralized + local_only,
(b) trace de convergence (AUC global vs round) pour 3 methodes.

Charge PTB-XL une seule fois (~8 min de lecture wfdb) puis enchaine. Les
bornes passent par run_one (code audite, reprenable : saute les JSON
existants). La convergence reutilise les primitives de src.federated avec
une evaluation tous les `eval_every` rounds -- les boucles repliquent
fidelement run_fedavg / run_fedbn / run_fedsubgroup.

Usage : uv run python experiments/extras_ptbxl.py
"""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run import build_factory, load_dataset, run_one
from src.federated import (
    _bn_param_names, _collect_preds, clone_state, make_splits, predict,
    train_local, weighted_average,
)
from src.metrics import fairness_summary
from src.utils import get_device, load_config

SENS = ["sex", "age"]
SEEDS = list(range(6))
EVAL_EVERY = 5
OUT = "results/ptbxl"


def global_auc(model, splits, use_sg=False):
    yt, ys, sx, _ = _collect_preds(
        splits, lambda s: predict(model, s, use_sg=use_sg))
    return fairness_summary(yt, ys, sx)["auc_global"]


def trace_fedavg(splits, factory, n_rounds, lr, le, bs, log):
    model = factory()
    sizes = [len(s.X_tr) for s in splits]
    for r in range(n_rounds):
        states = []
        for s in splits:
            m = copy.deepcopy(model)
            train_local(m, s, lr, le, bs)
            states.append(clone_state(m))
        model.load_state_dict(weighted_average(states, sizes))
        if (r + 1) % EVAL_EVERY == 0:
            log.append(("fedavg", r + 1, global_auc(model, splits)))
            print(f"  [fedavg] round {r+1}: AUC={log[-1][2]:.4f}")


def trace_fedbn(splits, factory, n_rounds, lr, le, bs, log):
    model = factory()
    sizes = [len(s.X_tr) for s in splits]
    bn_set = _bn_param_names(model)
    bn_keys = [k for k in model.state_dict() if k in bn_set]
    non_bn = [k for k in model.state_dict() if k not in bn_set]
    local_bn = [{k: model.state_dict()[k].clone() for k in bn_keys}
                for _ in splits]
    for r in range(n_rounds):
        states = []
        for i, s in enumerate(splits):
            m = copy.deepcopy(model)
            sd = m.state_dict()
            for k in bn_keys:
                sd[k] = local_bn[i][k]
            m.load_state_dict(sd)
            train_local(m, s, lr, le, bs)
            st = m.state_dict()
            local_bn[i] = {k: st[k].clone() for k in bn_keys}
            states.append(st)
        avg = weighted_average(states, sizes, keys=non_bn)
        sd = model.state_dict()
        for k in non_bn:
            sd[k] = avg[k]
        model.load_state_dict(sd)
        if (r + 1) % EVAL_EVERY == 0:
            def pf(i, s):
                m = copy.deepcopy(model)
                sd = m.state_dict()
                for k in bn_keys:
                    sd[k] = local_bn[i][k]
                m.load_state_dict(sd)
                return predict(m, s)
            yt, ys, sx, _ = _collect_preds(splits, pf, indexed=True)
            auc = fairness_summary(yt, ys, sx)["auc_global"]
            log.append(("fedbn", r + 1, auc))
            print(f"  [fedbn] round {r+1}: AUC={auc:.4f}")


def trace_fedsubgroup(splits, factory, n_rounds, lr, le, bs, log):
    model = factory(multi_head=True)
    sizes = [len(s.X_tr) for s in splits]
    for r in range(n_rounds):
        states = []
        for s in splits:
            m = copy.deepcopy(model)
            train_local(m, s, lr, le, bs, use_sg=True)
            states.append(clone_state(m))
        model.load_state_dict(weighted_average(states, sizes))
        if (r + 1) % EVAL_EVERY == 0:
            log.append(("fedsubgroup", r + 1, global_auc(model, splits, use_sg=True)))
            print(f"  [fedsubgroup] round {r+1}: AUC={log[-1][2]:.4f}")


def main():
    cfg = load_config("configs/ptbxl.yaml")
    cfg["_config_file"] = "configs/ptbxl.yaml"
    device = get_device()
    clients, in_dim, n_classes = load_dataset(cfg)

    # ---- (a) bornes centralized + local_only (audite, reprenable) ----
    print("\n=== (a) bornes centralized + local_only ===")
    out_dir = Path(OUT)
    t0 = time.time()
    for method in ["centralized", "local_only"]:
        for sens in SENS:
            for seed in SEEDS:
                p = out_dir / f"ptbxl_{method}_{sens}_seed{seed}.json"
                if p.exists():
                    print(f"  SKIP {method}/{sens}/seed{seed} (deja fait)")
                    continue
                print(f"  RUN  {method}/{sens}/seed{seed} "
                      f"(ecoule {(time.time()-t0)/60:.1f} min)")
                run_one(cfg, clients, in_dim, n_classes, method, seed, sens,
                        out=OUT, device=device)
    print(f"=== bornes terminees en {(time.time()-t0)/60:.1f} min ===")

    # ---- (b) convergence sur seed 0 ----
    print("\n=== (b) trace de convergence (seed 0) ===")
    splits = make_splits(clients, test_frac=0.2, seed=0, device=str(device))
    factory = build_factory(cfg, 12, n_classes, device)
    nr, lr, le, bs = cfg["n_rounds"], cfg["lr"], cfg["local_epochs"], cfg["batch_size"]
    log: list = []
    from src.utils import set_seed
    set_seed(0); trace_fedavg(splits, factory, nr, lr, le, bs, log)
    set_seed(0); trace_fedbn(splits, factory, nr, lr, le, bs, log)
    set_seed(0); trace_fedsubgroup(splits, factory, nr, lr, le, bs, log)

    conv_path = out_dir / "convergence_ptbxl.csv"
    with open(conv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "round", "auc_global"])
        w.writerows(log)
    print(f"=== convergence -> {conv_path} ===")


if __name__ == "__main__":
    main()
