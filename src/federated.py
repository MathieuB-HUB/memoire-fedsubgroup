"""Boucle d'apprentissage federe simulee + 6 methodes d'agregation.

Tout est mono-machine : on garde les datasets clients en RAM et on simule
les rounds en boucle. Pas de communication reseau, pas de Flower. Ca rend
FedSubgroup et FedBN plus simples a ecrire (manipulation directe de
state_dict PyTorch).

Methodes :
- fedavg       : moyenne ponderee de tous les parametres
- fedbn        : idem mais BatchNorm gardes locaux
- fedper       : moyenne du tronc, tete locale par client
- fedrep       : alternance tete locale / tronc partage
- ditto        : modele global FedAvg + modele local regularise vers le global
- fedsubgroup  : tronc partage + tetes par sous-groupe (contribution)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ClientData, subgroup_id


@dataclass
class ClientSplit:
    """ClientData decoupe en train/test, avec tenseurs torch deja prets."""
    name: str
    X_tr: torch.Tensor
    y_tr: torch.Tensor
    sg_tr: torch.Tensor       # sous-groupe (sex, age) pour le routage FedSubgroup
    X_te: torch.Tensor
    y_te: torch.Tensor
    sg_te: torch.Tensor
    sex_te: np.ndarray        # gardes en numpy pour les metriques d'equite
    age_te: np.ndarray


def make_splits(clients: list[ClientData], test_frac: float = 0.2,
                seed: int = 0, device: str = "cpu") -> list[ClientSplit]:
    rng = np.random.default_rng(seed)
    out = []
    for c in clients:
        n = len(c)
        idx = rng.permutation(n)
        n_te = max(1, int(round(n * test_frac)))
        te, tr = idx[:n_te], idx[n_te:]
        sg = subgroup_id(c.sex, c.age_bin)
        out.append(ClientSplit(
            name=c.name,
            X_tr=torch.as_tensor(c.X[tr], dtype=torch.float32, device=device),
            y_tr=torch.as_tensor(c.y[tr], dtype=torch.float32, device=device),
            sg_tr=torch.as_tensor(sg[tr], dtype=torch.long, device=device),
            X_te=torch.as_tensor(c.X[te], dtype=torch.float32, device=device),
            y_te=torch.as_tensor(c.y[te], dtype=torch.float32, device=device),
            sg_te=torch.as_tensor(sg[te], dtype=torch.long, device=device),
            sex_te=c.sex[te],
            age_te=c.age_bin[te],
        ))
    return out


def iter_batches(X: torch.Tensor, y: torch.Tensor, sg: torch.Tensor,
                 batch_size: int, shuffle: bool = True):
    n = len(X)
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        sel = idx[i:i + batch_size]
        if len(sel) < 2:   # BatchNorm refuse les batches de taille 1 en train
            continue
        yield X[sel], y[sel], sg[sel]


# Helpers BatchNorm / agregation
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _bn_param_names(model: nn.Module) -> set[str]:
    """Cles de state_dict appartenant a un module BatchNorm."""
    names: set[str] = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, _BN_TYPES):
            prefix = mod_name + "." if mod_name else ""
            for p_name, _ in mod.named_parameters(recurse=False):
                names.add(prefix + p_name)
            for b_name, _ in mod.named_buffers(recurse=False):
                names.add(prefix + b_name)
    return names


def _is_bn_key(k: str, bn_param_names: set[str]) -> bool:
    return k in bn_param_names


def _is_head_key(k: str) -> bool:
    return k.startswith("head") or k.startswith("heads")


def weighted_average(states: list[dict], weights: list[float],
                     keys: list[str] | None = None) -> dict:
    """Moyenne ponderee de state_dicts sur les cles donnees (toutes par defaut)."""
    keys = keys if keys is not None else list(states[0].keys())
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in keys:
        # num_batches_tracked est un Long, on ne le moyenne pas (on prend le premier)
        if "num_batches_tracked" in k:
            out[k] = states[0][k].clone()
            continue
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, wi in zip(states, w):
            acc = acc + wi * s[k].float()
        out[k] = acc.to(states[0][k].dtype)
    return out


def clone_state(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_local(model: nn.Module, split: ClientSplit, lr: float,
                local_epochs: int, batch_size: int,
                use_sg: bool = False,
                prox_target: dict | None = None, prox_mu: float = 0.0) -> nn.Module:
    """Entraine `model` en place sur split.X_tr.

    Si `prox_target` est fourni, ajoute une regularisation L2 vers ces parametres
    (proximal term de Ditto / FedProx).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for xb, yb, sgb in iter_batches(split.X_tr, split.y_tr, split.sg_tr, batch_size):
            opt.zero_grad()
            logits = model(xb, sgb) if use_sg else model(xb)
            if yb.ndim == 1:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), yb)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            if prox_target is not None and prox_mu > 0:
                # Penalite vers les parametres global (Ditto local objective).
                reg = 0.0
                for name, p in model.named_parameters():
                    if name in prox_target:
                        reg = reg + ((p - prox_target[name]) ** 2).sum()
                loss = loss + 0.5 * prox_mu * reg
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, split: ClientSplit, use_sg: bool = False) -> np.ndarray:
    model.eval()
    logits = model(split.X_te, split.sg_te) if use_sg else model(split.X_te)
    return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# Methodes FL : meme signature, renvoient (y_true, y_score, sex, age) concatenes
# ---------------------------------------------------------------------------

def run_fedavg(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    global_model = build_fn()
    sizes = [len(s.X_tr) for s in splits]
    for _ in range(n_rounds):
        local_states = []
        for s in splits:
            m = copy.deepcopy(global_model)
            train_local(m, s, lr, local_epochs, batch_size)
            local_states.append(clone_state(m))
        global_model.load_state_dict(weighted_average(local_states, sizes))
    return _collect_preds(splits, lambda s: predict(global_model, s))


def _augment_with_sg(splits, n_sg=4):
    """Concatene one-hot du sous-groupe aux features. Pour ablation fedavg_sg."""
    out = []
    for s in splits:
        oh_tr = F.one_hot(s.sg_tr, num_classes=n_sg).float()
        oh_te = F.one_hot(s.sg_te, num_classes=n_sg).float()
        out.append(ClientSplit(
            name=s.name,
            X_tr=torch.cat([s.X_tr, oh_tr], dim=1),
            y_tr=s.y_tr, sg_tr=s.sg_tr,
            X_te=torch.cat([s.X_te, oh_te], dim=1),
            y_te=s.y_te, sg_te=s.sg_te,
            sex_te=s.sex_te, age_te=s.age_te,
        ))
    return out


def run_fedavg_sg(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """FedAvg avec sous-groupe one-hot concatene a x. Ablation pour FedSubgroup."""
    return run_fedavg(_augment_with_sg(splits), build_fn,
                      n_rounds, lr, local_epochs, batch_size)


def run_fedbn(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """Comme FedAvg sauf que les parametres de BatchNorm restent locaux."""
    global_model = build_fn()
    sizes = [len(s.X_tr) for s in splits]
    # Etat BN par client (initialise par copie du global).
    bn_set = _bn_param_names(global_model)
    bn_keys = [k for k in global_model.state_dict() if k in bn_set]
    non_bn = [k for k in global_model.state_dict() if k not in bn_set]
    local_bn = [{k: global_model.state_dict()[k].clone() for k in bn_keys}
                for _ in splits]

    for _ in range(n_rounds):
        local_states = []
        for i, s in enumerate(splits):
            m = copy.deepcopy(global_model)
            # On restaure le BN local de ce client avant l'entrainement.
            sd = m.state_dict()
            for k in bn_keys:
                sd[k] = local_bn[i][k]
            m.load_state_dict(sd)
            train_local(m, s, lr, local_epochs, batch_size)
            # On extrait BN pour la prochaine ronde + non-BN pour aggregation.
            state = m.state_dict()
            local_bn[i] = {k: state[k].clone() for k in bn_keys}
            local_states.append(state)
        # Agrege uniquement les cles non-BN.
        avg = weighted_average(local_states, sizes, keys=non_bn)
        sd = global_model.state_dict()
        for k in non_bn:
            sd[k] = avg[k]
        global_model.load_state_dict(sd)

    # Inference : chaque client utilise le global avec son BN local.
    def predict_for(i, s):
        m = copy.deepcopy(global_model)
        sd = m.state_dict()
        for k in bn_keys:
            sd[k] = local_bn[i][k]
        m.load_state_dict(sd)
        return predict(m, s)
    return _collect_preds(splits, predict_for, indexed=True)


def run_fedper(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """Trunk partage, tete locale par client."""
    global_model = build_fn()
    sizes = [len(s.X_tr) for s in splits]
    head_keys = [k for k in global_model.state_dict() if _is_head_key(k)]
    trunk_keys = [k for k in global_model.state_dict() if not _is_head_key(k)]
    local_heads = [{k: global_model.state_dict()[k].clone() for k in head_keys}
                   for _ in splits]

    for _ in range(n_rounds):
        local_states = []
        for i, s in enumerate(splits):
            m = copy.deepcopy(global_model)
            sd = m.state_dict()
            for k in head_keys:
                sd[k] = local_heads[i][k]
            m.load_state_dict(sd)
            train_local(m, s, lr, local_epochs, batch_size)
            state = m.state_dict()
            local_heads[i] = {k: state[k].clone() for k in head_keys}
            local_states.append(state)
        avg = weighted_average(local_states, sizes, keys=trunk_keys)
        sd = global_model.state_dict()
        for k in trunk_keys:
            sd[k] = avg[k]
        global_model.load_state_dict(sd)

    def predict_for(i, s):
        m = copy.deepcopy(global_model)
        sd = m.state_dict()
        for k in head_keys:
            sd[k] = local_heads[i][k]
        m.load_state_dict(sd)
        return predict(m, s)
    return _collect_preds(splits, predict_for, indexed=True)


def run_fedrep(splits, build_fn, n_rounds, lr, local_epochs, batch_size,
               head_epochs: int = 2):
    """Alternance : entraine tete locale (trunk fige) puis trunk (tete figee).

    Seul le trunk est aggrege.
    """
    global_model = build_fn()
    sizes = [len(s.X_tr) for s in splits]
    head_keys = [k for k in global_model.state_dict() if _is_head_key(k)]
    trunk_keys = [k for k in global_model.state_dict() if not _is_head_key(k)]
    local_heads = [{k: global_model.state_dict()[k].clone() for k in head_keys}
                   for _ in splits]

    for _ in range(n_rounds):
        local_states = []
        for i, s in enumerate(splits):
            m = copy.deepcopy(global_model)
            sd = m.state_dict()
            for k in head_keys:
                sd[k] = local_heads[i][k]
            m.load_state_dict(sd)
            # Phase 1 : tete seulement.
            for p in _params_by_prefix(m, "head"):
                p.requires_grad_(True)
            for n_, p in m.named_parameters():
                p.requires_grad_(_is_head_key(n_))
            train_local(m, s, lr, head_epochs, batch_size)
            # Phase 2 : tronc seulement.
            for n_, p in m.named_parameters():
                p.requires_grad_(not _is_head_key(n_))
            train_local(m, s, lr, local_epochs, batch_size)
            state = m.state_dict()
            local_heads[i] = {k: state[k].clone() for k in head_keys}
            local_states.append(state)
        avg = weighted_average(local_states, sizes, keys=trunk_keys)
        sd = global_model.state_dict()
        for k in trunk_keys:
            sd[k] = avg[k]
        global_model.load_state_dict(sd)

    def predict_for(i, s):
        m = copy.deepcopy(global_model)
        sd = m.state_dict()
        for k in head_keys:
            sd[k] = local_heads[i][k]
        m.load_state_dict(sd)
        return predict(m, s)
    return _collect_preds(splits, predict_for, indexed=True)


def run_ditto(splits, build_fn, n_rounds, lr, local_epochs, batch_size,
              mu: float = 0.1):
    """Global FedAvg en parallele + modele local regularise vers le global."""
    global_model = build_fn()
    local_models = [copy.deepcopy(global_model) for _ in splits]
    sizes = [len(s.X_tr) for s in splits]
    for _ in range(n_rounds):
        # 1) update global via FedAvg classique.
        local_states = []
        for s in splits:
            m = copy.deepcopy(global_model)
            train_local(m, s, lr, local_epochs, batch_size)
            local_states.append(clone_state(m))
        global_model.load_state_dict(weighted_average(local_states, sizes))
        # 2) update modeles locaux avec terme proximal vers global.
        prox_target = dict(global_model.named_parameters())
        prox_target = {k: v.detach().clone() for k, v in prox_target.items()}
        for i, s in enumerate(splits):
            train_local(local_models[i], s, lr, local_epochs, batch_size,
                        prox_target=prox_target, prox_mu=mu)

    def predict_for(i, s):
        return predict(local_models[i], s)
    return _collect_preds(splits, predict_for, indexed=True)


def run_fedsubgroup(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """FedSubgroup : trunk + tetes par sous-groupe, tout aggrege globalement."""
    global_model = build_fn(multi_head=True)
    sizes = [len(s.X_tr) for s in splits]
    for _ in range(n_rounds):
        local_states = []
        for s in splits:
            m = copy.deepcopy(global_model)
            train_local(m, s, lr, local_epochs, batch_size, use_sg=True)
            local_states.append(clone_state(m))
        global_model.load_state_dict(weighted_average(local_states, sizes))
    return _collect_preds(splits, lambda s: predict(global_model, s, use_sg=True))


def _params_by_prefix(model: nn.Module, prefix: str):
    return [p for n, p in model.named_parameters() if n.startswith(prefix)]


def _collect_preds(splits: list[ClientSplit], predict_fn, indexed: bool = False):
    """Concatene les predictions sur le test pool de tous les clients."""
    y_true_all, y_score_all, sex_all, age_all = [], [], [], []
    for i, s in enumerate(splits):
        scores = predict_fn(i, s) if indexed else predict_fn(s)
        y_true_all.append(s.y_te.cpu().numpy())
        y_score_all.append(scores)
        sex_all.append(s.sex_te)
        age_all.append(s.age_te)
    return (np.concatenate(y_true_all), np.concatenate(y_score_all),
            np.concatenate(sex_all), np.concatenate(age_all))


# ---------------------------------------------------------------------------
# Baselines non-federees (bornes d'encadrement)
# ---------------------------------------------------------------------------

def run_centralized(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """Borne superieure (oracle) : tous les patients pooles, un seul modele.
    Budget total = n_rounds * local_epochs pour matcher les methodes FL."""
    model = build_fn()
    X_tr = torch.cat([s.X_tr for s in splits], dim=0)
    y_tr = torch.cat([s.y_tr for s in splits], dim=0)
    sg_tr = torch.cat([s.sg_tr for s in splits], dim=0)
    # On fabrique un ClientSplit virtuel "pool" pour reutiliser train_local.
    pool = ClientSplit(
        name="pool", X_tr=X_tr, y_tr=y_tr, sg_tr=sg_tr,
        X_te=X_tr[:1], y_te=y_tr[:1], sg_te=sg_tr[:1],
        sex_te=np.zeros(1), age_te=np.zeros(1),
    )
    total_epochs = n_rounds * local_epochs
    train_local(model, pool, lr, total_epochs, batch_size)
    return _collect_preds(splits, lambda s: predict(model, s))


def run_local_only(splits, build_fn, n_rounds, lr, local_epochs, batch_size):
    """Borne inferieure : chaque hopital entraine son propre modele.

    Pas de communication. Budget de calcul total identique aux methodes
    federees (n_rounds * local_epochs epoques locales par client).
    """
    local_models = []
    total_epochs = n_rounds * local_epochs
    for s in splits:
        m = build_fn()
        train_local(m, s, lr, total_epochs, batch_size)
        local_models.append(m)

    def predict_for(i, s):
        return predict(local_models[i], s)
    return _collect_preds(splits, predict_for, indexed=True)


# Dispatcher pour run.py
METHODS = {
    "fedavg": run_fedavg,
    "fedavg_sg": run_fedavg_sg,
    "fedbn": run_fedbn,
    "fedper": run_fedper,
    "fedrep": run_fedrep,
    "ditto": run_ditto,
    "fedsubgroup": run_fedsubgroup,
    "centralized": run_centralized,
    "local_only": run_local_only,
}
