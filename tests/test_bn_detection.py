"""Test : _bn_param_names detecte correctement tous les params BN sur les
deux modeles utilises dans le memoire (MLP Fed-Heart, ResNet1D PTB-XL).

Bug initial : `"bn" in k` ratait les BN nommees par index, ex `net.1.weight`
(MLP) et `stem.1.weight`, `shortcut.1.weight` (ResNet1D). FedBN se comportait
alors comme FedAvg sur ces parametres.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn as nn

from src.federated import _bn_param_names
from src.models import FedModel, MLPTrunk, ResNet1DTrunk


def _expected_bn_keys(model: nn.Module) -> set[str]:
    """Reference oracle : ce qu'on attend, calcule via le meme mecanisme
    que la prod (mais reverifie ici en parcourant les modules)."""
    out: set[str] = set()
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for mod_name, mod in model.named_modules():
        if isinstance(mod, bn_types):
            prefix = mod_name + "." if mod_name else ""
            for pn, _ in mod.named_parameters(recurse=False):
                out.add(prefix + pn)
            for bn, _ in mod.named_buffers(recurse=False):
                out.add(prefix + bn)
    return out


def test_bn_detection_mlp():
    trunk = MLPTrunk(13, [32, 16], out_dim=16)
    model = FedModel(trunk, n_classes=1)
    bn_set = _bn_param_names(model)
    sd_keys = set(model.state_dict().keys())

    # 1) tous les BN attendus sont detectes
    expected = _expected_bn_keys(model)
    assert bn_set == expected, f"diff: {bn_set ^ expected}"

    # 2) les BN par index sont bien attrapes
    assert "trunk.net.1.weight" in bn_set
    assert "trunk.net.1.bias" in bn_set
    assert "trunk.net.1.running_mean" in bn_set
    assert "trunk.net.1.running_var" in bn_set
    assert "trunk.net.1.num_batches_tracked" in bn_set
    assert "trunk.net.4.weight" in bn_set

    # 3) les Linear ne sont PAS classes BN
    assert "trunk.net.0.weight" not in bn_set
    assert "trunk.net.0.bias" not in bn_set
    assert "trunk.net.3.weight" not in bn_set
    assert "trunk.net.6.weight" not in bn_set
    assert "head.fc.weight" not in bn_set
    assert "head.fc.bias" not in bn_set

    # 4) tous les bn_set sont bien des cles du state_dict
    assert bn_set.issubset(sd_keys)


def test_bn_detection_resnet1d():
    trunk = ResNet1DTrunk(in_channels=12, base=16, out_dim=64)
    model = FedModel(trunk, n_classes=5)
    bn_set = _bn_param_names(model)
    sd_keys = set(model.state_dict().keys())

    expected = _expected_bn_keys(model)
    assert bn_set == expected, f"diff: {bn_set ^ expected}"

    # cas par index (stem, shortcut)
    assert "trunk.stem.1.weight" in bn_set
    assert "trunk.stem.1.running_mean" in bn_set
    assert "trunk.layer2.shortcut.1.weight" in bn_set
    assert "trunk.layer3.shortcut.1.running_var" in bn_set

    # cas nommes (bn1, bn2)
    assert "trunk.layer1.bn1.weight" in bn_set
    assert "trunk.layer3.bn2.num_batches_tracked" in bn_set

    # convs et fc ne doivent PAS etre BN
    assert "trunk.stem.0.weight" not in bn_set
    assert "trunk.layer1.conv1.weight" not in bn_set
    assert "trunk.layer2.shortcut.0.weight" not in bn_set
    assert "trunk.fc.weight" not in bn_set
    assert "head.fc.weight" not in bn_set

    assert bn_set.issubset(sd_keys)
