"""Modeles : MLP pour Fed-Heart, ResNet1D pour PTB-XL.

Chaque modele est construit comme (trunk, head). Les methodes FedPer/FedRep/
FedSubgroup ont besoin de cette separation pour aggreger differemment trunk
et tete. Pour FedSubgroup, on remplace la tete unique par un ModuleDict
de tetes indexees par sous-groupe demographique (sg = 2*sex + age_bin).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP (Fed-Heart-Disease, tabulaire binaire)
# ---------------------------------------------------------------------------

class MLPTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int = 16):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class LinearHead(nn.Module):
    """Tete lineaire generique : trunk_out -> n_classes logits."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


# ---------------------------------------------------------------------------
# ResNet1D (PTB-XL, ECG 12 derivations, 100Hz, 1000 echantillons)
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=7, stride=stride,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet1DTrunk(nn.Module):
    """Petit ResNet1D ~100k parametres. Tient en VRAM sur RTX 4070 Ti Super."""
    def __init__(self, in_channels: int = 12, base: int = 16, out_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer1 = BasicBlock1D(base, base, stride=1)
        self.layer2 = BasicBlock1D(base, base * 2, stride=2)
        self.layer3 = BasicBlock1D(base * 2, base * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Wrapper unifie trunk + tete (mono-tete ou multi-tete subgroup)
# ---------------------------------------------------------------------------

class FedModel(nn.Module):
    """Modele FL standard : trunk + une seule tete partagee."""
    def __init__(self, trunk: nn.Module, n_classes: int):
        super().__init__()
        self.trunk = trunk
        self.head = LinearHead(trunk.out_dim, n_classes)

    def forward(self, x, sg=None):  # sg ignore en mono-tete
        return self.head(self.trunk(x))


class SubgroupModel(nn.Module):
    """Modele FedSubgroup : trunk partage + une tete par sous-groupe demographique.

    sg dans {0,1,2,3} (= 2*sex + age_bin). En forward, on route chaque sample
    vers sa tete. Pour rester vectorise, on calcule les 4 sorties et on selectionne.
    """
    def __init__(self, trunk: nn.Module, n_classes: int, n_subgroups: int = 4):
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleList([
            LinearHead(trunk.out_dim, n_classes) for _ in range(n_subgroups)
        ])
        self.n_subgroups = n_subgroups
        self.n_classes = n_classes

    def forward(self, x, sg):
        z = self.trunk(x)
        # Pour eviter une boucle Python, on calcule chaque tete sur tout le batch
        # puis on gather selon sg. Cout : 4x mais c'est peu cher (head est lineaire).
        outs = torch.stack([h(z) for h in self.heads], dim=1)   # (B, K, C)
        idx = sg.view(-1, 1, 1).expand(-1, 1, self.n_classes)
        return outs.gather(1, idx).squeeze(1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, in_dim_or_channels: int, n_classes: int,
                multi_head: bool = False) -> nn.Module:
    if cfg["model"] == "mlp":
        trunk = MLPTrunk(in_dim_or_channels, cfg["hidden_dims"], out_dim=16)
    elif cfg["model"] == "resnet1d":
        trunk = ResNet1DTrunk(in_channels=in_dim_or_channels, out_dim=64)
    else:
        raise ValueError(f"modele inconnu: {cfg['model']}")

    if multi_head:
        return SubgroupModel(trunk, n_classes)
    return FedModel(trunk, n_classes)
