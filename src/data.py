"""Chargement Fed-Heart-Disease et PTB-XL en format federe.

Convention : chaque dataset renvoie une liste de ClientData (un par site).
Les attributs sensibles (sexe, age_bin) sont conserves pour l'evaluation
d'equite et pour le routage FedSubgroup.

Layout attendu sous data/ :
  data/fed_heart/processed.{cleveland,hungarian,switzerland,va}.data
  data/ptbxl/ptbxl_database.csv
  data/ptbxl/scp_statements.csv
  data/ptbxl/records100/...
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


# ---------------------------------------------------------------------------
# Structure commune
# ---------------------------------------------------------------------------

@dataclass
class ClientData:
    """Un client federe : features, labels, attributs sensibles."""
    name: str
    X: np.ndarray             # (n, d) tabulaire ou (n, 12, T) ECG
    y: np.ndarray             # (n,) binaire ou (n, n_classes) multi-label
    sex: np.ndarray           # (n,) {0, 1}
    age_bin: np.ndarray       # (n,) bin d'age, 0 = jeune, 1 = age

    def __len__(self) -> int:
        return len(self.y)


# ---------------------------------------------------------------------------
# Fed-Heart-Disease (UCI, 4 sites)
# ---------------------------------------------------------------------------

_HEART_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
_HEART_SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
# Seuil median agrege Cleveland, fixe a priori (voir memoire §5.1).
_AGE_THRESHOLD = 54


def _load_heart_site(path: Path, name: str) -> ClientData:
    df = pd.read_csv(path, header=None, names=_HEART_COLS, na_values="?")
    # Imputation mediane par site (les fichiers UCI ont beaucoup de NaN
    # sur Switzerland et VA). Mediane = compromis robuste sans deps lourdes.
    df = df.fillna(df.median(numeric_only=True))
    # Target d'origine = 0..4 (severite). FLamby et la litterature reduisent
    # a binaire : 0 = sain, >=1 = maladie presente.
    y = (df["target"].values > 0).astype(np.float32)
    sex = df["sex"].values.astype(np.int64)
    age_bin = (df["age"].values >= _AGE_THRESHOLD).astype(np.int64)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    return ClientData(name=name, X=X, y=y, sex=sex, age_bin=age_bin)


def load_fed_heart(data_dir: str | Path) -> list[ClientData]:
    """Charge les 4 sites Heart-Disease. Meme partition que FLamby."""
    data_dir = Path(data_dir)
    clients: list[ClientData] = []
    for name, fname in _HEART_SITES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Telecharger depuis "
                f"https://archive.ics.uci.edu/dataset/45/heart+disease"
            )
        clients.append(_load_heart_site(path, name))
    return clients


def standardize_clients(clients: list[ClientData]) -> list[ClientData]:
    """Standardise les features avec mu/sigma globaux (cf. memoire §5.1)."""
    X_all = np.vstack([c.X for c in clients])
    mu, sigma = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
    return [
        ClientData(
            name=c.name, X=(c.X - mu) / sigma, y=c.y,
            sex=c.sex, age_bin=c.age_bin,
        )
        for c in clients
    ]


# ---------------------------------------------------------------------------
# PTB-XL (PhysioNet, ECG 12 derivations)
# ---------------------------------------------------------------------------

# 5 superclasses diagnostiques officielles de PTB-XL.
_PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _aggregate_superclass(scp_dict: dict, scp_df: pd.DataFrame) -> list[str]:
    """Convertit le dict de codes SCP-ECG en superclasses diagnostiques."""
    out = set()
    for code in scp_dict:
        if code in scp_df.index:
            sc = scp_df.loc[code, "diagnostic_class"]
            if isinstance(sc, str):
                out.add(sc)
    return sorted(out)


def load_ptbxl(
    data_dir: str | Path,
    sampling_rate: int = 100,
    min_records_per_site: int = 200,
) -> list[ClientData]:
    """Charge PTB-XL, split federe par site de recueil.

    Renvoie une liste de ClientData avec X de forme (n, 12, T),
    T = 1000 a 100Hz (10s d'enregistrement).

    Les sites avec moins de `min_records_per_site` enregistrements sont
    fusionnes dans un client "small_sites" pour eviter des clients
    pathologiquement petits qui plombent la convergence FL.
    """
    data_dir = Path(data_dir)
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]

    db["superclass"] = db["scp_codes"].apply(
        lambda d: _aggregate_superclass(d, scp_df)
    )
    # On retire les ECG sans label diagnostique exploitable.
    db = db[db["superclass"].apply(len) > 0]

    # Construction du label multi-label (n, 5).
    y_full = np.zeros((len(db), len(_PTBXL_SUPERCLASSES)), dtype=np.float32)
    for i, classes in enumerate(db["superclass"].values):
        for c in classes:
            j = _PTBXL_SUPERCLASSES.index(c)
            y_full[i, j] = 1.0

    # Regroupement des petits sites.
    site_counts = db["site"].value_counts()
    big_sites = site_counts[site_counts >= min_records_per_site].index
    db["client"] = db["site"].where(db["site"].isin(big_sites), other=-1)

    # Lecture des signaux. Colonne filename_lr pour 100Hz, filename_hr pour 500Hz.
    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    clients: list[ClientData] = []
    for client_id, sub in db.groupby("client"):
        idx = sub.index.values
        # Signaux : (n, T, 12) en sortie wfdb, on transpose en (n, 12, T).
        signals = np.stack([
            wfdb.rdsamp(str(data_dir / sub.loc[i, fname_col]))[0]
            for i in idx
        ]).astype(np.float32).transpose(0, 2, 1)

        sex = sub["sex"].values.astype(np.int64)         # 0 = male, 1 = female
        ages = sub["age"].values
        # Seuil 60 ans : retient le passage "age >= 60" souvent utilise comme
        # cohorte a risque en cardiologie. Choix fige a priori.
        age_bin = (ages >= 60).astype(np.int64)

        rows_in_db = [list(db.index).index(i) for i in idx]
        y = y_full[rows_in_db]

        name = f"site_{int(client_id)}" if client_id != -1 else "small_sites"
        clients.append(ClientData(
            name=name, X=signals, y=y, sex=sex, age_bin=age_bin,
        ))
    return clients


# ---------------------------------------------------------------------------
# Helpers de partitionnement par sous-groupe (pour FedSubgroup)
# ---------------------------------------------------------------------------

def subgroup_id(sex: np.ndarray, age_bin: np.ndarray) -> np.ndarray:
    """Encode le couple (sexe, age_bin) en un entier dans {0, 1, 2, 3}.

    Convention : sg = 2 * sex + age_bin. Stable, utilise pour le routage
    vers les tetes par sous-groupe.
    """
    return (2 * sex + age_bin).astype(np.int64)
