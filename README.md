# Mémoire M2 — FedSubgroup

Code expérimental du mémoire *Apprentissage fédéré en santé : impact de la profondeur de personnalisation sur l'équité algorithmique en contexte non-IID* (Mathieu Baleydier, M2 MIAGE, Paris Nanterre, juin 2026).

## Quickstart

```bash
uv venv && uv sync
python experiments/grid_fed_heart.py     # grille principale (~30 min, GPU optionnel)
python experiments/stress_dirichlet.py   # stress test Dirichlet
python experiments/figures.py            # figures du mémoire
```

Dataset Fed-Heart-Disease inclus dans `data/fed_heart/` (UCI, libre).

## Structure

- `src/` — méthodes FL (FedAvg, FedBN, FedPer, FedRep, Ditto, FedSubgroup), métriques, données.
- `experiments/` — scripts de grille, stress test, agrégation, figures.
- `configs/` — YAML d'hyperparamètres.
- `results/` — JSON par run + CSV agrégés.
- `tests/` — sanity checks.
