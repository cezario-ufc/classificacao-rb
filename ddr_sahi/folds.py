"""Etapa 3 (parte 1) — geração dos folds da validação cruzada repetida e estratificada.

Unidade de divisão = IMAGEM (as 757 anotadas). Usa RepeatedMultilabelStratifiedKFold
(5 folds × 3 repetições = 15 splits) estratificado pelo vetor de presença de lesão
(has_MA/EX/SE/HE) do manifest — essencial dado o desbalanceamento (SE em só 239/757 imagens).

Para cada split externo, gera também um holdout interno (val) estratificado a partir do treino,
usado para early-stopping do treino (Etapa 3 parte 2).

Não materializa pastas: escreve, por fold, três arquivos .txt com caminhos de imagem
(train/val/test) + um data.yaml. O Ultralytics resolve as labels trocando 'images/'->'labels/'.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    RepeatedMultilabelStratifiedKFold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = REPO_ROOT / "data" / "yolo"
FOLDS_DIR = REPO_ROOT / "runs" / "folds"          # gitignored (runs/)
LESION_COLS = ["has_MA", "has_EX", "has_SE", "has_HE"]
CLASS_NAMES = {0: "MA", 1: "EX", 2: "SE", 3: "HE"}


def load_manifest(manifest_path: Path | None = None):
    """Retorna (images: np.ndarray[str], Y: np.ndarray[N,4] presença de lesão)."""
    manifest_path = manifest_path or (YOLO_DIR / "manifest.csv")
    rows = list(csv.DictReader(open(manifest_path)))
    images = np.array([r["image"] for r in rows])
    Y = np.array([[int(r[c]) for c in LESION_COLS] for r in rows], dtype=int)
    return images, Y


def make_folds(images, Y, n_splits=5, n_repeats=3, val_size=0.2, seed=42):
    """Gera os 15 splits (5 folds × 3 repetições). Cada item: dict com repeat, fold,
    train, val, test (nomes de imagem). 15 pares bastam para o Wilcoxon (mínimo 6)."""
    rmskf = RepeatedMultilabelStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits = []
    for i, (train_idx, test_idx) in enumerate(rmskf.split(images, Y)):
        repeat, fold = divmod(i, n_splits)
        # holdout interno estratificado (val) a partir do treino do fold
        inner = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        tr_rel, val_rel = next(inner.split(images[train_idx], Y[train_idx]))
        train_div_idx = train_idx[tr_rel]
        val_idx = train_idx[val_rel]
        splits.append({
            "repeat": repeat,
            "fold": fold,
            "train": images[train_div_idx].tolist(),   # treino do GridSearch
            "val": images[val_idx].tolist(),           # validação (escolha de hiperparâmetro)
            "test": images[test_idx].tolist(),         # teste do fold (métrica reportada)
            "train_full": images[train_idx].tolist(),  # train+val: re-treino final do fold
        })
    return splits


def _write_list(txt_path: Path, names):
    """Escreve caminhos ABSOLUTOS das imagens (um por linha)."""
    with open(txt_path, "w") as f:
        for n in names:
            f.write(str((YOLO_DIR / "images" / n).resolve()) + "\n")


def build_fold_dirs(split, base_dir: Path | None = None):
    """Materializa os .txt (train/val/test/train_full) e o data.yaml de um fold.

    Retorna o caminho do data.yaml (val) e do data.yaml (train_full) para uso no treino.
    """
    base_dir = base_dir or FOLDS_DIR
    name = f"r{split['repeat']}_f{split['fold']}"
    d = base_dir / name
    d.mkdir(parents=True, exist_ok=True)

    for key in ["train", "val", "test", "train_full"]:
        _write_list(d / f"{key}.txt", split[key])

    # data.yaml do GridSearch: treina em train, valida em val
    grid_yaml = d / "data_grid.yaml"
    with open(grid_yaml, "w") as f:
        yaml.safe_dump({
            "train": str((d / "train.txt").resolve()),
            "val": str((d / "val.txt").resolve()),
            "test": str((d / "test.txt").resolve()),
            "names": CLASS_NAMES,
        }, f, sort_keys=False, allow_unicode=True)

    # data.yaml do re-treino final: treina em train_full, avalia em test
    full_yaml = d / "data_full.yaml"
    with open(full_yaml, "w") as f:
        yaml.safe_dump({
            "train": str((d / "train_full.txt").resolve()),
            "val": str((d / "test.txt").resolve()),
            "test": str((d / "test.txt").resolve()),
            "names": CLASS_NAMES,
        }, f, sort_keys=False, allow_unicode=True)

    return {"dir": d, "grid_yaml": grid_yaml, "full_yaml": full_yaml}
