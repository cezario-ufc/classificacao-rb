"""Índice, folds e cache de imagens do subconjunto de graduação do DDR.

- Índice: DR_grading/{train,valid,test}.txt reunidos (todas as 6 classes, inclusive a 5 =
  não graduável), SEM as 757 imagens anotadas: o detector final treinou nelas, e o vetor de
  lesões dessas imagens viria de imagens vistas no treino.
- Folds: o mesmo protocolo da detecção (ddr_sahi.folds.make_folds) — 5 partições × 3
  repetições, holdout interno de 20% do treino como val, seed 42 — com estratificação simples
  pelo grau (64% treino / 16% val / 20% teste).
- Cache: cada imagem é recortada no campo de visão (retina), completada até ficar quadrada e
  reduzida a CACHE_SIZE px. Ler as originais (até 5184×3456) a cada época prenderia o treino
  na CPU. A fração da imagem ocupada pela retina (fov_frac) vai para o índice: normaliza a
  área das lesões no vetor de atributos.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DDR_GRADING = REPO_ROOT / "data" / "raw" / "oia-ddr" / "DDR-dataset" / "DR_grading"
GRADING_DIR = REPO_ROOT / "data" / "grading"
INDEX_CSV = GRADING_DIR / "index.csv"
IMAGES_TXT = GRADING_DIR / "images.txt"      # entrada do 05_train_final.py --stage predict
CACHE_SIZE = 512
CACHE_DIR = GRADING_DIR / f"img{CACHE_SIZE}"

CLASS_NAMES = {0: "Sem RD", 1: "RDNP leve", 2: "RDNP moderada", 3: "RDNP grave", 4: "RDP",
               5: "Não graduável"}
FOV_THRESHOLD = 15        # nível (0-255) do canal mais claro acima do qual o pixel é retina
FOV_PROBE = 512           # lado da miniatura usada para achar o campo de visão


def read_split_lists(root: Path = DDR_GRADING):
    """[(image, path, label, split_origem)] dos três .txt oficiais."""
    rows = []
    for split in ("train", "valid", "test"):
        with open(root / f"{split}.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    name, label = line.rsplit(None, 1)
                    rows.append((name, root / split / name, int(label), split))
    names = [r[0] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("nomes de imagem repetidos entre os splits do DR_grading")
    return rows


def fov_box(img: Image.Image):
    """(caixa do campo de visão em pixels da original, fração da imagem ocupada pela retina)."""
    w, h = img.size
    s = FOV_PROBE / max(w, h)
    small = np.asarray(img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s)))))
    mask = small.max(axis=2) > FOV_THRESHOLD
    if mask.sum() < 0.05 * mask.size:      # imagem quase toda escura: usa a imagem inteira
        return (0, 0, w, h), 1.0
    ys, xs = np.nonzero(mask)
    box = (int(xs.min() / s), int(ys.min() / s),
           min(w, int(np.ceil((xs.max() + 1) / s))), min(h, int(np.ceil((ys.max() + 1) / s))))
    return box, float(mask.mean())


def square_resize(img: Image.Image, box, size: int):
    """Recorta a caixa, completa com preto até ficar quadrada (sem distorcer) e redimensiona."""
    crop = img.convert("RGB").crop(box)
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def prepare_one(args):
    """Processa uma imagem (worker do multiprocessing): grava o cache e devolve os metadados."""
    name, path, label, split = args
    out = CACHE_DIR / name
    with Image.open(path) as img:
        w, h = img.size
        box, fov_frac = fov_box(img)
        if not out.exists():
            square_resize(img, box, CACHE_SIZE).save(out, quality=95)
    return {"image": name, "path": str(path), "label": label, "split_origem": split,
            "width": w, "height": h, "fov_frac": round(fov_frac, 5)}


def load_index():
    """Lista de dicts do índice (gerado por scripts/06_grading_prepare.py)."""
    if not INDEX_CSV.exists():
        raise SystemExit(f"{INDEX_CSV} não existe: rode scripts/06_grading_prepare.py")
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["width"], r["height"] = int(r["width"]), int(r["height"])
        r["fov_frac"] = float(r["fov_frac"])
    return rows


def make_grading_folds(images, labels, n_splits=5, n_repeats=3, val_size=0.2, seed=42):
    """Mesmo esquema de ddr_sahi.folds.make_folds, estratificado pelo grau."""
    images, labels = np.asarray(images), np.asarray(labels)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = []
    for i, (train_idx, test_idx) in enumerate(rskf.split(images, labels)):
        repeat, fold = divmod(i, n_splits)
        inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_rel, val_rel = next(inner.split(images[train_idx], labels[train_idx]))
        splits.append({
            "repeat": repeat, "fold": fold,
            "train": images[train_idx[tr_rel]].tolist(),
            "val": images[train_idx[val_rel]].tolist(),
            "test": images[test_idx].tolist(),
        })
    return splits
