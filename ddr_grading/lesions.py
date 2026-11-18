"""Vetor de atributos das lesões, a partir das detecções do detector final (JSONL do
scripts/05_train_final.py --stage predict).

Por lesão (MA, EX, SE, HE), 4 valores — 16 no total:
  n      log(1 + nº de detecções)
  area   log(1 + 1000 × soma das áreas das caixas / área do campo de visão)  (‰ da retina)
  cmax   confiança máxima (0 sem detecção)
  cmean  confiança média  (0 sem detecção)
Os logs só comprimem a cauda longa (há imagens com centenas de exsudatos); a MLP recebe o
vetor padronizado com média/desvio do treino de cada fold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LESIONS = ("MA", "EX", "SE", "HE")
FEATURES = [f"{les}_{k}" for les in LESIONS for k in ("n", "area", "cmax", "cmean")]


def lesion_vector(dets, width, height, fov_frac):
    fov_area = max(fov_frac * width * height, 1.0)
    out = []
    for les in LESIONS:
        d = [x for x in dets if x["class"] == les]
        scores = [x["score"] for x in d]
        area = sum(x["bbox_xywh"][2] * x["bbox_xywh"][3] for x in d)
        out += [math.log1p(len(d)), math.log1p(1000 * area / fov_area),
                max(scores, default=0.0), (sum(scores) / len(scores)) if scores else 0.0]
    return out


def load_lesion_features(jsonl_paths, index_rows):
    """{image: np.array(16)} para todas as imagens do índice. Falta de imagem = erro (o vetor
    de uma imagem sem predição seria confundido com 'nenhuma lesão')."""
    meta = {r["image"]: r for r in index_rows}
    feats = {}
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = Path(rec["image"]).name
                if name in meta:
                    m = meta[name]
                    feats[name] = np.array(lesion_vector(rec["detections"], m["width"],
                                                         m["height"], m["fov_frac"]),
                                           dtype=np.float32)
    missing = [n for n in meta if n not in feats]
    if missing:
        raise SystemExit(f"{len(missing)} imagens do índice sem detecções no JSONL "
                         f"(ex.: {missing[:3]}): rode o 05_train_final.py --stage predict")
    return feats
