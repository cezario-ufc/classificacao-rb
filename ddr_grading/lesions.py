"""Vetor de atributos das lesões, a partir das detecções do detector final (JSONL do
scripts/05_train_final.py --stage predict).

Por lesão (MA, EX, SE, HE), 4 valores — 16 no total:
  n      log(1 + nº de detecções)
  area   log(1 + 1000 × soma das áreas das caixas / área do campo de visão)  (‰ da retina)
  cmax   confiança máxima (0 sem detecção)
  cmean  confiança média  (0 sem detecção)
Os logs só comprimem a cauda longa (há imagens com centenas de exsudatos); a MLP recebe o
vetor padronizado com média/desvio do treino de cada fold.

min_score: só conta detecções com score >= min_score (o JSONL vem com conf 0.1, o ponto de
operação do mAP; para graduar, EX/HE de baixa confiança empurram o grau leve para moderada).
lesion_vector_multi: n e área em vários limiares + cmax/cmean + indicador "só MA" por limiar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LESIONS = ("MA", "EX", "SE", "HE")
FEATURES = [f"{les}_{k}" for les in LESIONS for k in ("n", "area", "cmax", "cmean")]


def _count_area(d, fov_area):
    area = sum(x["bbox_xywh"][2] * x["bbox_xywh"][3] for x in d)
    return [math.log1p(len(d)), math.log1p(1000 * area / fov_area)]


def lesion_vector(dets, width, height, fov_frac, min_score=0.0):
    fov_area = max(fov_frac * width * height, 1.0)
    out = []
    for les in LESIONS:
        d = [x for x in dets if x["class"] == les and x["score"] >= min_score]
        scores = [x["score"] for x in d]
        out += _count_area(d, fov_area) + [max(scores, default=0.0),
                                           (sum(scores) / len(scores)) if scores else 0.0]
    return out


def lesion_vector_multi(dets, width, height, fov_frac, thresholds=(0.1, 0.3, 0.5)):
    """Por lesão: n e área em cada limiar, + cmax e cmean de todas; por limiar: 1 se há MA e
    nenhuma EX/SE/HE (o padrão do grau leve). 4·(2·T + 2) + T valores."""
    fov_area = max(fov_frac * width * height, 1.0)
    out = []
    for les in LESIONS:
        d = [x for x in dets if x["class"] == les]
        for t in thresholds:
            out += _count_area([x for x in d if x["score"] >= t], fov_area)
        scores = [x["score"] for x in d]
        out += [max(scores, default=0.0), (sum(scores) / len(scores)) if scores else 0.0]
    for t in thresholds:
        n = {les: sum(1 for x in dets if x["class"] == les and x["score"] >= t) for les in LESIONS}
        out.append(float(n["MA"] > 0 and n["EX"] + n["SE"] + n["HE"] == 0))
    return out


def load_detections(jsonl_paths, index_rows):
    """{image: (detecções, width, height, fov_frac)} para todas as imagens do índice. Falta de
    imagem = erro (o vetor de uma imagem sem predição seria confundido com 'nenhuma lesão')."""
    meta = {r["image"]: r for r in index_rows}
    out = {}
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = Path(rec["image"]).name
                if name in meta:
                    m = meta[name]
                    out[name] = (rec["detections"], m["width"], m["height"], m["fov_frac"])
    missing = [n for n in meta if n not in out]
    if missing:
        raise SystemExit(f"{len(missing)} imagens do índice sem detecções no JSONL "
                         f"(ex.: {missing[:3]}): rode o 05_train_final.py --stage predict")
    return out


def featurize(dets_by_image, fn=lesion_vector, **kw):
    """{image: np.array} aplicando fn(dets, width, height, fov_frac, **kw)."""
    return {n: np.array(fn(*v, **kw), dtype=np.float32) for n, v in dets_by_image.items()}


def load_lesion_features(jsonl_paths, index_rows):
    """{image: np.array(16)} — o vetor original (todas as detecções do JSONL)."""
    return featurize(load_detections(jsonl_paths, index_rows))
