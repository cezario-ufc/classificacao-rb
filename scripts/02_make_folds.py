"""Etapa 3 (parte 1) — gera os folds (5x3, estratificado), materializa os .txt/data.yaml
e roda checagens de integridade (anti-vazamento) e de balanço da estratificação.

Uso:
    python scripts/02_make_folds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddr_sahi.folds import (  # noqa: E402
    CLASS_NAMES,
    LESION_COLS,
    build_fold_dirs,
    load_manifest,
    make_folds,
)

N_SPLITS, N_REPEATS = 5, 3


def check_integrity(splits, images):
    """Falha (assert) se houver vazamento; devolve nada."""
    all_imgs = set(images.tolist())
    for s in splits:
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        tf = set(s["train_full"])
        tag = f"r{s['repeat']}_f{s['fold']}"
        assert tr.isdisjoint(va), f"{tag}: train∩val não vazio (vazamento)"
        assert tr.isdisjoint(te), f"{tag}: train∩test não vazio (vazamento)"
        assert va.isdisjoint(te), f"{tag}: val∩test não vazio (vazamento)"
        assert tf == tr | va, f"{tag}: train_full != train∪val"
        assert tf.isdisjoint(te), f"{tag}: train_full∩test não vazio (vazamento)"
        assert tf | te == all_imgs, f"{tag}: train_full∪test != todas as imagens"

    # cada imagem no teste exatamente 1x por repetição (partição por repeat)
    for rep in range(N_REPEATS):
        counts = {}
        for s in splits:
            if s["repeat"] != rep:
                continue
            for img in s["test"]:
                counts[img] = counts.get(img, 0) + 1
        bad = {k: v for k, v in counts.items() if v != 1}
        assert not bad, f"repeat {rep}: imagens fora de 'exatamente 1x no teste': {list(bad)[:3]}"
        assert set(counts) == all_imgs, f"repeat {rep}: nem toda imagem entrou no teste"
    print("[OK] integridade: sem vazamento train/val/test; teste = partição por repetição")


def report_stratification(splits, images, Y):
    """Distribuição de imagens POSITIVAS por classe entre os folds de teste (repetição 0)."""
    col = {c: i for i, c in enumerate(LESION_COLS)}
    idx = {img: i for i, img in enumerate(images)}
    print(f"\nBalanço da estratificação (nº de imagens positivas no TESTE, repetição 0, {N_SPLITS} folds):")
    header = "fold |" + "".join(f" {CLASS_NAMES[c]:>4}" for c in CLASS_NAMES) + " | total"
    print(header)
    print("-" * len(header))
    for s in sorted([s for s in splits if s["repeat"] == 0], key=lambda s: s["fold"]):
        per = {c: 0 for c in CLASS_NAMES}
        for img in s["test"]:
            row = Y[idx[img]]
            for cid, name in CLASS_NAMES.items():
                per[cid] += int(row[col[f"has_{name}"]])
        line = f"{s['fold']:>4} |" + "".join(f" {per[c]:>4}" for c in CLASS_NAMES)
        print(line + f" | {len(s['test']):>5}")
    # total de imagens SE-positivas e min por fold (o ponto crítico)
    se_per_fold = []
    for s in [s for s in splits if s["repeat"] == 0]:
        se_per_fold.append(sum(int(Y[idx[i]][col["has_SE"]]) for i in s["test"]))
    print(f"\nSE por fold (rep 0): min={min(se_per_fold)}, max={max(se_per_fold)} "
          f"(total SE-positivas={int(Y[:, col['has_SE']].sum())}). "
          f"{'OK: nenhum fold sem SE' if min(se_per_fold) > 0 else 'ALERTA: fold sem SE!'}")


def main():
    images, Y = load_manifest()
    print(f"Imagens no pool: {len(images)}  |  presença de lesão (soma por classe): "
          + ", ".join(f"{CLASS_NAMES[i]}={int(Y[:, i].sum())}" for i in CLASS_NAMES))

    splits = make_folds(images, Y, n_splits=N_SPLITS, n_repeats=N_REPEATS)
    print(f"Splits gerados: {len(splits)} ({N_SPLITS} folds × {N_REPEATS} repetições)")

    check_integrity(splits, images)
    report_stratification(splits, images, Y)

    # materializa os arquivos de todos os folds
    for s in splits:
        build_fold_dirs(s)
    print(f"\n[OK] materializados train/val/test + data.yaml de {len(splits)} folds em runs/folds/")


if __name__ == "__main__":
    main()
