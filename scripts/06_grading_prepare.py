"""Graduação (parte 1) — índice, cache de imagens e conferência dos folds do DR_grading do DDR.

- Reúne DR_grading/{train,valid,test}.txt (as 6 classes, inclusive a 5 = não graduável).
- Remove as 757 imagens anotadas (data/yolo/manifest.csv): o detector final treinou nelas.
- Recorta cada imagem no campo de visão, completa até quadrada e grava em data/grading/img512/
  (ler as originais a cada época prenderia o treino na CPU); guarda a fração da imagem ocupada
  pela retina (normaliza a área das lesões no vetor de atributos).
- Grava data/grading/index.csv, data/grading/images.txt (entrada do detector) e
  results/grading_folds.csv (atribuição de cada imagem em cada fold, para auditoria).
Idempotente: imagens já no cache não são regravadas.

Uso (na Balerion):
    python scripts/06_grading_prepare.py
    python scripts/05_train_final.py --stage predict --model yolo26n.pt --optimizer MuSGD --images data/grading/images.txt --out results/final_dets_grading.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ddr_grading.data import (  # noqa: E402
    CACHE_DIR,
    CLASS_NAMES,
    GRADING_DIR,
    IMAGES_TXT,
    INDEX_CSV,
    make_grading_folds,
    prepare_one,
    read_split_lists,
)

MANIFEST = REPO_ROOT / "data" / "yolo" / "manifest.csv"
FOLDS_CSV = REPO_ROOT / "results" / "grading_folds.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = read_split_lists()
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        annotated = {Path(r["image"]).stem for r in csv.DictReader(f)}
    kept = [r for r in rows if Path(r[0]).stem not in annotated]
    n_overlap = len(rows) - len(kept)
    print(f"DR_grading: {len(rows)} imagens | {n_overlap} entre as {len(annotated)} anotadas "
          f"(removidas) | restam {len(kept)}")
    if n_overlap != len(annotated):
        print(f"  [aviso] {len(annotated) - n_overlap} imagens anotadas não aparecem no DR_grading")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        meta = list(tqdm(pool.imap(prepare_one, kept, chunksize=16), total=len(kept),
                         desc="cache 512 px"))

    GRADING_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    IMAGES_TXT.write_text("\n".join(m["path"] for m in meta) + "\n", encoding="utf-8")

    counts = Counter(m["label"] for m in meta)
    print("\nGraus (para a Tabela tab:ddr_graus):")
    for c in sorted(counts):
        print(f"  {c} {CLASS_NAMES.get(c, '?'):<15} {counts[c]:>6}  {100 * counts[c] / len(meta):5.1f}%")
    print(f"  total               {len(meta):>6}")
    full = sum(m["fov_frac"] == 1.0 for m in meta)
    if full:
        print(f"  [info] {full} imagens usadas inteiras, sem recorte (fov_frac=1): retina "
              "ocupando o quadro todo, sem borda preta, ou imagem quase toda escura")

    splits = make_grading_folds([m["image"] for m in meta], [m["label"] for m in meta])
    FOLDS_CSV.parent.mkdir(exist_ok=True)
    with open(FOLDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repeat", "fold", "split", "image"])
        for s in splits:
            for part in ("train", "val", "test"):
                w.writerows([s["repeat"], s["fold"], part, n] for n in s[part])
    s0 = splits[0]
    n = len(meta)
    print(f"\nFolds: {len(splits)} (5 × 3) | r0_f0: treino {len(s0['train'])} "
          f"({100 * len(s0['train']) / n:.0f}%), val {len(s0['val'])} "
          f"({100 * len(s0['val']) / n:.0f}%), teste {len(s0['test'])} ({100 * len(s0['test']) / n:.0f}%)")
    print(f"\n-> {INDEX_CSV}\n-> {IMAGES_TXT}\n-> {FOLDS_CSV}\n-> {CACHE_DIR}/")


if __name__ == "__main__":
    main()
