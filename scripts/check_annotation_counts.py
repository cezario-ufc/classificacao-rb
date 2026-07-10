"""Sanity-check do GT: conta caixas e imagens-com-lesao por classe e compara com o DDR.

Ancora EXTERNA barata (sem GPU, sem treino): se a conversao VOC->YOLO estiver correta, os
totais tem que bater com os numeros publicados do DDR (Santos et al., Sensors 2022, Tabela 2):

    # caixas por classe:            MA 10388 | HE 13093 | EX 23713 | SE 1558
    # imagens com a lesao anotada:  MA  570  | HE   601 | EX   486 | SE  239

Le as MESMAS labels que o avaliador usa (img2label) a partir do manifest das 757 imagens.

Uso (na Balerion, so precisa do data/ montado; sem GPU):
    sudo docker run --rm --entrypoint python -v "$PWD/data:/app/data" \
        retino-sahi scripts/check_annotation_counts.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddr_sahi.coco_eval import CLASS_NAMES, img2label  # {0:MA,1:EX,2:SE,3:HE}  # noqa: E402
from ddr_sahi.folds import YOLO_DIR, load_manifest  # noqa: E402

# Referencia publicada (DDR, subconjunto anotado de 757 imagens).
REF_BOXES = {"MA": 10388, "HE": 13093, "EX": 23713, "SE": 1558}
REF_IMAGES = {"MA": 570, "HE": 601, "EX": 486, "SE": 239}
ORDER = ["MA", "HE", "EX", "SE"]  # ordem da tabela do Santos


def main():
    images, _ = load_manifest()
    box_counts = Counter()          # caixas por classe
    img_counts = Counter()          # imagens que contem >=1 caixa da classe
    n_imgs, n_missing = 0, 0

    for name in images:
        img_path = str((YOLO_DIR / "images" / name).resolve())
        lbl = img2label(img_path)
        n_imgs += 1
        if not os.path.exists(lbl):
            n_missing += 1
            continue
        seen = set()
        for line in Path(lbl).read_text().splitlines():
            if not line.strip():
                continue
            cid = int(line.split()[0])
            cls = CLASS_NAMES[cid]
            box_counts[cls] += 1
            seen.add(cls)
        for cls in seen:
            img_counts[cls] += 1

    print(f"imagens no manifest: {n_imgs}  (sem arquivo de label: {n_missing})")
    print(f"total de caixas: {sum(box_counts.values())}\n")

    def table(title, found, ref):
        print(title)
        print(f"  {'classe':6} {'encontrado':>11} {'esperado':>9} {'delta':>7}  status")
        all_ok = True
        for cls in ORDER:
            f, r = found.get(cls, 0), ref[cls]
            d = f - r
            ok = (d == 0)
            all_ok &= ok
            print(f"  {cls:6} {f:>11} {r:>9} {d:>+7}  {'OK' if ok else 'DIVERGE'}")
        print(f"  => {'TODOS BATEM' if all_ok else 'HA DIVERGENCIA'}\n")
        return all_ok

    ok1 = table("Caixas por classe:", box_counts, REF_BOXES)
    ok2 = table("Imagens com a lesao anotada:", img_counts, REF_IMAGES)

    if ok1 and ok2:
        print("RESULTADO: GT bate 100% com o DDR publicado -> conversao VOC->YOLO validada.")
    else:
        print("RESULTADO: ha divergencia. Pequenos deltas podem vir de caixas degeneradas "
              "descartadas na conversao; deltas grandes indicam erro de mapeamento de classe "
              "ou subconjunto de imagens diferente do publicado.")


if __name__ == "__main__":
    main()