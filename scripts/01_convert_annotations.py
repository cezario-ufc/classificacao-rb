"""Etapa 2 — Conversão das anotações de detecção do DDR (Pascal VOC XML) para o formato YOLO.

O DDR já fornece as bounding boxes oficiais em `lesion_detection/{train,valid,test}/*.xml`
(Pascal VOC). Este script parseia esses XML e escreve:

  data/yolo/
  ├── images/<nome>.jpg    (hardlink p/ a imagem original; fallback = cópia)
  ├── labels/<nome>.txt    (YOLO: "classe cx cy w h", normalizados em [0,1])
  └── manifest.csv         (proveniência + vetor de presença de lesão por imagem)

Junta os 3 splits oficiais num único pool de 757 imagens — a divisão de treino/val/teste
é feita depois pela validação cruzada repetida (Etapa 3), não pelo split oficial do DDR.

Uso:
    python scripts/01_convert_annotations.py
    python scripts/01_convert_annotations.py --copy   # copia imagens em vez de hardlink

Não requer dependências externas (só a biblioteca padrão): o tamanho da imagem é lido do
próprio XML, então não precisa abrir os JPGs.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# --- Caminhos (relativos à raiz do repositório) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DDR_ROOT = REPO_ROOT / "data" / "raw" / "oia-ddr" / "DDR-dataset"
DET_DIR = DDR_ROOT / "lesion_detection"          # XMLs Pascal VOC, por split
IMG_DIR = DDR_ROOT / "lesion_segmentation"        # imagens em <split>/image/<nome>.jpg
OUT_DIR = REPO_ROOT / "data" / "yolo"
SPLITS = ["train", "valid", "test"]

# Ordem de classe conforme o spec (0 MA, 1 EX, 2 SE, 3 HE). Nomes do XML são minúsculos.
CLASS_MAP = {"ma": 0, "ex": 1, "se": 2, "he": 3}
CLASS_NAMES = {0: "MA", 1: "EX", 2: "SE", 3: "HE"}


def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    """Converte caixa VOC (pixels) para YOLO normalizado (cx, cy, bw, bh)."""
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return cx, cy, bw, bh


def parse_xml(xml_path: Path):
    """Retorna (image_filename, (W, H), lista de (class_id, cx, cy, w, h) deduplicada)."""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    W = int(size.findtext("width"))
    H = int(size.findtext("height"))

    seen = set()
    boxes = []
    skipped_cls = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_MAP:
            skipped_cls += 1
            continue
        cid = CLASS_MAP[name]
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:      # caixa degenerada
            continue
        # dedup: o DDR repete caixas idênticas dentro do mesmo XML
        key = (cid, xmin, ymin, xmax, ymax)
        if key in seen:
            continue
        seen.add(key)
        cx, cy, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, W, H)
        boxes.append((cid, cx, cy, bw, bh))
    return filename, (W, H), boxes, skipped_cls


def link_or_copy(src: Path, dst: Path, do_copy: bool):
    if dst.exists():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)          # hardlink (NTFS, mesmo volume) — sem duplicar disco
        return "link"
    except OSError:
        shutil.copy2(src, dst)     # fallback (volumes diferentes, sem suporte, etc.)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="copia as imagens em vez de criar hardlink")
    args = ap.parse_args()

    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    manifest = []
    per_class_boxes = {c: 0 for c in CLASS_NAMES}
    n_images = n_boxes = n_dupes_files = 0
    missing_images = []
    link_mode_count = {"link": 0, "copy": 0}

    for split in SPLITS:
        for xml_path in sorted((DET_DIR / split).glob("*.xml")):
            filename, (W, H), boxes, _ = parse_xml(xml_path)
            stem = xml_path.stem
            img_src = IMG_DIR / split / "image" / filename
            if not img_src.exists():
                missing_images.append(str(img_src))
                continue

            # escreve label YOLO
            label_path = OUT_DIR / "labels" / f"{stem}.txt"
            with open(label_path, "w") as f:
                for cid, cx, cy, bw, bh in boxes:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # linka/copia imagem
            img_dst = OUT_DIR / "images" / filename
            mode = link_or_copy(img_src, img_dst, args.copy)
            link_mode_count[mode] += 1

            # estatísticas + vetor de presença por classe (p/ estratificação, Etapa 3)
            present = {c: 0 for c in CLASS_NAMES}
            for cid, *_ in boxes:
                per_class_boxes[cid] += 1
                present[cid] = 1
            n_images += 1
            n_boxes += len(boxes)
            manifest.append({
                "image": filename,
                "split_orig": split,
                "n_boxes": len(boxes),
                "has_MA": present[0], "has_EX": present[1],
                "has_SE": present[2], "has_HE": present[3],
            })

    # manifest CSV
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    # --- Resumo ---
    print(f"Imagens convertidas : {n_images}")
    print(f"Caixas (total)      : {n_boxes}")
    for c, name in CLASS_NAMES.items():
        n_img_with = sum(m[f"has_{name}"] for m in manifest)
        print(f"  {name}: {per_class_boxes[c]:>6} caixas  |  em {n_img_with} imagens")
    print(f"Modo de imagem      : {link_mode_count['link']} hardlink, "
          f"{link_mode_count['copy']} cópia")
    if missing_images:
        print(f"AVISO: {len(missing_images)} imagens do XML não encontradas (ex.: "
              f"{missing_images[0]})")
    print(f"Saída em            : {OUT_DIR}")


if __name__ == "__main__":
    main()