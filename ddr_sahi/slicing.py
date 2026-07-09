"""Etapa 6 (Config C) — slicing-aided fine-tuning: fatia o dataset de treino em tiles.

Fluxo: labels YOLO das imagens de treino -> COCO -> sahi.slice_coco (tiles + anotações
recortadas) -> de volta para YOLO (labels ao lado dos tiles). O YOLO então treina sobre os
tiles; a inferência final continua sendo SAHI sobre as imagens cheias (Config B/Etapa 5).
"""

from __future__ import annotations

import json
from pathlib import Path

from sahi.slicing import slice_coco

from ddr_sahi.coco_eval import yolo_to_coco_gt


def _coco_to_yolo_labels(coco_dict, images_dir: Path):
    """Escreve labels YOLO (ao lado dos tiles, em ../labels) a partir do COCO fatiado."""
    labels_dir = images_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list] = {}
    for a in coco_dict["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco_dict["images"]:
        w, h = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        lines = []
        for a in by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{a['category_id'] - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        # tiles sem lesão viram arquivo vazio (negativos úteis para precisão)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
    return labels_dir


def slice_train_set(image_paths, out_dir, slice_size=512, overlap=0.2,
                    ignore_negative=False):
    """Fatia as imagens de treino em tiles YOLO. Retorna o diretório de imagens (tiles)."""
    image_paths = list(image_paths)
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # 1) COCO das imagens de treino (file_name = basename; todas no mesmo image_dir)
    gt, _ = yolo_to_coco_gt(image_paths)
    image_dir = str(Path(image_paths[0]).parent)
    coco_path = out_dir / "coco_src.json"
    with open(coco_path, "w") as f:
        json.dump(gt, f)

    # 2) fatia
    coco_dict, _ = slice_coco(
        coco_annotation_file_path=str(coco_path),
        image_dir=image_dir,
        output_coco_annotation_file_name="sliced",
        output_dir=str(img_out),
        ignore_negative_samples=ignore_negative,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        verbose=False,
    )

    # 3) COCO fatiado -> labels YOLO ao lado dos tiles
    _coco_to_yolo_labels(coco_dict, img_out)
    n_tiles = len(coco_dict["images"])
    n_ann = len(coco_dict["annotations"])
    return img_out, n_tiles, n_ann
