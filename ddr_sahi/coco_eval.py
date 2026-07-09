"""Avaliação de detecção em mAP (protocolo COCO) a IoU configurável (0,1 e 0,5).

Constrói o ground-truth COCO a partir das labels YOLO das imagens do split e avalia uma
lista de predições (formato COCO) com pycocotools. Devolve mAP global e AP por classe.

IoU 0,1 segue Li et al. (2019), o protocolo de detecção do DDR; IoU 0,5 é o padrão usual.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = {0: "MA", 1: "EX", 2: "SE", 3: "HE"}

# Lado minimo de caixa (px). O round-trip normalizado do YOLO gera coords sub-pixel
# (ex.: w=0.999) que fazem o shapely do SAHI calcular area 0 -> divisao por zero no
# slice_coco (Config C). Garantir >= 2px elimina isso sem afetar o mAP a IoU 0,1/0,5.
MIN_BOX_PX = 2.0


def img2label(img_path: str) -> str:
    """Caminho da label YOLO correspondente (mesma regra do Ultralytics)."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return os.path.splitext(sb.join(img_path.rsplit(sa, 1)))[0] + ".txt"


def yolo_to_coco_gt(image_paths):
    """Monta o dict COCO do ground-truth. category_id = class_id + 1 (COCO evita id 0)."""
    images, annotations, ann_id = [], [], 1
    name_to_id = {}
    for img_id, p in enumerate(image_paths, start=1):
        w, h = Image.open(p).size
        name_to_id[os.path.basename(p)] = img_id
        images.append({"id": img_id, "file_name": os.path.basename(p),
                       "width": w, "height": h})
        lbl = img2label(p)
        if not os.path.exists(lbl):
            continue
        for line in Path(lbl).read_text().splitlines():
            if not line.strip():
                continue
            cid, cx, cy, bw, bh = line.split()
            cid = int(cid)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            # garante lado minimo (evita area 0 no shapely do SAHI), expandindo do centro
            bw, bh = max(bw, MIN_BOX_PX), max(bh, MIN_BOX_PX)
            x, y = cx - bw / 2, cy - bh / 2
            # mantem a caixa dentro da imagem
            x, y = max(0.0, min(x, w - bw)), max(0.0, min(y, h - bh))
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": cid + 1, "name": n} for cid, n in CLASS_NAMES.items()]
    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, name_to_id


def _ap_from_eval(cocoeval: COCOeval, present_cats: set):
    """Extrai AP por classe e mAP do COCOeval já rodado (1 único IoU thr).

    Classes SEM ground-truth no split (não em `present_cats`) ficam como None e são
    excluídas da média — o COCO padrão só promedia sobre classes presentes, senão um
    fold que por acaso não tem uma classe teria o mAP puxado para baixo indevidamente.
    """
    # precision shape: [T(iou), R(recall), K(cat), A(area), M(maxDet)]
    prec = cocoeval.eval["precision"]
    per_class = {}
    for k, (cid, name) in enumerate(CLASS_NAMES.items()):
        if (cid + 1) not in present_cats:
            per_class[name] = None
            continue
        p = prec[0, :, k, 0, -1]        # iou0, todas as áreas, maior maxDet
        p = p[p > -1]
        per_class[name] = float(p.mean()) if p.size else 0.0
    vals = [v for v in per_class.values() if v is not None]
    mAP = float(np.mean(vals)) if vals else 0.0
    return mAP, per_class


def coco_map(gt_dict, dt_list, iou_thr):
    """Devolve (mAP, {classe: AP|None}) para um único limiar de IoU."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    present_cats = {a["category_id"] for a in gt_dict["annotations"]}

    if not dt_list:                      # sem predições -> mAP 0 (não quebra o pipeline)
        return 0.0, {n: (0.0 if (c + 1) in present_cats else None)
                     for c, n in CLASS_NAMES.items()}

    coco_dt = coco_gt.loadRes(dt_list)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.params.iouThrs = np.array([iou_thr])
    E.params.maxDets = [100, 300, 1000]  # muitas lesões por imagem
    E.evaluate()
    E.accumulate()
    return _ap_from_eval(E, present_cats)


def evaluate_predictions(image_paths, dt_list, iou_thrs=(0.1, 0.5)):
    """Avalia predições COCO nas imagens do split; retorna dict com mAP e AP por classe."""
    gt_dict, _ = yolo_to_coco_gt(image_paths)
    out = {}
    for thr in iou_thrs:
        mAP, per_class = coco_map(gt_dict, dt_list, thr)
        out[f"mAP@{thr:g}"] = mAP
        for name, ap in per_class.items():
            out[f"AP@{thr:g}_{name}"] = ap
    return out


def evaluate_per_image(image_paths, dt_list, iou_thr=0.1):
    """AP por imagem (média sobre as classes presentes NAQUELA imagem), a um único IoU.

    Devolve {basename: AP}. Imagens sem ground-truth ficam de fora (não têm AP definido).
    É a entrada da comparação pareada POR IMAGEM da Etapa 8 (mais poder que a por fold).
    """
    gt_dict, _ = yolo_to_coco_gt(image_paths)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    # categorias presentes em cada imagem (a partir do GT)
    cats_by_img: dict[int, set] = {}
    for a in gt_dict["annotations"]:
        cats_by_img.setdefault(a["image_id"], set()).add(a["category_id"])

    id_to_name = {im["id"]: im["file_name"] for im in gt_dict["images"]}
    coco_dt = coco_gt.loadRes(dt_list) if dt_list else None

    per_image = {}
    for img_id, present in cats_by_img.items():
        name = id_to_name[img_id]
        if coco_dt is None:
            per_image[name] = 0.0            # sem predições -> AP 0 (a imagem tem lesão)
            continue
        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.params.imgIds = [img_id]
        E.params.iouThrs = np.array([iou_thr])
        E.params.maxDets = [100, 300, 1000]
        E.evaluate()
        E.accumulate()
        prec = E.eval["precision"]           # [T, R, K, A, M]
        vals = []
        for k, cid in enumerate(CLASS_NAMES):
            if (cid + 1) not in present:
                continue
            p = prec[0, :, k, 0, -1]
            p = p[p > -1]
            vals.append(float(p.mean()) if p.size else 0.0)
        per_image[name] = float(np.mean(vals)) if vals else 0.0
    return per_image
