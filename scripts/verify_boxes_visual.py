"""Validação visual da conversão (Etapa 2): desenha as caixas YOLO sobre as imagens.

Lê as labels geradas em data/yolo/labels/*.txt (round-trip normalizado->pixel), desenha
color-coded por classe sobre a imagem e salva em outputs/verify_boxes/. Para cada imagem
gera também um RECORTE COM ZOOM em torno de uma caixa pequena (MA), já que microaneurismas
têm poucos pixels e somem na imagem inteira.

Escolhe imagens representativas a partir do manifest.csv: uma com SE (raro), a com mais
caixas, e algumas com MA.

Uso:
    python scripts/verify_boxes_visual.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = REPO_ROOT / "data" / "yolo"
OUT_DIR = REPO_ROOT / "outputs" / "verify_boxes"

CLASS_NAMES = {0: "MA", 1: "EX", 2: "SE", 3: "HE"}
# cores distinguíveis (RGB)
CLASS_COLORS = {0: (255, 60, 60), 1: (60, 200, 60), 2: (80, 140, 255), 3: (255, 190, 40)}


def load_boxes(stem: str, W: int, H: int):
    """Lê a label YOLO e devolve caixas em pixels: (cid, x0, y0, x1, y1)."""
    boxes = []
    label = YOLO_DIR / "labels" / f"{stem}.txt"
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        cid, cx, cy, bw, bh = line.split()
        cid = int(cid)
        cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
        boxes.append((cid, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def pick_images():
    """Seleciona imagens representativas a partir do manifest."""
    rows = list(csv.DictReader(open(YOLO_DIR / "manifest.csv")))
    for r in rows:
        r["n_boxes"] = int(r["n_boxes"])
    chosen = {}
    # 1) imagem com SE (rara) e mais caixas
    se = [r for r in rows if r["has_SE"] == "1"]
    if se:
        chosen["se"] = max(se, key=lambda r: r["n_boxes"])
    # 2) imagem com mais caixas no geral
    chosen["densa"] = max(rows, key=lambda r: r["n_boxes"])
    # 3) imagem com MA e contagem moderada (nem vazia, nem gigante)
    ma = sorted([r for r in rows if r["has_MA"] == "1"], key=lambda r: r["n_boxes"])
    if ma:
        chosen["ma"] = ma[len(ma) // 2]
    # dedup por nome de imagem
    seen, out = set(), []
    for tag, r in chosen.items():
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        out.append((tag, r))
    return out


def draw_boxes(img: Image.Image, boxes, width: int):
    draw = ImageDraw.Draw(img)
    for cid, x0, y0, x1, y1 in boxes:
        draw.rectangle([x0, y0, x1, y1], outline=CLASS_COLORS[cid], width=width)
    return img


def legend(img: Image.Image, counts: dict):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    y = 10
    for cid, name in CLASS_NAMES.items():
        txt = f"{name}: {counts.get(cid, 0)}"
        draw.rectangle([10, y, 40, y + 24], fill=CLASS_COLORS[cid])
        draw.text((48, y), txt, fill=(255, 255, 255), font=font)
        y += 34


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, row in pick_images():
        name = row["image"]
        stem = Path(name).stem
        img_path = YOLO_DIR / "images" / name
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = load_boxes(stem, W, H)
        counts = {}
        for cid, *_ in boxes:
            counts[cid] = counts.get(cid, 0) + 1

        # (a) imagem inteira com todas as caixas + legenda
        full = draw_boxes(img.copy(), boxes, width=max(2, W // 800))
        legend(full, counts)
        out_full = OUT_DIR / f"{tag}_{stem}_full.jpg"
        full.save(out_full, quality=90)

        # (b) zoom numa caixa PEQUENA (menor área) — tipicamente um MA
        if boxes:
            smallest = min(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
            cid, x0, y0, x1, y1 = smallest
            cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
            pad = 120
            crop_box = (max(0, cxm - pad), max(0, cym - pad),
                        min(W, cxm + pad), min(H, cym + pad))
            crop = img.crop(crop_box).resize((480, 480), Image.NEAREST)
            # reprojeta as caixas que caem no recorte
            sx = 480 / (crop_box[2] - crop_box[0])
            sy = 480 / (crop_box[3] - crop_box[1])
            cboxes = []
            for c, bx0, by0, bx1, by1 in boxes:
                if bx1 < crop_box[0] or bx0 > crop_box[2] or by1 < crop_box[1] or by0 > crop_box[3]:
                    continue
                cboxes.append((c, (bx0 - crop_box[0]) * sx, (by0 - crop_box[1]) * sy,
                               (bx1 - crop_box[0]) * sx, (by1 - crop_box[1]) * sy))
            crop = draw_boxes(crop, cboxes, width=2)
            out_crop = OUT_DIR / f"{tag}_{stem}_zoom_{CLASS_NAMES[cid]}.jpg"
            crop.save(out_crop, quality=95)

        print(f"[{tag}] {name}  ({W}x{H})  caixas={len(boxes)} {counts}")
    print(f"\nSalvo em: {OUT_DIR}")


if __name__ == "__main__":
    main()
