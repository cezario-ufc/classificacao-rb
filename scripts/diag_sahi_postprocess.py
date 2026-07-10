"""Diagnóstico (standalone) — o mAP baixo do Config B é o postprocess do SAHI?

Reusa pesos JÁ treinados do AB (não treina, não fatia, não escreve em runs/results do
pipeline). Roda a inferência do teste de UM fold com o postprocess default do SAHI
(== Config B atual) e com variantes (IOU / NMS), e imprime mAP@0.1 e @0.5.

Seguro para rodar em paralelo com o Config C:
  - não importa/edita nada que o C escreve; só LÊ os pesos do AB e as imagens;
  - default --device cpu para NÃO abrir um 2o contexto CUDA (evita OOM e queda do C).

Uso (na Balerion, de dentro do repo):
    python scripts/diag_sahi_postprocess.py --fold r0_f0            # CPU, todo o teste
    python scripts/diag_sahi_postprocess.py --fold r0_f0 --limit 30 # mais rapido (subset)
    python scripts/diag_sahi_postprocess.py --fold r0_f0 --device cuda:0  # so se sobrar VRAM

Leitura do resultado:
  - se 'B_default' reproduz o mAP@0.5 que voce ja tem para o fold  -> harness confere;
  - se 'B_iou'/'B_nms_iou' SOBEM bem acima de 'B_default'          -> era o postprocess
    (confundidor); vale corrigir e o C provavelmente sobe tambem;
  - se todas as variantes B ficam ABAIXO de 'A_fullimg'           -> achado robusto:
    o SAHI nao ajuda mesmo, independente do merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahi import AutoDetectionModel  # noqa: E402
from sahi.predict import get_prediction, get_sliced_prediction  # noqa: E402

from ddr_sahi.coco_eval import evaluate_predictions  # noqa: E402
from ddr_sahi.folds import YOLO_DIR, load_manifest, make_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Variantes de postprocess a comparar. A 1a linha (IOS/GREEDYNMM/std=True) e' EXATAMENTE
# o default do get_sliced_prediction -> reproduz o Config B atual.
VARIANTS = [
    # nome,          type,         metric, thr, perform_standard_pred
    ("B_default",   "GREEDYNMM",  "IOS",  0.5, True),
    ("B_iou",       "GREEDYNMM",  "IOU",  0.5, True),
    ("B_nms_iou",   "NMS",        "IOU",  0.5, True),
    ("B_iou_lo",    "GREEDYNMM",  "IOU",  0.3, True),
    ("B_no_std",    "GREEDYNMM",  "IOS",  0.5, False),
]


def test_images_for_fold(fold_tag: str, limit: int):
    """Reconstroi (deterministico, seed=42) o split e devolve os caminhos de teste do fold."""
    images, Y = load_manifest()
    for split in make_folds(images, Y):
        if f"r{split['repeat']}_f{split['fold']}" == fold_tag:
            names = split["test"]
            if limit:
                names = names[:limit]
            return [str((YOLO_DIR / "images" / n).resolve()) for n in names]
    raise SystemExit(f"fold {fold_tag} nao encontrado (use rREPEAT_fFOLD, ex.: r0_f0)")


def predict_full(model, image_paths):
    dt = []
    for img_id, p in enumerate(image_paths, start=1):
        for op in get_prediction(p, model).object_prediction_list:
            x, y, w, h = op.bbox.to_xywh()
            dt.append({"image_id": img_id, "category_id": op.category.id + 1,
                       "bbox": [x, y, w, h], "score": op.score.value})
    return dt


def predict_sliced(model, image_paths, slice_px, overlap, pp_type, pp_metric, pp_thr, std):
    dt = []
    for img_id, p in enumerate(image_paths, start=1):
        result = get_sliced_prediction(
            p, model,
            slice_height=slice_px, slice_width=slice_px,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type=pp_type,
            postprocess_match_metric=pp_metric,
            postprocess_match_threshold=pp_thr,
            perform_standard_pred=std,
            verbose=0,
        )
        for op in result.object_prediction_list:
            x, y, w, h = op.bbox.to_xywh()
            dt.append({"image_id": img_id, "category_id": op.category.id + 1,
                       "bbox": [x, y, w, h], "score": op.score.value})
    return dt


def row(name, metrics, ndet):
    print(f"  {name:12s}  mAP@0.1={metrics['mAP@0.1']:.4f}  "
          f"mAP@0.5={metrics['mAP@0.5']:.4f}  (#dets={ndet})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="r0_f0", help="fold do AB (ex.: r0_f0)")
    ap.add_argument("--weights", default=None,
                    help="best.pt; default runs/nested_cv/AB/<fold>/train/weights/best.pt")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, nao mexe na GPU do C) ou cuda:0 se sobrar VRAM")
    ap.add_argument("--limit", type=int, default=0, help="usar so as N 1as imagens de teste (0=todas)")
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else (
        REPO_ROOT / "runs" / "nested_cv" / "AB" / args.fold / "train" / "weights" / "best.pt")
    if not weights.exists():
        raise SystemExit(f"pesos nao encontrados: {weights}\n"
                         f"passe --weights com o caminho do best.pt do AB desse fold.")

    imgs = test_images_for_fold(args.fold, args.limit)
    print(f"[diag] fold={args.fold} | {len(imgs)} imgs de teste | device={args.device} | "
          f"pesos={weights}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights),
        confidence_threshold=args.conf, device=args.device)

    # Referencia: imagem cheia (== Config A). Deve bater com o A que voce ja tem no fold.
    dt = predict_full(model, imgs)
    row("A_fullimg", evaluate_predictions(imgs, dt), len(dt))

    # Variantes de SAHI (mesmos pesos, mesmos slices; muda so o merge).
    for name, pp_type, pp_metric, pp_thr, std in VARIANTS:
        dt = predict_sliced(model, imgs, args.slice, args.overlap,
                            pp_type, pp_metric, pp_thr, std)
        row(name, evaluate_predictions(imgs, dt), len(dt))


if __name__ == "__main__":
    main()
