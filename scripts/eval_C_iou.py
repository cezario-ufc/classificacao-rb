"""Re-avalia os pesos JÁ treinados do Config C com merge IOU (sem re-treino).

O run do C avalia com o postprocess default do SAHI (GREEDYNMM + IOS). O diagnóstico
(scripts/diag_sahi_postprocess.py) mostrou que trocar IOS->IOU dá um ganho pequeno mas
real. Este script varre os best.pt que o C salvou em runs/nested_cv/C/<fold>/, reconstrói
(deterministico, seed=42) o teste de cada fold, roda a inferência SAHI com merge IOU e
grava results/fold_maps_C_iou.csv no MESMO formato do fold_maps_C.csv original.

Não treina, não fatia, não apaga nada; escreve só o CSV novo (nome diferente do run do C).
Roda depois que o C terminar (aí as duas GPUs estão livres).

Uso (na Balerion, via mesmo esquema do diagnóstico):
    sudo docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
      --volumes-from run_c \
      -v "$(pwd)/scripts/eval_C_iou.py:/app/scripts/eval_C_iou.py" \
      "$(sudo docker inspect run_c --format '{{.Config.Image}}')" \
      python scripts/eval_C_iou.py --device cuda:0

Compare depois:
    fold_maps_C.csv      (IOS, do run original)
    fold_maps_C_iou.csv  (IOU, deste script)
    fold_maps_A.csv      (baseline imagem cheia)  <- a régua que interessa
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sahi import AutoDetectionModel  # noqa: E402
from sahi.predict import get_sliced_prediction  # noqa: E402

from ddr_sahi.coco_eval import evaluate_predictions  # noqa: E402
from ddr_sahi.folds import YOLO_DIR, load_manifest, make_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def weights_by_fold(runs_dir: Path) -> dict[str, Path]:
    """Mapeia 'rR_fF' -> best.pt encontrado sob runs_dir (C salva 1 por fold)."""
    out = {}
    for w in runs_dir.rglob("best.pt"):
        # .../C/rR_fF/train/weights/best.pt  -> parents[2].name == 'rR_fF'
        out[w.parents[2].name] = w
    return out


def predict_sliced_iou(model, image_paths, slice_px, overlap, pp_type, pp_metric, pp_thr):
    dt = []
    for img_id, p in enumerate(image_paths, start=1):
        result = get_sliced_prediction(
            p, model,
            slice_height=slice_px, slice_width=slice_px,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type=pp_type,
            postprocess_match_metric=pp_metric,
            postprocess_match_threshold=pp_thr,
            verbose=0,
        )
        for op in result.object_prediction_list:
            x, y, w, h = op.bbox.to_xywh()
            dt.append({"image_id": img_id, "category_id": op.category.id + 1,
                       "bbox": [x, y, w, h], "score": op.score.value})
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs" / "nested_cv" / "C"),
                    help="onde estão os best.pt do C")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "fold_maps_C_iou.csv"))
    ap.add_argument("--device", default="cpu", help="cpu ou cuda:0 (use cuda:0 apos o C terminar)")
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pp-type", default="GREEDYNMM")
    ap.add_argument("--pp-metric", default="IOU", help="IOU (default aqui) ou IOS (default do SAHI)")
    ap.add_argument("--pp-thr", type=float, default=0.5)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    wmap = weights_by_fold(runs_dir)
    if not wmap:
        raise SystemExit(f"nenhum best.pt encontrado em {runs_dir} (o C ja salvou algum fold?)")

    images, Y = load_manifest()
    splits = make_folds(images, Y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[eval-C-IOU] {len(wmap)} folds com pesos | merge={args.pp_type}/{args.pp_metric}"
          f"@{args.pp_thr} | device={args.device}")

    writer = None
    f_out = open(out_path, "w", newline="")
    maps01, maps05 = [], []
    for split in splits:
        tag = f"r{split['repeat']}_f{split['fold']}"
        if tag not in wmap:
            continue  # fold ainda nao treinado pelo C
        test_imgs = [str((YOLO_DIR / "images" / n).resolve()) for n in split["test"]]
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(wmap[tag]),
            confidence_threshold=args.conf, device=args.device)
        dt = predict_sliced_iou(model, test_imgs, args.slice, args.overlap,
                                args.pp_type, args.pp_metric, args.pp_thr)
        res = evaluate_predictions(test_imgs, dt, iou_thrs=(0.1, 0.5))
        row = {"config": "C_config", "repeat": split["repeat"], "fold": split["fold"], **res}

        if writer is None:  # cabecalho no formato do fold_maps_C.csv
            writer = csv.DictWriter(f_out, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        f_out.flush()  # sobrevive a interrupcao no meio dos 15 folds

        maps01.append(res["mAP@0.1"])
        maps05.append(res["mAP@0.5"])
        print(f"  [{tag}]  mAP@0.1={res['mAP@0.1']:.4f}  mAP@0.5={res['mAP@0.5']:.4f}  "
              f"({len(dt)} dets)")
    f_out.close()

    print(f"\n[C+IOU] mAP@0.1 = {np.mean(maps01):.4f} ± {np.std(maps01):.4f} | "
          f"mAP@0.5 = {np.mean(maps05):.4f} ± {np.std(maps05):.4f}  ({len(maps05)} folds)")
    print(f"gravado em {out_path}")
    print("compare com results/fold_maps_C.csv (IOS) e results/fold_maps_A.csv (baseline).")


if __name__ == "__main__":
    main()
