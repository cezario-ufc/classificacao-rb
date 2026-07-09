"""Treino YOLO e predição (imagem cheia ou SAHI), com avaliação em mAP COCO.

- map_params: traduz nossa grade de hiperparâmetros para kwargs do Ultralytics.
- train_yolo: treina e devolve o caminho do best.pt.
- predict_coco: roda inferência (SAHI ou imagem cheia) e devolve predições em formato COCO.
- evaluate_config: predict + avaliação (mAP@0.1 e @0.5, global e por classe).
"""

from __future__ import annotations

from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from ultralytics import YOLO

from ddr_sahi.coco_eval import evaluate_per_image, evaluate_predictions


def map_params(params: dict) -> dict:
    """Hiperparâmetros de treino do Ultralytics (o resto é de inferência/SAHI).

    optimizer é fixado (default SGD) DE PROPÓSITO: com 'optimizer=auto' o Ultralytics
    ignora o lr0, o que anularia o GridSearch sobre lr0. Fixando, o lr0 passa a valer.
    """
    return {
        "imgsz": params["imgsz"],
        "lr0": params["lr0"],
        "epochs": params["epochs"],
        "batch": params["batch"],
        "optimizer": params.get("optimizer", "SGD"),
    }


def train_yolo(model_name, data_yaml, params, device, project, name):
    """Treina um YOLO e devolve o caminho do best.pt."""
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        device=device,
        seed=42,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
        # augmentação moderada p/ lesões pequenas (mosaic off)
        mosaic=0.0,
        **map_params(params),
    )
    return Path(project) / name / "weights" / "best.pt"


def _build_sahi_model(weights, conf, device):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
        device=device,
    )


def predict_coco(weights, image_paths, params, device, use_sahi):
    """Roda inferência e devolve lista de predições COCO. image_id = ordem em image_paths."""
    model = _build_sahi_model(weights, params.get("conf", 0.1), device)
    dt = []
    for img_id, p in enumerate(image_paths, start=1):
        if use_sahi:
            result = get_sliced_prediction(
                p, model,
                slice_height=params["slice"], slice_width=params["slice"],
                overlap_height_ratio=params["overlap"],
                overlap_width_ratio=params["overlap"],
                verbose=0,
            )
        else:
            result = get_prediction(p, model)
        for op in result.object_prediction_list:
            x, y, w, h = op.bbox.to_xywh()
            dt.append({
                "image_id": img_id,
                "category_id": op.category.id + 1,
                "bbox": [x, y, w, h],
                "score": op.score.value,
            })
    return dt


def evaluate_config(weights, image_paths, params, device, use_sahi, iou_thrs=(0.1, 0.5)):
    """Prediz nas imagens e devolve o dict de métricas (mAP + AP por classe)."""
    image_paths = list(image_paths)
    dt = predict_coco(weights, image_paths, params, device, use_sahi)
    return evaluate_predictions(image_paths, dt, iou_thrs=iou_thrs)


def evaluate_config_full(weights, image_paths, params, device, use_sahi,
                         iou_thrs=(0.1, 0.5), per_image_thr=0.1):
    """Como evaluate_config, mas prediz UMA vez e devolve também o AP por imagem.

    Retorna (metrics_agg, per_image) — per_image = {basename: AP@per_image_thr}, entrada
    da comparação pareada por imagem da Etapa 8.
    """
    image_paths = list(image_paths)
    dt = predict_coco(weights, image_paths, params, device, use_sahi)
    agg = evaluate_predictions(image_paths, dt, iou_thrs=iou_thrs)
    per_image = evaluate_per_image(image_paths, dt, iou_thr=per_image_thr)
    return agg, per_image