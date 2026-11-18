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

    optimizer é sempre EXPLÍCITO (default SGD; MuSGD p/ YOLO26 via --optimizer no 03):
    com 'optimizer=auto' o Ultralytics ignora o lr0 e escolhe o otimizador pelo nº de
    iterações (AdamW em A/B vs MuSGD em C aqui). Explícito, o lr0 vale e é igual nas configs.
    """
    return {
        "imgsz": params["imgsz"],
        "lr0": params["lr0"],
        "epochs": params["epochs"],
        "patience": params.get("patience", 100),   # early-stopping
        "batch": params["batch"],
        "optimizer": params.get("optimizer", "SGD"),
    }


def _stop_after(n_epochs):
    """Callback que encerra o treino ao fim da época n_epochs (1-based). Mantém o agendamento
    de LR do 'epochs' original: o treino segue a mesma curva do run com early-stopping e só
    para mais cedo. Com self.stop=True o Ultralytics valida, salva e faz o final_eval."""
    def cb(trainer):
        if trainer.epoch + 1 >= n_epochs:
            trainer.stop = True
    return cb


def fixed_epochs_kwargs():
    """Sem validação por época e sem early-stopping (patience=0 -> infinita): no treino final
    (todas as imagens anotadas) não há conjunto livre para validar. O Ultralytics ainda valida na última
    época (só informativo; não escolhe pesos — best.pt = last.pt = pesos da última época)."""
    return {"val": False, "patience": 0}


def train_yolo(model_name, data_yaml, params, device, project, name, stop_epoch=None):
    """Treina um YOLO e devolve o caminho do best.pt.

    stop_epoch: treina exatamente esse nº de épocas, sem validação nem early-stopping (treino
    final, scripts/05_train_final.py); o teto 'epochs' de params continua definindo o agendamento
    de LR.
    """
    model = YOLO(model_name)
    extra = {}
    if stop_epoch is not None:
        model.add_callback("on_train_epoch_end", _stop_after(stop_epoch))
        extra = fixed_epochs_kwargs()
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
        **{**map_params(params), **extra},
    )
    return Path(project) / name / "weights" / "best.pt"


def best_epoch_from_results(csv_path, max_epoch=None) -> int | None:
    """Época (1-based) que o EarlyStopping/best.pt escolheu: a 1ª com o maior mAP50-95 (a
    fitness do Ultralytics 8.4) no results.csv. max_epoch ignora linhas além dele. None se o
    CSV não existe ou não tem a coluna."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    cols = [c.strip() for c in lines[0].split(",")]
    key = "metrics/mAP50-95(B)"
    if key not in cols:
        return None
    i = cols.index(key)
    vals = [(int(float(r.split(",")[0])), float(r.split(",")[i])) for r in lines[1:]]
    if max_epoch is not None:
        vals = [(e, v) for e, v in vals if e <= max_epoch]
    if not vals:
        return None
    top = max(v for _, v in vals)
    return next(e for e, v in vals if v == top)


def resumable_epoch(last_pt, require_data: bool = True) -> int | None:
    """Épocas concluídas no last.pt de um treino interrompido, ou None se não dá para retomar
    (ausente/corrompido, já finalizado, já na última época ou — com require_data — o
    data.yaml do checkpoint não existe mais)."""
    from ddr_sahi.checkpoints import meta_is_finalized, read_ckpt_meta

    last_pt = Path(last_pt)
    if not last_pt.exists():
        return None
    try:
        ck = read_ckpt_meta(last_pt)
    except Exception:  # noqa: BLE001  last.pt truncado (queda durante a gravação)
        return None
    args = ck.get("train_args") or {}
    epoch = ck.get("epoch")
    if meta_is_finalized(ck) or not isinstance(epoch, int) or epoch < 0:
        return None
    if not isinstance(args.get("data"), str):
        return None
    if require_data and not Path(args["data"]).exists():
        return None   # sem o data.yaml o Ultralytics cairia no dataset default
    if epoch + 1 >= int(args.get("epochs", 0)):
        return None   # Ultralytics recusa retomar após a última época
    return epoch + 1


def resume_yolo(last_pt, device, stop_epoch=None):
    """Retoma o treino do last.pt (Ultralytics restaura pesos, EMA, otimizador, scaler, época e
    best_fitness) e devolve o best.pt.

    O EarlyStopping NÃO é salvo no checkpoint: sem correção, a paciência recomeçaria do zero
    na retomada. Restauramos best_epoch = 1ª época com o maior mAP50-95 (a fitness do
    Ultralytics 8.4) no results.csv, como o EarlyStopping teria registrado (melhora estrita).
    O results.csv é cortado nas épocas do checkpoint (uma queda entre save_metrics e
    save_model deixaria uma linha a mais, que seria regravada).

    stop_epoch: retomada de um treino de nº fixo de épocas (train_yolo com stop_epoch): não há
    EarlyStopping a restaurar (val=False, patience infinita); só reinstala a parada na época.
    """
    last_pt = Path(last_pt)
    train_dir = last_pt.parents[1]
    done = resumable_epoch(last_pt)
    if done is None:
        raise ValueError(f"{last_pt} não é retomável")

    csv_path = train_dir / "results.csv"
    if csv_path.exists():
        lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        header, rows = lines[0], lines[1:]
        rows = [r for r in rows if int(float(r.split(",")[0])) <= done]
        csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    model = YOLO(str(last_pt))
    if stop_epoch is not None:
        model.add_callback("on_train_epoch_end", _stop_after(stop_epoch))
        print(f"  [resume] treino de épocas fixas: {done} feitas, para ao fim da {stop_epoch}")
    else:
        best_epoch = best_epoch_from_results(csv_path, max_epoch=done) or done

        def restore_stopper(trainer):
            trainer.stopper.best_fitness = trainer.best_fitness
            trainer.stopper.best_epoch = best_epoch
            print(f"  [resume] EarlyStopping restaurado: best_epoch={best_epoch}, "
                  f"best_fitness={float(trainer.best_fitness):.5f}, "
                  f"paciência={trainer.stopper.patience}")

        model.add_callback("on_pretrain_routine_end", restore_stopper)
    model.train(resume=True, device=device)
    return train_dir / "weights" / "best.pt"


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