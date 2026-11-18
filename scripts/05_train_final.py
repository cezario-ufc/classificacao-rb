"""Etapa final da detecção — treina o detector escolhido (Config C) com as 757 imagens anotadas
e o aplica, com o SAHI, a imagens novas (graduação do DDR e do IDRiD).

A validação cruzada (03 + tune_sahi) serviu para estimar o desempenho e escolher os
hiperparâmetros; aqui não se mede desempenho (não sobra imagem anotada fora do treino).

Hiperparâmetros (todos vêm dos resultados da validação cruzada, nada é escolhido aqui):
  treino    os do 03 (PARAMS + CONFIG_TRAIN_OVERRIDES['C']): modelo, otimizador, lr0, imgsz 512,
            batch 16, tiles de 512 px com sobreposição 0,2, mosaic off, seed 42.
  épocas    mediana, nos 15 folds, da época do best.pt do 03 (a que o early-stopping guardou
            no val). Sem val livre não há early-stopping: treina exatamente essas épocas, com o
            agendamento de LR do teto de 100 épocas (a mesma curva dos folds, parada no ponto
            típico). Pesos = os da última época (best.pt = last.pt).
  SAHI      a combinação mais frequente entre as escolhidas no val de cada fold, restrita a um
            modelo (results/sahi_tuning_choices<sufixo>_tiles.csv, do tune_sahi --tag _tiles):
            escala tile@imgsz, junção e rota (sahi = tiles + imagem cheia; sahi-nostd = só tiles).
            Confiança 0,1, a mesma da avaliação.
Tudo fica registrado em runs/final/C<sufixo>/hparams.json, que a etapa predict lê.

Etapas:
  --stage train    fatia as 757 imagens, treina e grava hparams.json. Retoma do last.pt se o
                   treino cair (segfault); best.pt finalizado -> não treina de novo.
  --stage predict  roda o SAHI com os parâmetros do hparams.json sobre --images (pasta ou
                   arquivo .txt com caminhos) e grava um JSONL (uma linha por imagem, inclusive
                   sem detecção), retomável: imagens já gravadas são puladas.

Uso (na Balerion; sufixo como no 03, ex.: --model yolo26n.pt --optimizer MuSGD -> _yolo26n_musgd):
    python scripts/05_train_final.py --stage train --model yolo26n.pt --optimizer MuSGD
    python scripts/05_train_final.py --stage predict --model yolo26n.pt --optimizer MuSGD --images data/raw/oia-ddr/DR_grading --out results/final_dets_ddr.jsonl
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ddr_sahi.checkpoints import is_finalized  # noqa: E402
from ddr_sahi.folds import CLASS_NAMES, load_manifest  # noqa: E402
from ddr_sahi.slicing import slice_train_set  # noqa: E402
from ddr_sahi.train_eval import (  # noqa: E402
    best_epoch_from_results,
    resumable_epoch,
    resume_yolo,
    train_yolo,
)

# hiperparâmetros, tags de sufixo e utilitários vêm do próprio 03 (fonte única)
_spec = importlib.util.spec_from_file_location("nested_cv", REPO_ROOT / "scripts" / "03_nested_cv.py")
ncv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ncv)

CONFIG = "C"
FINAL_DIR = REPO_ROOT / "runs" / "final"
SLICED_DIR = REPO_ROOT / "data" / "sliced_final"
RESULTS_DIR = REPO_ROOT / "results"
N_FOLDS_EXPECTED = 15
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- hiperparâmetros
def fold_epochs(suffix):
    """{fold: época do best.pt} dos runs do 03 da Config C com best.pt finalizado."""
    out = {}
    for d in sorted((ncv.RUNS_DIR / f"{CONFIG}{suffix}").glob("r*_f*/train")):
        w = d / "weights" / "best.pt"
        if w.exists() and is_finalized(w):
            e = best_epoch_from_results(d / "results.csv")
            if e is not None:
                out[d.parent.name] = e
    return out


def sahi_choice(suffix):
    """Combinação mais frequente (escala, overlap, junção, rota) entre as escolhas por fold."""
    path = RESULTS_DIR / f"sahi_tuning_choices{suffix}_tiles.csv"
    if not path.exists():
        raise SystemExit(f"{path} não existe: rode tune_sahi --stage test --weights C "
                         f"--routes sahi,sahi-nostd --tag _tiles --model-suffix {suffix}")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["weights"] == CONFIG]
    counts = Counter((r["scale"], r["overlap"], r["pp"], r["route"]) for r in rows)
    (scale, overlap, pp, route), n = counts.most_common(1)[0]
    tile, isz = (int(x) for x in scale.split("@"))
    kind, metric, thr = pp.split(":")
    return {
        "slice": tile, "image_size": isz, "overlap": float(overlap),
        "postprocess_type": kind, "postprocess_match_metric": metric,
        "postprocess_match_threshold": float(thr),
        "perform_standard_pred": route == "sahi",
        "route": route, "chosen_in_folds": f"{n}/{len(rows)}", "source": path.name,
    }


def build_hparams(params, suffix, epochs_override):
    epochs = fold_epochs(suffix)
    if len(epochs) < N_FOLDS_EXPECTED:
        print(f"  [aviso] só {len(epochs)}/{N_FOLDS_EXPECTED} folds do 03 com best.pt finalizado")
    if epochs_override is not None:
        n_epochs = epochs_override
    elif epochs:
        n_epochs = int(round(statistics.median(epochs.values())))
    else:
        raise SystemExit(f"nenhum run finalizado em {ncv.RUNS_DIR / (CONFIG + suffix)}; "
                         "passe --epochs")
    train = {**params, **ncv.CONFIG_TRAIN_OVERRIDES.get(CONFIG, {})}
    return {
        "config": CONFIG, "suffix": suffix,
        "train": {k: train[k] for k in ("model", "optimizer", "lr0", "imgsz", "batch", "epochs",
                                         "slice", "overlap")},
        "n_epochs": n_epochs,
        "n_epochs_source": "--epochs" if epochs_override is not None else "mediana dos folds",
        "fold_best_epochs": epochs,
        "sahi": sahi_choice(suffix),
        "conf": params["conf"],
    }


# --------------------------------------------------------------------------- treino
def data_yaml(names, sliced_dir, params):
    """Tiles das 757 imagens. Não há val livre: o val aponta para o próprio treino e só serve
    à métrica informativa que o Ultralytics calcula na última época (não escolhe pesos)."""
    tiles, n_tiles, n_ann = slice_train_set(ncv.img_paths(names), sliced_dir / "train",
                                            slice_size=params["slice"], overlap=params["overlap"])
    print(f"  fatiado: {len(names)} imagens -> {n_tiles} tiles, {n_ann} anotações")
    out = sliced_dir / "data.yaml"
    with open(out, "w") as f:
        yaml.safe_dump({"train": str(tiles.resolve()), "val": str(tiles.resolve()),
                        "names": CLASS_NAMES}, f, sort_keys=False, allow_unicode=True)
    return out


def finalize(train_dir):
    """Queda entre o save da última época e o final_eval: pesos completos, falta o
    strip_optimizer que o Ultralytics faria (retomar treinaria uma época a mais)."""
    from ultralytics.utils.torch_utils import strip_optimizer

    w = train_dir / "weights"
    strip_optimizer(w / "last.pt")
    shutil.copyfile(w / "last.pt", w / "best.pt")   # best.pt = last.pt neste regime
    return w / "best.pt"


def stage_train(args, params, suffix):
    run_dir = FINAL_DIR / f"{CONFIG}{suffix}"
    train_dir = run_dir / "train"
    weights, last = train_dir / "weights" / "best.pt", train_dir / "weights" / "last.pt"
    sliced_dir = SLICED_DIR / f"{CONFIG}{suffix}"

    if weights.exists() and is_finalized(weights):
        print(f"[reuse] treino final já concluído: {weights}")
        return
    hp = build_hparams(params, suffix, args.epochs)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hparams.json").write_text(json.dumps(hp, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    n_epochs = hp["n_epochs"]
    print(f"épocas = {n_epochs} ({hp['n_epochs_source']}; folds: "
          f"{sorted(hp['fold_best_epochs'].values())})")
    print(f"SAHI p/ a predição: {hp['sahi']}")

    images, _ = load_manifest()
    names = images.tolist()
    done = resumable_epoch(last)
    if done is None and resumable_epoch(last, require_data=False):
        print("  [resume] tiles do checkpoint ausentes -> refatia no mesmo caminho")
        shutil.rmtree(sliced_dir, ignore_errors=True)
        data_yaml(names, sliced_dir, params)
        done = resumable_epoch(last)

    if done is not None and done >= n_epochs:
        print(f"  [resume] {done} épocas já feitas (alvo {n_epochs}) -> só finaliza o checkpoint")
        weights = finalize(train_dir)
    elif done is not None:
        print(f"  [resume] treino interrompido após {done} épocas -> continua de {last}")
        weights = resume_yolo(last, ncv.ul_device(args.device), stop_epoch=n_epochs)
    else:
        if train_dir.exists():
            print(f"  [retrain] treino anterior incompleto em {train_dir} -> apaga e treina do zero")
            shutil.rmtree(train_dir)
        shutil.rmtree(sliced_dir, ignore_errors=True)
        yml = data_yaml(names, sliced_dir, params)
        tp = {**params, **ncv.CONFIG_TRAIN_OVERRIDES.get(CONFIG, {})}
        weights = train_yolo(tp["model"], yml, tp, ncv.ul_device(args.device), run_dir, "train",
                             stop_epoch=n_epochs)

    if not (weights.exists() and is_finalized(weights)):
        raise RuntimeError(f"treino terminou sem best.pt finalizado: {weights}")
    shutil.rmtree(sliced_dir, ignore_errors=True)
    print(f"\n[ok] pesos finais: {weights}\n     hiperparâmetros: {run_dir / 'hparams.json'}")


# --------------------------------------------------------------------------- predição
def list_images(src: Path):
    if src.is_file():
        return [Path(ln.strip()) for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)


def stage_predict(args, suffix):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image_as_pil

    run_dir = FINAL_DIR / f"{CONFIG}{suffix}"
    weights = run_dir / "train" / "weights" / "best.pt"
    if not (weights.exists() and is_finalized(weights)):
        raise SystemExit(f"sem pesos finais em {weights}: rode --stage train antes")
    hp = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    s = hp["sahi"]
    if not args.images:
        raise SystemExit("--stage predict precisa de --images (pasta ou .txt com caminhos)")
    images = list_images(Path(args.images))
    if not images:
        raise SystemExit(f"nenhuma imagem em {args.images}")
    out = Path(args.out) if args.out else RESULTS_DIR / f"final_dets{suffix}_{Path(args.images).stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(ln)["image"] for ln in f if ln.strip()}
    todo = [p for p in images if str(p) not in done]
    print(f"[predict] {len(images)} imagens ({len(done)} já feitas) -> {out}\n"
          f"  SAHI {s['slice']}@{s['image_size']} overlap {s['overlap']} "
          f"{s['postprocess_type']}:{s['postprocess_match_metric']}:{s['postprocess_match_threshold']} "
          f"rota {s['route']} conf {hp['conf']}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights), confidence_threshold=hp["conf"],
        device=args.device, image_size=s["image_size"])
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            pil = read_image_as_pil(str(p))
            res = get_sliced_prediction(
                pil, model, slice_height=s["slice"], slice_width=s["slice"],
                overlap_height_ratio=s["overlap"], overlap_width_ratio=s["overlap"],
                perform_standard_pred=s["perform_standard_pred"],
                postprocess_type=s["postprocess_type"],
                postprocess_match_metric=s["postprocess_match_metric"],
                postprocess_match_threshold=s["postprocess_match_threshold"],
                verbose=0)
            dets = []
            for op in res.object_prediction_list:
                x, y, w, h = op.bbox.to_xywh()
                dets.append({"class": CLASS_NAMES[int(op.category.id)],
                             "bbox_xywh": [round(float(v), 1) for v in (x, y, w, h)],
                             "score": round(float(op.score.value), 4)})
            f.write(json.dumps({"image": str(p), "width": pil.width, "height": pil.height,
                                "detections": dets}) + "\n")
            f.flush()   # checkpoint por imagem
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "predict"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=None, help=f"pesos YOLO (default: {ncv.PARAMS['model']})")
    ap.add_argument("--optimizer", default=ncv.PARAMS["optimizer"],
                    type=lambda s: next((o for o in ncv.OPTIMIZERS if o.lower() == s.lower()), s),
                    choices=ncv.OPTIMIZERS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="(train) força o nº de épocas em vez da mediana dos folds")
    ap.add_argument("--images", default=None, help="(predict) pasta ou .txt com caminhos")
    ap.add_argument("--out", default=None, help="(predict) JSONL de saída")
    args = ap.parse_args()

    params = dict(ncv.PARAMS)
    if args.model:
        params["model"] = args.model
    params["optimizer"] = args.optimizer
    suffix = (ncv.model_tag(args.model) if args.model else "") + ncv.optimizer_tag(args.optimizer)
    print(f"treino final | config {CONFIG} | modelo {params['model']} | otimizador "
          f"{params['optimizer']} | sufixo '{suffix}' -> runs/final/{CONFIG}{suffix}/")

    if args.stage == "train":
        stage_train(args, params, suffix)
    else:
        stage_predict(args, suffix)


if __name__ == "__main__":
    main()
