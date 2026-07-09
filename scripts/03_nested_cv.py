"""Etapa 3 (parte 2) — validação cruzada aninhada (10x3) com GridSearch + treino/avaliação.

Espelha o fluxo do tutorial (KNN) transposto para YOLO+SAHI:
  para cada fold externo (30 = 10x3):
    GridSearch: treina em train, valida em val (mAP@0.1) -> escolhe melhores hiperparâmetros
    re-treina no treino COMPLETO do fold e avalia no TESTE do fold (com/sem SAHI)
  desempenho final = média ± desvio dos 30 mAPs; guarda vetor por config p/ Wilcoxon (Etapa 8)

Configs (Seção 3 do spec):
  A: inferência em imagem cheia   (use_sahi=False)
  B: inferência com SAHI          (use_sahi=True)

Uso:
    python scripts/03_nested_cv.py --config B --smoke      # smoke-test (1 fold, subset, 1 época)
    python scripts/03_nested_cv.py --config B              # execução real (na máquina de treino)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddr_sahi.folds import (  # noqa: E402
    CLASS_NAMES,
    YOLO_DIR,
    build_fold_dirs,
    load_manifest,
    make_folds,
)
from ddr_sahi.slicing import slice_train_set  # noqa: E402
from ddr_sahi.train_eval import evaluate_config_full, train_yolo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs" / "nested_cv"
SLICED_DIR = REPO_ROOT / "data" / "sliced"

# Hiperparâmetros FIXOS (sem GridSearch): a busca aninhada era inviável no tempo real
# (multiplicava os treinos por fold). slice/overlap só afetam B/C. patience = early-stopping
# (para quando o val do Ultralytics estabiliza), então 'epochs' é só o teto de segurança.
PARAMS = {
    "model": "yolov8m.pt",
    "lr0": 0.01,
    "imgsz": 1024,
    "epochs": 100,
    "patience": 20,
    "batch": 8,
    "conf": 0.1,
    "slice": 512,
    "overlap": 0.2,
}

# Overrides do smoke-test (roda em segundos/minutos).
SMOKE_PARAMS = {
    "model": "yolov8n.pt",
    "lr0": 0.01,
    "imgsz": 640,
    "epochs": 1,
    "patience": 0,
    "batch": 4,
    "conf": 0.1,
    "slice": 320,
    "overlap": 0.2,
}
SMOKE_SUBSET = {"train": 8, "val": 4, "test": 4}


def img_paths(names):
    return [str((YOLO_DIR / "images" / n).resolve()) for n in names]


def ul_device(device: str):
    """Ultralytics prefere 0/'cpu'; SAHI usa 'cuda:0'/'cpu'."""
    if device.startswith("cuda"):
        return int(device.split(":")[1]) if ":" in device else 0
    return device


def subset_split(split, subset):
    s = dict(split)
    for key, n in subset.items():
        s[key] = s[key][:n]
    s["train_full"] = s["train"] + s["val"]
    return s


def sliced_data_yaml(image_names, val_txt, out_dir, params):
    """Config C: fatia as imagens de treino em tiles e monta um data.yaml (train=tiles,
    val=imagens cheias). slice/overlap seguem os hiperparâmetros do fold (mesma escala da
    inferência SAHI)."""
    img_dir, n_tiles, n_ann = slice_train_set(
        img_paths(image_names), out_dir,
        slice_size=params["slice"], overlap=params["overlap"],
    )
    yaml_path = Path(out_dir) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({
            "train": str(img_dir.resolve()),
            "val": str(val_txt), "test": str(val_txt),
            "names": CLASS_NAMES,
        }, f, sort_keys=False, allow_unicode=True)
    print(f"    [C] fatiado: {n_tiles} tiles, {n_ann} caixas")
    return yaml_path


def train_data_yaml(config, split, info, params):
    """data.yaml de treino. A/B treinam em imagens cheias (train), com val p/ early-stopping.
    C treina nos tiles do train do fold; val continua sendo imagens cheias."""
    if config in ("A", "B"):
        return info["grid_yaml"]        # train=train.txt, val=val.txt
    out = SLICED_DIR / info["dir"].name
    return sliced_data_yaml(split["train"], info["dir"] / "val.txt", out, params)


def run_fold(split, params, config, device):
    """Um treino por fold (sem GridSearch): treina no train do fold (val p/ early-stopping)
    e avalia no test do fold com a inferência da config (imagem cheia ou SAHI)."""
    use_sahi = config in ("B", "C")
    info = build_fold_dirs(split)
    proj = RUNS_DIR / info["dir"].name

    data_yaml = train_data_yaml(config, split, info, params)
    weights = train_yolo(params["model"], data_yaml, params,
                         ul_device(device), proj, "train")
    res, per_image = evaluate_config_full(weights, img_paths(split["test"]), params,
                                          device, use_sahi)

    # Config C gera ~15-18k tiles por fatiamento; apaga os tiles do fold após o uso
    # para não acumular disco ao longo dos folds.
    if config == "C":
        fold_tiles = SLICED_DIR / info["dir"].name
        shutil.rmtree(fold_tiles, ignore_errors=True)
        print(f"  [C] tiles do fold removidos ({fold_tiles.name})")

    return res, per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B", "C"], required=True,
                    help="A=imagem cheia, B=SAHI, C=SAHI + fine-tuning em tiles")
    ap.add_argument("--smoke", action="store_true", help="1 fold, subset, 1 época")
    ap.add_argument("--folds", type=int, default=None, metavar="N",
                    help="roda só os N primeiros folds em ESCALA REAL (sanity de VRAM/tempo/disco)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    params = SMOKE_PARAMS if args.smoke else PARAMS

    images, Y = load_manifest()
    splits = make_folds(images, Y)
    if args.smoke:
        splits = [subset_split(splits[0], SMOKE_SUBSET)]
        print(f"[SMOKE] config {args.config} | 1 fold | subset {SMOKE_SUBSET} | device {args.device}")
    elif args.folds is not None:
        total = len(splits)
        splits = splits[:args.folds]
        print(f"[SANITY] config {args.config} | {len(splits)}/{total} folds em escala real "
              f"| device {args.device}")
    else:
        print(f"config {args.config} | {len(splits)} folds | device {args.device}")

    RESULTS_DIR.mkdir(exist_ok=True)
    cfg_name = f"{args.config}_config"
    # sufixo separa saidas parciais (smoke/sanity) das do resultado final
    if args.smoke:
        suffix = "_smoke"
    elif args.folds is not None:
        suffix = f"_folds{args.folds}"
    else:
        suffix = ""
    rows, per_image_rows, test_maps = [], [], []
    for split in splits:
        tag = f"r{split['repeat']}_f{split['fold']}"
        print(f"\n[{tag}]")
        res, per_image = run_fold(split, params, args.config, args.device)
        print(f"  TESTE: mAP@0.1={res['mAP@0.1']:.4f}  mAP@0.5={res['mAP@0.5']:.4f}")
        test_maps.append(res["mAP@0.1"])
        rows.append({"config": cfg_name, "repeat": split["repeat"],
                     "fold": split["fold"], **res})
        for img, ap in per_image.items():
            per_image_rows.append({"config": cfg_name, "repeat": split["repeat"],
                                   "fold": split["fold"], "image": img, "AP": ap})

    # (1) vetor de mAPs por fold — comparação pareada por fold (Etapa 8)
    fold_csv = RESULTS_DIR / f"fold_maps_{args.config}{suffix}.csv"
    with open(fold_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # (2) AP por imagem — comparação pareada por imagem (Etapa 8, mais poder)
    img_csv = RESULTS_DIR / f"ap_per_image_{args.config}{suffix}.csv"
    with open(img_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    print(f"\nmAP@0.1: {np.mean(test_maps):.4f} ± {np.std(test_maps):.4f} "
          f"({len(test_maps)} folds)")
    print(f"Salvo: {fold_csv}\n       {img_csv}")


if __name__ == "__main__":
    main()
