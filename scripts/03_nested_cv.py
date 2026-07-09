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
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid

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
from ddr_sahi.train_eval import (  # noqa: E402
    evaluate_config,
    evaluate_config_full,
    train_yolo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs" / "nested_cv"
SLICED_DIR = REPO_ROOT / "data" / "sliced"

# Grade completa (execução real). Config B usa slice/overlap; A ignora.
PARAM_GRID = {
    "model": ["yolov8m.pt"],
    "lr0": [0.01, 0.001],
    "imgsz": [1024],
    "epochs": [150],
    "batch": [8],
    "conf": [0.1],
    "slice": [512, 640],
    "overlap": [0.2],
}

# Overrides do smoke-test (roda em segundos/minutos na GPU local).
SMOKE_GRID = {
    "model": ["yolov8n.pt"],
    "lr0": [0.01],
    "imgsz": [640],
    "epochs": [1],
    "batch": [4],
    "conf": [0.1],
    "slice": [320],
    "overlap": [0.2],
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


def train_data_yaml(config, split, info, params, phase):
    """data.yaml de treino conforme a config. phase: 'grid' (train/val) ou 'best' (train_full)."""
    if config in ("A", "B"):
        return info["grid_yaml"] if phase == "grid" else info["full_yaml"]
    # Config C: treina em tiles
    names = split["train"] if phase == "grid" else split["train_full"]
    out = SLICED_DIR / info["dir"].name / phase
    return sliced_data_yaml(names, info["dir"] / "val.txt", out, params)


def run_fold(split, grid, config, device):
    use_sahi = config in ("B", "C")
    info = build_fold_dirs(split)
    proj = RUNS_DIR / info["dir"].name

    # --- GridSearch interno: treina em train, valida em val ---
    val_maps, params_list = [], []
    for params in ParameterGrid(grid):
        data_yaml = train_data_yaml(config, split, info, params, "grid")
        weights = train_yolo(params["model"], data_yaml, params,
                             ul_device(device), proj, "grid")
        m = evaluate_config(weights, img_paths(split["val"]), params, device, use_sahi)
        val_maps.append(m["mAP@0.1"])
        params_list.append(params)
        print(f"    grid {params['model']} lr0={params['lr0']} "
              f"slice={params.get('slice')} -> val mAP@0.1={m['mAP@0.1']:.4f}")

    best = params_list[int(np.argmax(val_maps))]
    print(f"  melhores hiperparâmetros: lr0={best['lr0']} slice={best.get('slice')}")

    # --- re-treino no treino COMPLETO do fold e avaliação no TESTE ---
    data_yaml = train_data_yaml(config, split, info, best, "best")
    weights = train_yolo(best["model"], data_yaml, best,
                         ul_device(device), proj, "best")
    res, per_image = evaluate_config_full(weights, img_paths(split["test"]), best,
                                          device, use_sahi)
    return best, res, per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B", "C"], required=True,
                    help="A=imagem cheia, B=SAHI, C=SAHI + fine-tuning em tiles")
    ap.add_argument("--smoke", action="store_true", help="1 fold, subset, 1 época")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    grid = SMOKE_GRID if args.smoke else PARAM_GRID

    images, Y = load_manifest()
    splits = make_folds(images, Y)
    if args.smoke:
        splits = [subset_split(splits[0], SMOKE_SUBSET)]
        print(f"[SMOKE] config {args.config} | 1 fold | subset {SMOKE_SUBSET} | device {args.device}")
    else:
        print(f"config {args.config} | {len(splits)} folds | device {args.device}")

    RESULTS_DIR.mkdir(exist_ok=True)
    cfg_name = f"{args.config}_config"
    suffix = "_smoke" if args.smoke else ""
    rows, per_image_rows, test_maps = [], [], []
    for split in splits:
        tag = f"r{split['repeat']}_f{split['fold']}"
        print(f"\n[{tag}]")
        best, res, per_image = run_fold(split, grid, args.config, args.device)
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
