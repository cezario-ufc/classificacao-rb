"""Ancora gold — treina o baseline no split OFICIAL do DDR (383/149/225) e compara com a
literatura. Valida os numeros ABSOLUTOS do pipeline contra Santos 2022 / Pereira 2023.

Nao usa a CV: usa a divisao oficial do DDR, ja gravada no manifest (coluna split_orig por
01_convert). Treina o Config A (imagem cheia, yolov8m@1024) em train(383), early-stop em
valid(149) e avalia em test(225) com o MESMO coco_eval. Um treino so (~30-60 min na GPU).

Nao interfere no C: roda na GPU 0 (NVIDIA_VISIBLE_DEVICES=0). Escreve pesos em runs/gold_anchor
e o resultado em results/gold_anchor.csv.

Uso (Balerion, GPU 0 livre, em paralelo ao C na GPU 1):
    sudo docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
      --volumes-from run_c \
      -v "$(pwd)/scripts/gold_anchor_official_split.py:/app/scripts/gold_anchor_official_split.py" \
      "$(sudo docker inspect run_c --format '{{.Config.Image}}')" \
      python scripts/gold_anchor_official_split.py --device cuda:0

Interpretacao: teste @0.5 caindo em ~0.13-0.22 (yolov8m >= YOLOv5-s do Santos ~0.15),
com MA a menor classe -> numeros absolutos validados contra a literatura. E' 1 amostra
(sem CV), entao serve como faixa de sanidade, nao como resultado do paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from ddr_sahi.coco_eval import CLASS_NAMES  # {0:MA,1:EX,2:SE,3:HE}  # noqa: E402
from ddr_sahi.folds import YOLO_DIR  # noqa: E402
from ddr_sahi.train_eval import evaluate_config, train_yolo  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = REPO_ROOT / "runs" / "gold_anchor"

# Mesmos hiperparametros do Config A (scripts/03_nested_cv.py::PARAMS).
PARAMS = {"model": "yolov8m.pt", "lr0": 0.01, "imgsz": 1024, "epochs": 100,
          "patience": 20, "batch": 8, "conf": 0.1}

# Referencia da literatura no TESTE oficial do DDR @0.5 (AP por classe + mAP).
LIT = {
    "Santos2022 (YOLOv5-s+Tilling, Adam)": {"MA": 0.055, "EX": 0.221, "SE": 0.157, "HE": 0.184, "mAP": 0.154},
    "Pereira2023 (YOLOR-CSP+SAHI, SGD)":   {"MA": 0.140, "EX": 0.328, "SE": 0.203, "HE": 0.219, "mAP": 0.2225},
}


def load_official_split():
    """Le o manifest e agrupa as imagens por split_orig oficial do DDR."""
    manifest = YOLO_DIR / "manifest.csv"
    groups = {"train": [], "valid": [], "test": []}
    for r in csv.DictReader(open(manifest)):
        groups.setdefault(r["split_orig"], []).append(r["image"])
    return groups


def img_paths(names):
    return [str((YOLO_DIR / "images" / n).resolve()) for n in names]


def write_list(path: Path, names):
    path.write_text("\n".join(img_paths(names)) + "\n")


def ul_device(device: str):
    """Ultralytics quer 0/'cpu'; SAHI/eval quer 'cuda:0'/'cpu'."""
    if device.startswith("cuda"):
        return int(device.split(":")[1]) if ":" in device else 0
    return device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=PARAMS["model"])
    ap.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    ap.add_argument("--batch", type=int, default=PARAMS["batch"])
    args = ap.parse_args()
    params = {**PARAMS, "model": args.model, "epochs": args.epochs, "batch": args.batch}

    groups = load_official_split()
    n_tr, n_va, n_te = len(groups["train"]), len(groups["valid"]), len(groups["test"])
    print(f"[gold] split oficial DDR: train={n_tr} valid={n_va} test={n_te} (esperado 383/149/225)")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    write_list(OUT_BASE / "train.txt", groups["train"])
    write_list(OUT_BASE / "val.txt", groups["valid"])
    write_list(OUT_BASE / "test.txt", groups["test"])
    data_yaml = OUT_BASE / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.safe_dump({
            "train": str((OUT_BASE / "train.txt").resolve()),
            "val": str((OUT_BASE / "val.txt").resolve()),
            "test": str((OUT_BASE / "test.txt").resolve()),
            "names": CLASS_NAMES,
        }, f, sort_keys=False, allow_unicode=True)

    print(f"[gold] treinando {params['model']} (imgsz={params['imgsz']}, batch={params['batch']}, "
          f"epochs<= {params['epochs']}, patience={params['patience']}) ...")
    weights = train_yolo(params["model"], data_yaml, params, ul_device(args.device),
                         OUT_BASE, "train")

    print("[gold] avaliando no TESTE oficial (imagem cheia, use_sahi=False) ...")
    res = evaluate_config(weights, img_paths(groups["test"]), params, args.device,
                          use_sahi=False, iou_thrs=(0.1, 0.5))

    # --- Resultado + comparacao com a literatura (@0.5) ---
    print(f"\n[gold] mAP@0.1 = {res['mAP@0.1']:.4f}   |   mAP@0.5 = {res['mAP@0.5']:.4f}\n")
    print("Comparacao no TESTE oficial do DDR @0.5 (AP por classe):")
    hdr = f"  {'fonte':38} " + " ".join(f"{c:>7}" for c in ['MA', 'EX', 'SE', 'HE']) + f" {'mAP':>7}"
    print(hdr)
    my = {c: res[f"AP@0.5_{c}"] for c in ['MA', 'EX', 'SE', 'HE']}
    print(f"  {'ESTE (yolov8m@1024, split oficial)':38} " +
          " ".join(f"{(my[c] if my[c] is not None else 0):>7.4f}" for c in ['MA', 'EX', 'SE', 'HE']) +
          f" {res['mAP@0.5']:>7.4f}")
    for name, ref in LIT.items():
        print(f"  {name:38} " + " ".join(f"{ref[c]:>7.4f}" for c in ['MA', 'EX', 'SE', 'HE']) +
              f" {ref['mAP']:>7.4f}")

    # grava CSV (host, via bind de results/)
    out_csv = REPO_ROOT / "results" / "gold_anchor.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "n_train", "n_val", "n_test", *res.keys()])
        w.writeheader()
        w.writerow({"split": "ddr_official", "n_train": n_tr, "n_val": n_va, "n_test": n_te, **res})
    print(f"\n[gold] gravado em {out_csv}")
    print("Faixa de sanidade: teste @0.5 ~0.13-0.22 e MA a menor classe -> pipeline validado "
          "contra a literatura (1 amostra, sem CV).")


if __name__ == "__main__":
    main()
