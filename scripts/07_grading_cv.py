"""Graduação (parte 2) — validação cruzada de um backbone: base vs. fusão com o vetor de lesões.

Para cada um dos 15 folds (5 × 3, mesmo protocolo da detecção; ver ddr_grading/data.py):

  1. Grid search do fine-tuning: cada combinação (lr × batch × weight decay) ajusta o backbone
     no TREINO interno, com early-stopping na AUC-ROC macro do VAL interno. Vence a combinação
     de maior AUC no val. Variante 'base' = esse modelo (a classificação da própria rede) no
     TESTE do fold, avaliado uma única vez.
  2. Embeddings (entrada da camada de classificação) do modelo vencedor, sem augmentação, para
     treino, val e teste.
  3. Cabeças MLP sobre os embeddings, cada uma com a sua grade (lr × batch × weight decay) e a
     mesma escolha pela AUC do val:
       'mlp_emb'     só o embedding — controle: mesma cabeça da fusão sem as lesões, isola o
                     efeito do vetor do efeito de trocar a camada linear por uma MLP;
       'mlp_fusion'  embedding + vetor de 16 atributos das lesões (ddr_grading/lesions.py).
O teste do fold não participa de nenhuma escolha (pesos, épocas, combinação da grade).

Saídas (retomada por fold; dentro do fold, por combinação, via done.json/last.pt):
  results/grading_<backbone>.csv        uma linha por fold × variante (hiperparâmetros
                                        escolhidos, AUC/F1 do val, métricas do teste)
  results/grading_grid_<backbone>.csv   todas as combinações avaliadas (val)
  results/grading_preds_<backbone>.csv  probabilidades por imagem do teste (pareamento)
  runs/grading/<backbone>/r*_f*/        pesos (só o vencedor é mantido) e embeddings

Uso (na Balerion, um backbone por GPU; depois de 06 e do --stage predict do 05):
    python scripts/07_grading_cv.py --backbone resnet152
    python scripts/07_grading_cv.py --backbone vit_b_16 --batch-grid 16,32 --wd-grid 0,1e-4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ddr_grading.data import CLASS_NAMES, load_index, make_grading_folds  # noqa: E402
from ddr_grading.lesions import load_lesion_features  # noqa: E402
from ddr_grading.models import BACKBONES, Net  # noqa: E402
from ddr_grading.train import (  # noqa: E402
    SELECT,
    embed,
    eval_tf,
    finetune,
    fit_mlp,
    loader,
    metrics,
    predict_probs,
)

RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs" / "grading"
DEFAULT_DETS = RESULTS_DIR / "final_dets_grading.jsonl"
VARIANTS = ("base", "mlp_emb", "mlp_fusion")


def floats(s):
    return [float(x) for x in s.split(",")]


def ints(s):
    return [int(x) for x in s.split(",")]


def grid(lrs, batches, wds):
    return [{"lr": lr, "batch": b, "weight_decay": wd}
            for lr, b, wd in itertools.product(lrs, batches, wds)]


def tag(hp):
    return f"lr{hp['lr']:g}_b{hp['batch']}_wd{hp['weight_decay']:g}"


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def best_of(results):
    """(hp, val) de maior AUC no val; empate -> 1ª na ordem da grade."""
    return max(results, key=lambda r: r[1]["val_auc_ovr_macro"])


def run_fold(split, args, labels_of, lesions, num_classes, device):
    bb, cfg = args.backbone, BACKBONES[args.backbone]
    fold = f"r{split['repeat']}_f{split['fold']}"
    run_dir = RUNS_DIR / bb / fold
    names = {k: split[k] for k in ("train", "val", "test")}
    y = {k: [labels_of[n] for n in v] for k, v in names.items()}
    dl = {k: loader(names[k], y[k], eval_tf(cfg["input"]), shuffle=False)
          for k in ("val", "test")}
    grid_rows = []

    # 1. grade do fine-tuning (treino interno -> val interno)
    ft = []
    for hp in grid(args.lr_grid or cfg["lr_grid"], args.batch_grid, args.wd_grid):
        print(f"  [ft {tag(hp)}]")
        torch.manual_seed(42)
        net = Net(bb, num_classes)
        val = finetune(net, names["train"], y["train"], dl["val"], y["val"], num_classes, hp,
                       cfg["input"], device, run_dir / "ft" / tag(hp))
        ft.append((hp, val))
        grid_rows.append({"stage": "finetune", "repeat": split["repeat"], "fold": split["fold"],
                          **hp, **val})
        del net
        torch.cuda.empty_cache()
    hp_ft, val_ft = best_of(ft)
    print(f"  escolhida (fine-tuning): {tag(hp_ft)} | val AUC {val_ft['val_auc_ovr_macro']:.4f}")
    best_dir = run_dir / "ft" / tag(hp_ft)
    net = Net(bb, num_classes)
    net.load_state_dict(torch.load(best_dir / "best.pt", map_location="cpu"))
    net.to(device)
    if not args.keep_all:   # só o vencedor fica (VGG19: ~550 MB por combinação × fold)
        for hp, _ in ft:
            if hp is not hp_ft:
                (run_dir / "ft" / tag(hp) / "best.pt").unlink(missing_ok=True)

    out = {"base": (hp_ft, val_ft, predict_probs(net, dl["test"], device))}

    # 2. embeddings do vencedor (cache por fold)
    emb_npz = run_dir / "emb.npz"
    cached = np.load(emb_npz) if emb_npz.exists() else None
    if cached is not None and str(cached["tag"]) == tag(hp_ft):
        emb = {k: cached[k].astype(np.float32) for k in ("train", "val", "test")}
    else:
        dl_tr = loader(names["train"], y["train"], eval_tf(cfg["input"]), shuffle=False)
        emb = {"train": embed(net, dl_tr, device), "val": embed(net, dl["val"], device),
               "test": embed(net, dl["test"], device)}
        np.savez(emb_npz, tag=tag(hp_ft), **{k: v.astype(np.float16) for k, v in emb.items()})
    del net
    torch.cuda.empty_cache()

    # 3. cabeças MLP (só embedding / embedding + lesões), cada uma com a sua grade
    les = {k: np.stack([lesions[n] for n in v]) for k, v in names.items()}
    for variant, les_in in (("mlp_emb", None), ("mlp_fusion", les)):
        res = []
        for hp in grid(args.mlp_lr_grid, args.mlp_batch_grid, args.mlp_wd_grid):
            prob, val = fit_mlp(emb, les_in, y, num_classes, hp, device)
            res.append((hp, val, prob))
            grid_rows.append({"stage": variant, "repeat": split["repeat"],
                              "fold": split["fold"], **hp, **val})
        hp_v, val_v, prob_v = max(res, key=lambda r: r[1]["val_auc_ovr_macro"])
        out[variant] = (hp_v, val_v, prob_v)
        print(f"  escolhida ({variant}): {tag(hp_v)} | val AUC {val_v['val_auc_ovr_macro']:.4f}")

    rows, preds = [], []
    for variant in VARIANTS:
        hp, val, prob = out[variant]
        m = metrics(y["test"], prob, num_classes)
        rows.append({"backbone": bb, "variant": variant, "repeat": split["repeat"],
                     "fold": split["fold"], **{f"hp_{k}": v for k, v in hp.items()},
                     "val_auc_ovr_macro": val["val_auc_ovr_macro"],
                     "val_f1_macro": val["val_f1_macro"], "best_epoch": val["best_epoch"],
                     "n_test": len(y["test"]), **{f"test_{k}": v for k, v in m.items()}})
        preds += [{"variant": variant, "repeat": split["repeat"], "fold": split["fold"],
                   "image": n, "label": t, "pred": int(p.argmax()),
                   **{f"p{c}": round(float(p[c]), 5) for c in range(num_classes)}}
                  for n, t, p in zip(names["test"], y["test"], prob)]
        print(f"  [{variant}] TESTE: AUC {m['auc_ovr_macro']:.4f} | sens "
              f"{m['sensitivity_macro']:.4f} | F1 {m['f1_macro']:.4f} | prec "
              f"{m['precision_macro']:.4f} | espec {m['specificity_macro']:.4f} | acc "
              f"{m['accuracy']:.4f}")
    # grava o fold inteiro de uma vez (a retomada pula folds presentes no CSV principal)
    append_rows(RESULTS_DIR / f"grading_grid_{bb}.csv", grid_rows)
    append_rows(RESULTS_DIR / f"grading_preds_{bb}.csv", preds)
    append_rows(RESULTS_DIR / f"grading_{bb}.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dets", default=str(DEFAULT_DETS), help="JSONL do 05 --stage predict")
    ap.add_argument("--folds", type=int, default=None, metavar="N", help="só os N primeiros folds")
    ap.add_argument("--lr-grid", type=floats, default=None,
                    help="default por família: CNNs 1e-4,3e-5; transformers 3e-5,1e-5")
    ap.add_argument("--batch-grid", type=ints, default=[16, 32])
    ap.add_argument("--wd-grid", type=floats, default=[0.0, 1e-4])
    ap.add_argument("--mlp-lr-grid", type=floats, default=[1e-3, 1e-4])
    ap.add_argument("--mlp-batch-grid", type=ints, default=[64, 256])
    ap.add_argument("--mlp-wd-grid", type=floats, default=[0.0, 1e-4])
    ap.add_argument("--keep-all", action="store_true",
                    help="mantém o best.pt de todas as combinações (default: só o vencedor)")
    args = ap.parse_args()

    index = load_index()
    labels_of = {r["image"]: r["label"] for r in index}
    num_classes = max(labels_of.values()) + 1
    lesions = load_lesion_features([args.dets], index)   # falha cedo se faltar detecção
    splits = make_grading_folds([r["image"] for r in index], [r["label"] for r in index])
    if args.folds is not None:
        splits = splits[:args.folds]
    device = torch.device(args.device)
    n_ft = len(grid(args.lr_grid or BACKBONES[args.backbone]["lr_grid"], args.batch_grid,
                    args.wd_grid))
    print(f"graduação | {args.backbone} | {len(index)} imagens, {num_classes} classes "
          f"({', '.join(CLASS_NAMES[c] for c in range(num_classes))}) | {len(splits)} folds | "
          f"{n_ft} combinações de fine-tuning por fold | seleção: {SELECT} no val")

    main_csv = RESULTS_DIR / f"grading_{args.backbone}.csv"
    done = {(int(r["repeat"]), int(r["fold"])) for r in read_rows(main_csv)}
    RESULTS_DIR.mkdir(exist_ok=True)
    for split in splits:
        key = (split["repeat"], split["fold"])
        print(f"\n[r{key[0]}_f{key[1]}]")
        if key in done:
            print("  já concluído -> pula")
            continue
        run_fold(split, args, labels_of, lesions, num_classes, device)

    rows = read_rows(main_csv)
    print()
    keys = [("AUC", "auc_ovr_macro"), ("sens", "sensitivity_macro"), ("F1", "f1_macro"),
            ("prec", "precision_macro"), ("espec", "specificity_macro"), ("acc", "accuracy")]
    for v in VARIANTS:
        sel = [r for r in rows if r["variant"] == v]
        if sel:
            print(f"[{v:<10}] ({len(sel)} folds) " + " | ".join(
                f"{lbl} {np.mean([float(r['test_' + k]) for r in sel]):.4f} ± "
                f"{np.std([float(r['test_' + k]) for r in sel]):.4f}" for lbl, k in keys))


if __name__ == "__main__":
    main()
