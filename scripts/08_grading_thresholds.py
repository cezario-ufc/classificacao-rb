"""Graduação (parte 3) — limiar de confiança das detecções no vetor de lesões.

O JSONL do detector final vem com conf 0.1 (ponto de operação do mAP@0.1). Na graduação, as
EX/SE/HE de baixa confiança detectadas em imagens do grau leve (que, por definição, só tem MA)
empurram o leve para moderada. Este script refaz SÓ a cabeça de fusão com o vetor de lesões
calculado em outros limiares — sem fine-tuning e sem ler imagens: usa os embeddings que o
07_grading_cv.py guardou em runs/grading/<backbone>/r*_f*/emb.npz.

Variantes (cada uma com a mesma grade da MLP do 07 e a mesma escolha pela AUC do val):
  fus_t<t>    vetor original de 16 atributos só com as detecções de score >= t
              (fus_t0.1 = mlp_fusion do 07: confere que nada mudou além do limiar);
  fus_multi   n e área em vários limiares + cmax/cmean + indicador "só MA" por limiar
              (ddr_grading/lesions.py: lesion_vector_multi); a MLP aprende o peso de cada um;
  fus_tval    em cada fold, o fus_t<t> de maior AUC no val — o número "honesto" do limiar
              escolhido, já que o teste não participa da escolha.

Saídas (retomada por fold):
  results/grading_thr_<backbone>.csv        uma linha por fold × variante (mesmas colunas do 07)
  results/grading_thr_grid_<backbone>.csv   todas as combinações avaliadas (val)
  results/grading_thr_preds_<backbone>.csv  probabilidades por imagem do teste

Uso (na Balerion, depois do 07 de cada backbone):
    python scripts/08_grading_thresholds.py --backbones resnet152,mobilenet_v3,swin_t
    python scripts/08_grading_thresholds.py --backbones vgg19 --thresholds 0.1,0.3,0.5
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ddr_grading.data import load_index, make_grading_folds  # noqa: E402
from ddr_grading.lesions import (  # noqa: E402
    featurize,
    lesion_vector,
    lesion_vector_multi,
    load_detections,
)
from ddr_grading.models import BACKBONES  # noqa: E402
from ddr_grading.train import fit_mlp, metrics  # noqa: E402

g07 = importlib.import_module("07_grading_cv")   # mesma grade, tags, CSV e caminhos do 07


def thr_name(t):
    return f"fus_t{t:g}"


def emb_for(bb, split, winners):
    """Embeddings do fold (cache do 07). Confere que são do modelo vencedor do fine-tuning."""
    key = (split["repeat"], split["fold"])
    path = g07.RUNS_DIR / bb / f"r{key[0]}_f{key[1]}" / "emb.npz"
    if not path.exists():
        raise SystemExit(f"{path} não existe: rode o 07_grading_cv.py --backbone {bb} antes")
    z = np.load(path)
    if key in winners and str(z["tag"]) != winners[key]:
        raise SystemExit(f"{path}: tag {z['tag']} != vencedor do fine-tuning {winners[key]} "
                         f"em results/grading_{bb}.csv")
    emb = {k: z[k].astype(np.float32) for k in ("train", "val", "test")}
    for k in emb:
        if len(emb[k]) != len(split[k]):
            raise SystemExit(f"{path}: {k} tem {len(emb[k])} linhas, o fold tem {len(split[k])}")
    return emb


def run_fold(bb, split, args, labels_of, feats, winners, num_classes, device):
    names = {k: split[k] for k in ("train", "val", "test")}
    y = {k: [labels_of[n] for n in v] for k, v in names.items()}
    emb = emb_for(bb, split, winners)
    hps = g07.grid(args.mlp_lr_grid, args.mlp_batch_grid, args.mlp_wd_grid)
    out, grid_rows = {}, []
    for variant, fmap in feats.items():
        les = {k: np.stack([fmap[n] for n in v]) for k, v in names.items()}
        res = []
        for hp in hps:
            prob, val = fit_mlp(emb, les, y, num_classes, hp, device)
            res.append((hp, val, prob))
            grid_rows.append({"stage": variant, "repeat": split["repeat"],
                              "fold": split["fold"], **hp, **val})
        out[variant] = max(res, key=lambda r: r[1]["val_auc_ovr_macro"])
        print(f"  {variant:<10} {g07.tag(out[variant][0])} | val AUC "
              f"{out[variant][1]['val_auc_ovr_macro']:.4f}")
    thr = [v for v in out if v.startswith("fus_t")]
    chosen = max(thr, key=lambda v: out[v][1]["val_auc_ovr_macro"])
    order = list(out) + ["fus_tval"]
    out["fus_tval"] = out[chosen]
    print(f"  fus_tval   -> {chosen}")

    rows, preds = [], []
    for variant in order:
        hp, val, prob = out[variant]
        m = metrics(y["test"], prob, num_classes)
        rows.append({"backbone": bb, "variant": variant, "repeat": split["repeat"],
                     "fold": split["fold"], **{f"hp_{k}": v for k, v in hp.items()},
                     "val_auc_ovr_macro": val["val_auc_ovr_macro"],
                     "val_f1_macro": val["val_f1_macro"], "best_epoch": val["best_epoch"],
                     "chosen": chosen if variant == "fus_tval" else "",
                     "n_test": len(y["test"]), **{f"test_{k}": v for k, v in m.items()}})
        preds += [{"variant": variant, "repeat": split["repeat"], "fold": split["fold"],
                   "image": n, "label": t, "pred": int(p.argmax()),
                   **{f"p{c}": round(float(p[c]), 5) for c in range(num_classes)}}
                  for n, t, p in zip(names["test"], y["test"], prob)]
        print(f"  [{variant:<10}] TESTE: AUC {m['auc_ovr_macro']:.4f} | F1 {m['f1_macro']:.4f}"
              f" | sens {m['sensitivity_macro']:.4f} | acc {m['accuracy']:.4f}")
    res_dir = g07.RESULTS_DIR
    g07.append_rows(res_dir / f"grading_thr_grid_{bb}.csv", grid_rows)
    g07.append_rows(res_dir / f"grading_thr_preds_{bb}.csv", preds)
    g07.append_rows(res_dir / f"grading_thr_{bb}.csv", rows)


def summary(bb):
    rows = g07.read_rows(g07.RESULTS_DIR / f"grading_thr_{bb}.csv")
    ref = [r for r in g07.read_rows(g07.RESULTS_DIR / f"grading_{bb}.csv")
           if r["variant"] in ("mlp_emb", "mlp_fusion")]
    print(f"\n== {bb} (média nos folds) ==")
    for v in dict.fromkeys([r["variant"] for r in ref] + [r["variant"] for r in rows]):
        sel = [r for r in ref + rows if r["variant"] == v]
        print(f"  {v:<10} ({len(sel):>2} folds) " + " | ".join(
            f"{lbl} {np.mean([float(r['test_' + k]) for r in sel]):.4f}"
            for lbl, k in [("AUC", "auc_ovr_macro"), ("F1", "f1_macro"),
                           ("sens", "sensitivity_macro"), ("acc", "accuracy")]))
    ch = [r["chosen"] for r in rows if r["variant"] == "fus_tval"]
    if ch:
        print("  limiar escolhido no val: " + ", ".join(
            f"{c} ×{ch.count(c)}" for c in sorted(set(ch))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(BACKBONES),
                    help="lista separada por vírgula (default: todos)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dets", default=str(g07.DEFAULT_DETS), help="JSONL do 05 --stage predict")
    ap.add_argument("--folds", type=int, default=None, metavar="N", help="só os N primeiros folds")
    ap.add_argument("--thresholds", type=g07.floats, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help="limiares das variantes fus_t<t>")
    ap.add_argument("--multi", type=g07.floats, default=[0.1, 0.3, 0.5],
                    help="limiares do fus_multi (vazio = sem fus_multi)")
    ap.add_argument("--mlp-lr-grid", type=g07.floats, default=[1e-3, 1e-4])
    ap.add_argument("--mlp-batch-grid", type=g07.ints, default=[64, 256])
    ap.add_argument("--mlp-wd-grid", type=g07.floats, default=[0.0, 1e-4])
    args = ap.parse_args()

    backbones = [b for b in args.backbones.split(",") if b]
    unknown = [b for b in backbones if b not in BACKBONES]
    if unknown:
        raise SystemExit(f"backbone desconhecido: {unknown} (opções: {list(BACKBONES)})")

    index = load_index()
    labels_of = {r["image"]: r["label"] for r in index}
    num_classes = max(labels_of.values()) + 1
    dets = load_detections([args.dets], index)
    feats = {thr_name(t): featurize(dets, lesion_vector, min_score=t) for t in args.thresholds}
    if args.multi:
        feats["fus_multi"] = featurize(dets, lesion_vector_multi, thresholds=tuple(args.multi))
    splits = make_grading_folds([r["image"] for r in index], [r["label"] for r in index])
    if args.folds is not None:
        splits = splits[:args.folds]
    device = torch.device(args.device)
    n_fit = len(feats) * len(g07.grid(args.mlp_lr_grid, args.mlp_batch_grid, args.mlp_wd_grid))
    print(f"limiares | {backbones} | variantes {list(feats)} + fus_tval | {len(splits)} folds | "
          f"{n_fit} MLPs por fold")

    for bb in backbones:
        winners = {(int(r["repeat"]), int(r["fold"])):
                   g07.tag({"lr": float(r["hp_lr"]), "batch": int(r["hp_batch"]),
                            "weight_decay": float(r["hp_weight_decay"])})
                   for r in g07.read_rows(g07.RESULTS_DIR / f"grading_{bb}.csv")
                   if r["variant"] == "base"}
        done = {(int(r["repeat"]), int(r["fold"]))
                for r in g07.read_rows(g07.RESULTS_DIR / f"grading_thr_{bb}.csv")}
        for split in splits:
            key = (split["repeat"], split["fold"])
            print(f"\n[{bb} r{key[0]}_f{key[1]}]")
            if key in done:
                print("  já concluído -> pula")
                continue
            run_fold(bb, split, args, labels_of, feats, winners, num_classes, device)
        summary(bb)


if __name__ == "__main__":
    main()
