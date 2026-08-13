"""Benchmark de tempo de inferência por imagem: imagem cheia (A) vs. SAHI (B e C).

Responde ao parecer do CBEB (custo computacional/tempo de inferência) sem re-treino:
reusa o best.pt de UM fold já treinado, reconstrói (determinístico, seed=42) o teste
desse fold e cronometra a inferência imagem a imagem nos três regimes do artigo:

    A: imagem cheia (get_prediction), pesos treinados em imagem cheia
    B: SAHI 512/0.2 (get_sliced_prediction), MESMOS pesos de A
    C: SAHI 512/0.2, pesos treinados em tiles (runs do C)

Não treina e não altera nenhum resultado existente; grava results/inference_time.csv
(uma linha por imagem/config) e imprime o resumo para o artigo (média ± desvio, mediana,
imagens/s, tiles/imagem e razão de custo vs. A).

Uso (na Balerion, mesmo esquema do eval_C_iou.py):
    sudo docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
      --volumes-from run_c \
      -v "$(pwd)/scripts/bench_inference_time.py:/app/scripts/bench_inference_time.py" \
      "$(sudo docker inspect run_c --format '{{.Config.Image}}')" \
      python scripts/bench_inference_time.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sahi import AutoDetectionModel  # noqa: E402
from sahi.predict import get_prediction, get_sliced_prediction  # noqa: E402
from sahi.slicing import get_slice_bboxes  # noqa: E402

from ddr_sahi.folds import YOLO_DIR, load_manifest, make_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "nested_cv"


def weights_by_fold(runs_dir: Path) -> dict[str, Path]:
    """Mapeia 'rR_fF' -> best.pt sob runs_dir (mesma convenção do eval_C_iou.py)."""
    out = {}
    for w in runs_dir.rglob("best.pt"):
        out[w.parents[2].name] = w
    return out


def find_fullimage_weights() -> dict[str, Path]:
    """Pesos treinados em imagem cheia: o run 'AB' (ou 'A'/'B', se rodados separados)."""
    for cfg in ("AB", "A", "B"):
        wmap = weights_by_fold(RUNS_DIR / cfg)
        if wmap:
            print(f"[bench] pesos de imagem cheia: runs/nested_cv/{cfg} ({len(wmap)} folds)")
            return wmap
    raise SystemExit("nenhum best.pt de AB/A/B encontrado sob runs/nested_cv/")


def n_tiles(img_path: str, slice_px: int, overlap: float) -> int:
    w, h = Image.open(img_path).size  # lazy: lê só o cabeçalho da imagem
    return len(get_slice_bboxes(image_height=h, image_width=w,
                                slice_height=slice_px, slice_width=slice_px,
                                overlap_height_ratio=overlap,
                                overlap_width_ratio=overlap))


def bench(model, image_paths, sliced, slice_px, overlap, warmup):
    """Cronometra a inferência imagem a imagem (wall-clock, inclui pós-processamento)."""
    def infer(p):
        if sliced:
            get_sliced_prediction(p, model, slice_height=slice_px, slice_width=slice_px,
                                  overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                  verbose=0)
        else:
            get_prediction(p, model)

    for p in image_paths[:warmup]:  # aquece CUDA/cudnn fora do cronômetro
        infer(p)
    times = []
    for p in image_paths:
        t0 = time.perf_counter()
        infer(p)
        times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fold", default=None, metavar="rR_fF",
                    help="fold a usar (default: primeiro com pesos de imagem cheia E de C)")
    ap.add_argument("--limit", type=int, default=None, help="limita o nº de imagens do teste")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "inference_time.csv"))
    args = ap.parse_args()

    w_full = find_fullimage_weights()
    w_c = weights_by_fold(RUNS_DIR / "C")
    if not w_c:
        raise SystemExit("nenhum best.pt do C encontrado sob runs/nested_cv/C")

    common = sorted(set(w_full) & set(w_c))
    if not common:
        raise SystemExit("nenhum fold com pesos de imagem cheia E de C "
                         f"(imagem cheia: {sorted(w_full)}; C: {sorted(w_c)})")
    tag = args.fold or common[0]
    if tag not in common:
        raise SystemExit(f"fold {tag} sem pesos nas duas configs; disponíveis: {common}")

    images, Y = load_manifest()
    splits = make_folds(images, Y)
    split = next(s for s in splits if f"r{s['repeat']}_f{s['fold']}" == tag)
    test_imgs = [str((YOLO_DIR / "images" / n).resolve()) for n in split["test"]]
    if args.limit:
        test_imgs = test_imgs[:args.limit]

    gpu = (torch.cuda.get_device_name(0)
           if args.device.startswith("cuda") and torch.cuda.is_available() else "CPU")
    tiles = [n_tiles(p, args.slice, args.overlap) for p in test_imgs]
    print(f"[bench] fold {tag} | {len(test_imgs)} imagens de teste | {gpu} | "
          f"torch {torch.__version__}")
    print(f"[bench] tiles/imagem ({args.slice}/{args.overlap}): "
          f"média {statistics.mean(tiles):.1f} (min {min(tiles)}, max {max(tiles)})")

    jobs = [("A", w_full[tag], False), ("B", w_full[tag], True), ("C", w_c[tag], True)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["config", "fold", "image", "seconds", "n_tiles"])
        for cfg, weights, sliced in jobs:
            model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(weights),
                confidence_threshold=args.conf, device=args.device)
            times = bench(model, test_imgs, sliced, args.slice, args.overlap, args.warmup)
            for p, t, nt in zip(test_imgs, times, tiles):
                wcsv.writerow([cfg, tag, Path(p).name, f"{t:.4f}", nt if sliced else 1])
            summary[cfg] = times
            print(f"  [{cfg}] {statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} s/img "
                  f"(mediana {statistics.median(times):.3f}) -> "
                  f"{1 / statistics.mean(times):.2f} img/s")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    base = statistics.mean(summary["A"])
    print("\n[bench] resumo p/ o artigo:")
    for cfg, times in summary.items():
        m = statistics.mean(times)
        print(f"  {cfg}: {m:.3f} ± {statistics.stdev(times):.3f} s/img "
              f"({m / base:.1f}x o baseline A)")
    print(f"gravado em {out_path}")


if __name__ == "__main__":
    main()
