"""Dump de predições A/B/C para a figura qualitativa (FP/FN) do artigo.

Roda na Balerion (pesos em runs/nested_cv/{AB|A|B} e runs/nested_cv/C). Para cada imagem
candidata, usa um fold em que ela está no TESTE (sem vazamento nem na ilustração) e que
tenha best.pt nas duas configs; roda A (imagem cheia), B (SAHI, pesos de A) e C (SAHI,
pesos do C) e grava tudo em results/qualitative_preds.json, com caixas absolutas em
pixels. A figura em si é montada localmente (sem GPU) a partir desse JSON.

Candidatas escolhidas pelos ap_per_image_*.csv: maior queda A->B em imagens com SE/HE
(fragmentação esperada) e maior ganho A->C em imagens ricas em MA.

Uso (na Balerion, mesmo esquema do bench_inference_time.py):
    sudo docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
      --volumes-from run_c \
      -v "$(pwd)/scripts/dump_qualitative_preds.py:/app/scripts/dump_qualitative_preds.py" \
      "$(sudo docker inspect run_c --format '{{.Config.Image}}')" \
      python scripts/dump_qualitative_preds.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from sahi import AutoDetectionModel  # noqa: E402
from sahi.predict import get_prediction, get_sliced_prediction  # noqa: E402

from ddr_sahi.folds import YOLO_DIR, load_manifest, make_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "nested_cv"

# SE/HE: maior queda A->B | MA: maior ganho A->C (ver _tmp_pick_candidates de 13/08/2026)
CANDIDATES = [
    "20170521093504075.jpg",
    "20170602084403248.jpg",
    "20170505094751770.jpg",
    "007-5505-300.jpg",
    "007-3372-200.jpg",
    "007-5296-300.jpg",
    "20170524181209695.jpg",
    "007-5286-300.jpg",
    "007-6220-300.jpg",
]


def weights_by_fold(runs_dir: Path) -> dict[str, Path]:
    out = {}
    for w in runs_dir.rglob("best.pt"):
        out[w.parents[2].name] = w
    return out


def find_fullimage_weights() -> dict[str, Path]:
    for cfg in ("AB", "A", "B"):
        wmap = weights_by_fold(RUNS_DIR / cfg)
        if wmap:
            print(f"[dump] pesos de imagem cheia: runs/nested_cv/{cfg} ({len(wmap)} folds)")
            return wmap
    raise SystemExit("nenhum best.pt de AB/A/B encontrado sob runs/nested_cv/")


def serialize(result):
    out = []
    for op in result.object_prediction_list:
        x, y, w, h = op.bbox.to_xywh()
        # np.float32/np.int64 nao sao serializaveis pelo json -> tipos nativos
        out.append([int(op.category.id), round(float(x), 1), round(float(y), 1),
                    round(float(w), 1), round(float(h), 1),
                    round(float(op.score.value), 4)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images", nargs="*", default=CANDIDATES)
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "qualitative_preds.json"))
    args = ap.parse_args()

    w_full = find_fullimage_weights()
    w_c = weights_by_fold(RUNS_DIR / "C")
    if not w_c:
        raise SystemExit("nenhum best.pt do C encontrado sob runs/nested_cv/C")

    images, Y = load_manifest()
    splits = make_folds(images, Y)

    models = {}  # cache: caminho do best.pt -> modelo carregado

    def get_model(weights):
        key = str(weights)
        if key not in models:
            models[key] = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=key,
                confidence_threshold=args.conf, device=args.device)
        return models[key]

    dump = {}
    for name in args.images:
        # fold em que a imagem está no TESTE e cujos pesos existem nas duas configs
        tag = next((f"r{s['repeat']}_f{s['fold']}" for s in splits
                    if name in s["test"]
                    and f"r{s['repeat']}_f{s['fold']}" in w_full
                    and f"r{s['repeat']}_f{s['fold']}" in w_c), None)
        if tag is None:
            print(f"  [skip] {name}: sem fold de teste com pesos nas duas configs")
            continue
        path = str((YOLO_DIR / "images" / name).resolve())
        wpx, hpx = Image.open(path).size
        kw = dict(slice_height=args.slice, slice_width=args.slice,
                  overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                  verbose=0)
        dump[name] = {
            "fold": tag, "width": wpx, "height": hpx,
            "preds": {
                "A": serialize(get_prediction(path, get_model(w_full[tag]))),
                "B": serialize(get_sliced_prediction(path, get_model(w_full[tag]), **kw)),
                "C": serialize(get_sliced_prediction(path, get_model(w_c[tag]), **kw)),
            },
        }
        n = {c: len(v) for c, v in dump[name]["preds"].items()}
        print(f"  [{tag}] {name}: A={n['A']} B={n['B']} C={n['C']} predições")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump))
    print(f"[dump] {len(dump)} imagens gravadas em {out_path}")


if __name__ == "__main__":
    main()
