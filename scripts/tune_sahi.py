"""Ajuste dos parâmetros de inferência do SAHI no VAL de cada fold (sem re-treino).

Duas etapas:

  --stage val   Para cada fold e cada conjunto de pesos (B = pesos de imagem cheia, C = pesos
                de tiles), roda a grade de parâmetros nas imagens de VAL do fold e grava
                results/sahi_tuning_val<sufixo>.csv (uma linha por fold × combinação).
  --stage test  Para cada fold, escolhe a combinação de MAIOR mAP@0.1 NO VAL DAQUELE FOLD e
                avalia uma única vez no TESTE do fold. Grava fold_maps/ap_per_image de A, B e C
                com sufixo '<sufixo>_tuned' (A = imagem cheia, recalculado no mesmo pipeline
                para o pareamento) e results/sahi_tuning_choices<sufixo>.csv.

Escolha POR FOLD (não global): o val do fold i contém imagens que são TESTE de outros folds
da mesma repetição; escolher uma combinação única olhando todos os vals usaria imagens de
teste na decisão. Aqui o val do fold i decide só o teste do fold i.

Grade (ver --scales/--pp/--routes):
  escala   'tile@imgsz': tamanho do tile e resolução em que a rede o vê. O fator de zoom é
           imgsz/tile (512@1024 = ×2, o B atual; 512@512 = ×1, o C atual). image_size é sempre
           explícito: sem ele o SAHI usa o imgsz do checkpoint e um tile maior seria reduzido.
  pp       junção das predições 'TIPO:MÉTRICA:LIMIAR' (GREEDYNMM:IOS:0.5 = default do SAHI).
  rota     de onde vem cada predição:
             sahi        tiles + imagem cheia do MESMO modelo, juntados (= get_sliced_prediction
                         padrão, perform_standard_pred=True; reproduz B/C atuais)
             sahi-nostd  só tiles
             cls:MA+EX   classes listadas vêm dos tiles; as demais da imagem cheia do modelo A
             size:64     caixas com lado < 64 px (na imagem original) vêm dos tiles; as maiores
                         da imagem cheia do modelo A
           Nas rotas híbridas a imagem cheia vem SEMPRE dos pesos de imagem cheia (config A, no
           imgsz do treino). Para C isso combina dois modelos (C nos tiles + A na imagem cheia):
           o modelo de tiles quase não detecta na imagem inteira.

Combinações que reproduzem os pipelines originais (sanidade): B = 512@1024 GREEDYNMM:IOS:0.5
sahi; C = 512@512 GREEDYNMM:IOS:0.5 sahi.

A inferência é feita UMA vez por imagem e escala; junção e rotas são aplicadas sobre as
predições em cache. Para isso o laço do get_sliced_prediction é reimplementado com as peças
do próprio SAHI (slice_image + get_prediction + postprocess). Rode --check antes do val
completo: compara com o get_sliced_prediction original.

Uso (na Balerion, imagem antiga com código montado; mesmo esquema do 03):
    python scripts/tune_sahi.py --check 5                                # equivalência
    python scripts/tune_sahi.py --stage val                              # yolov8m
    python scripts/tune_sahi.py --stage test
    python scripts/04_stats_and_plots.py --suffix _tuned
    python scripts/tune_sahi.py --stage val --model-suffix _yolo26n_musgd   # outro modelo

Ablação sem o modelo A (só rotas de um modelo; reaproveita o val, escolhe só entre elas):
    python scripts/tune_sahi.py --stage test --weights C --routes sahi,sahi-nostd --tag _tiles
    python scripts/04_stats_and_plots.py --suffix _tuned_tiles
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahi import AutoDetectionModel  # noqa: E402
from sahi.postprocess.combine import (  # noqa: E402
    GreedyNMMPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
from sahi.predict import get_prediction, get_sliced_prediction  # noqa: E402
from sahi.slicing import slice_image  # noqa: E402
from sahi.utils.cv import read_image_as_pil  # noqa: E402

from ddr_sahi.checkpoints import is_finalized  # noqa: E402
from ddr_sahi.coco_eval import (  # noqa: E402
    CLASS_NAMES,
    coco_map,
    evaluate_per_image,
    yolo_to_coco_gt,
)
from ddr_sahi.folds import YOLO_DIR, load_manifest, make_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "nested_cv"
RESULTS_DIR = REPO_ROOT / "results"

PP_CLASSES = {"GREEDYNMM": GreedyNMMPostprocess, "NMM": NMMPostprocess, "NMS": NMSPostprocess}
NAME_TO_ID = {n: i for i, n in CLASS_NAMES.items()}

DEFAULT_SCALES = "512@512,512@1024,768@1024,1024@1024"
DEFAULT_PP = ("GREEDYNMM:IOS:0.5,GREEDYNMM:IOU:0.5,NMM:IOS:0.5,NMS:IOU:0.5,"
              "GREEDYNMM:IOS:0.3,GREEDYNMM:IOS:0.7")
DEFAULT_ROUTES = "sahi,sahi-nostd,cls:MA,cls:MA+EX,size:32,size:64,size:128"


# --------------------------------------------------------------------------- pesos/modelos
def weights_map(model_suffix: str) -> dict[str, dict[str, Path]]:
    """{'full': {tag: best.pt}, 'C': {...}}. 'full' = pesos de imagem cheia (run AB, ou A/B)."""
    def first(*runs):
        for r in runs:
            d = RUNS_DIR / f"{r}{model_suffix}"
            m = {w.parents[2].name: w for w in d.glob("r*_f*/train/weights/best.pt")}
            if m:
                return m
        return {}
    return {"full": first("AB", "A", "B"), "C": first("C")}


class Models:
    """Cache de AutoDetectionModel por (pesos, image_size). image_size=None -> imgsz do ckpt."""

    def __init__(self, conf, device):
        self.conf, self.device, self.cache = conf, device, {}

    def get(self, weights: Path, image_size=None):
        key = (str(weights), image_size)
        if key not in self.cache:
            self.cache[key] = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(weights),
                confidence_threshold=self.conf, device=self.device, image_size=image_size)
        return self.cache[key]

    def clear(self):
        self.cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# --------------------------------------------------------------------------- inferência
def parse_scale(s: str):
    tile, isz = s.split("@")
    return int(tile), int(isz)


def tile_preds(model, pil, tile, overlap):
    """Predições de todos os tiles, já deslocadas p/ coordenadas da imagem (sem junção).
    Mesmo laço do get_sliced_prediction (slice_image + get_prediction + shift)."""
    sres = slice_image(image=pil, slice_height=tile, slice_width=tile,
                       overlap_height_ratio=overlap, overlap_width_ratio=overlap)
    full_shape = [sres.original_image_height, sres.original_image_width]
    out = []
    for img, start in zip(sres.images, sres.starting_pixels):
        pr = get_prediction(image=img, detection_model=model,
                            shift_amount=start, full_shape=full_shape)
        out.extend(op.get_shifted_object_prediction() for op in pr.object_prediction_list if op)
    return out, len(sres.images)


def full_preds(model, pil):
    """Predição na imagem inteira (= 'standard pred' do get_sliced_prediction)."""
    w, h = pil.size
    pr = get_prediction(image=pil, detection_model=model, shift_amount=[0, 0], full_shape=[h, w])
    return list(pr.object_prediction_list)


def load(path):
    """Lê a imagem sob demanda (guardar ~150 imagens de até 10 MP decodificadas estoura RAM)."""
    return read_image_as_pil(path)


def run_pp(pp_spec: str, preds):
    """Junção SAHI sobre uma CÓPIA (o postprocess reescreve a lista que recebe)."""
    if len(preds) <= 1:
        return list(preds)
    kind, metric, thr = pp_spec.split(":")
    pp = PP_CLASSES[kind](match_threshold=float(thr), match_metric=metric, class_agnostic=False)
    return pp(copy.deepcopy(preds))


def side(op):
    _, _, w, h = op.bbox.to_xywh()
    return math.sqrt(max(w, 0) * max(h, 0))


def route_preds(route, tiles_pp, sahi_pp, full_a):
    """Predições finais de uma imagem para uma rota (ver docstring do módulo)."""
    if route == "sahi":
        return sahi_pp
    if route == "sahi-nostd":
        return tiles_pp
    kind, arg = route.split(":")
    if kind == "cls":
        small = {NAME_TO_ID[n] for n in arg.split("+")}
        return ([p for p in tiles_pp if p.category.id in small]
                + [p for p in full_a if p.category.id not in small])
    if kind == "size":
        t = float(arg)
        return [p for p in tiles_pp if side(p) < t] + [p for p in full_a if side(p) >= t]
    raise ValueError(f"rota desconhecida: {route}")


def to_coco(preds_per_image):
    """Lista de listas de ObjectPrediction (alinhada às imagens) -> predições COCO."""
    dt = []
    for img_id, preds in enumerate(preds_per_image, start=1):
        for op in preds:
            x, y, w, h = op.bbox.to_xywh()
            dt.append({"image_id": img_id, "category_id": int(op.category.id) + 1,
                       "bbox": [float(x), float(y), float(w), float(h)],
                       "score": float(op.score.value)})
    return dt


def evaluate(gt, dt):
    """Mesmas chaves do coco_eval.evaluate_predictions, com o GT montado uma única vez."""
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):   # pycocotools é verboso
        for thr in (0.1, 0.5):
            m, per = coco_map(gt, dt, thr)
            out[f"mAP@{thr:g}"] = m
            for name, ap in per.items():
                out[f"AP@{thr:g}_{name}"] = ap
    return out


# --------------------------------------------------------------------------- utilidades
def img_paths(names):
    return [str((YOLO_DIR / "images" / n).resolve()) for n in names]


def read_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def usable(w: Path | None, what: str, tag: str) -> bool:
    if w is None:
        print(f"  [skip] {tag}: sem best.pt de {what}")
        return False
    if not is_finalized(w):
        print(f"  [skip] {tag}: best.pt de {what} NÃO finalizado (treino interrompido): {w}")
        return False
    return True


def tile_weights(kind, tag, wmaps):
    return wmaps["full"].get(tag) if kind == "B" else wmaps["C"].get(tag)


# --------------------------------------------------------------------------- etapa val
def stage_val(args, splits, wmaps):
    scales = [parse_scale(s) for s in args.scales.split(",")]
    pps, routes = args.pp.split(","), args.routes.split(",")
    need_std = "sahi" in routes
    out_csv = RESULTS_DIR / f"sahi_tuning_val{args.model_suffix}.csv"
    done = {(r["weights"], r["repeat"], r["fold"]) for r in read_rows(out_csv)}
    n_comb = len(scales) * len(pps) * len(routes)
    print(f"[val] {n_comb} combinações por fold × pesos {args.weights} -> {out_csv.name}")

    models = Models(args.conf, args.device)
    for split in splits:
        tag = f"r{split['repeat']}_f{split['fold']}"
        todo = [k for k in args.weights
                if (k, str(split["repeat"]), str(split["fold"])) not in done]
        if not todo:
            print(f"[{tag}] já concluído -> pula")
            continue
        # checa os pesos ANTES de qualquer inferência (não gasta GPU num fold que será pulado)
        if not usable(wmaps["full"].get(tag), "imagem cheia", tag):
            continue
        todo = [k for k in todo if usable(tile_weights(k, tag, wmaps), k, tag)]
        if not todo:
            continue
        t0 = time.time()
        paths = img_paths(split["val"])
        gt, _ = yolo_to_coco_gt(paths)
        full_a = [full_preds(models.get(wmaps["full"][tag]), load(p)) for p in paths]

        for kind in todo:
            tw = tile_weights(kind, tag, wmaps)
            rows = []
            for tile, isz in scales:
                m = models.get(tw, isz)
                tiles, std = [], [] if need_std else None
                for p in paths:
                    im = load(p)
                    tiles.append(tile_preds(m, im, tile, args.overlap)[0])
                    if need_std:
                        std.append(full_preds(m, im))
                for pp in pps:
                    tiles_pp = [run_pp(pp, t) for t in tiles]
                    sahi_pp = ([run_pp(pp, t + s) for t, s in zip(tiles, std)]
                               if need_std else None)
                    for route in routes:
                        preds = [route_preds(route, tiles_pp[i],
                                             sahi_pp[i] if need_std else None, full_a[i])
                                 for i in range(len(paths))]
                        rows.append({"weights": kind, "repeat": split["repeat"],
                                     "fold": split["fold"], "scale": f"{tile}@{isz}",
                                     "overlap": args.overlap, "pp": pp, "route": route,
                                     "conf": args.conf, "n_val": len(paths),
                                     **evaluate(gt, to_coco(preds))})
            append_rows(out_csv, rows)   # checkpoint por fold × pesos
            best = max(rows, key=lambda r: r["mAP@0.1"])
            base = next((r for r in rows if r["route"] == "sahi" and r["pp"] == "GREEDYNMM:IOS:0.5"
                         and r["scale"] == ("512@1024" if kind == "B" else "512@512")), None)
            print(f"[{tag}] {kind}: melhor val mAP@0.1={best['mAP@0.1']:.4f} "
                  f"({best['scale']} {best['pp']} {best['route']})"
                  + (f" | pipeline original={base['mAP@0.1']:.4f}" if base else ""))
        models.clear()
        print(f"[{tag}] {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- etapa test
def choose(val_rows):
    """{(weights, repeat, fold): linha de maior mAP@0.1 no val} (empate: 1ª na ordem da grade)."""
    best = {}
    for r in val_rows:
        k = (r["weights"], int(r["repeat"]), int(r["fold"]))
        if k not in best or float(r["mAP@0.1"]) > float(best[k]["mAP@0.1"]):
            best[k] = r
    return best


def in_grid(row, args):
    """Linha do val pertence à grade pedida (--scales/--pp/--routes)?"""
    return (row["scale"] in args.scales.split(",") and row["pp"] in args.pp.split(",")
            and row["route"] in args.routes.split(","))


def stage_test(args, splits, wmaps):
    val_csv = RESULTS_DIR / f"sahi_tuning_val{args.model_suffix}.csv"
    rows = read_rows(val_csv)
    if not rows:
        raise SystemExit(f"nada em {val_csv}: rode --stage val antes")
    kept = [r for r in rows if in_grid(r, args)]
    if not kept:
        raise SystemExit("nenhuma linha do val dentro da grade pedida (--scales/--pp/--routes)")
    if len(kept) < len(rows) and not args.tag:
        raise SystemExit("a grade pedida exclui combinações do val: passe --tag (ex.: _tiles) p/ "
                         "não misturar com as saídas *_tuned da grade completa")
    chosen = choose(kept)
    suffix = f"{args.model_suffix}_tuned{args.tag}"
    fold_csv = {c: RESULTS_DIR / f"fold_maps_{c}{suffix}.csv" for c in ["A", *args.weights]}
    img_csv = {c: RESULTS_DIR / f"ap_per_image_{c}{suffix}.csv" for c in fold_csv}
    choice_csv = RESULTS_DIR / f"sahi_tuning_choices{args.model_suffix}{args.tag}.csv"
    done = {c: {(int(r["repeat"]), int(r["fold"])) for r in read_rows(p)}
            for c, p in fold_csv.items()}
    print(f"[test] saídas: fold_maps_{{{','.join(fold_csv)}}}{suffix}.csv")

    models = Models(args.conf, args.device)
    for split in splits:
        key = (split["repeat"], split["fold"])
        tag = f"r{key[0]}_f{key[1]}"
        todo = [c for c in fold_csv if key not in done[c]]
        if not todo:
            print(f"[{tag}] já concluído -> pula")
            continue
        # checa escolha e pesos ANTES de qualquer inferência
        if not usable(wmaps["full"].get(tag), "imagem cheia", tag):
            continue
        ok = []
        for c in todo:
            if c == "A":
                ok.append(c)
            elif (c, *key) not in chosen:
                print(f"  [skip] {tag} {c}: fold sem resultado no val")
            elif usable(tile_weights(c, tag, wmaps), c, tag):
                ok.append(c)
        todo = ok
        if not todo:
            continue
        paths = img_paths(split["test"])
        gt, _ = yolo_to_coco_gt(paths)
        full_a = [full_preds(models.get(wmaps["full"][tag]), load(p)) for p in paths]

        for c in todo:
            if c == "A":
                preds, note = full_a, None
            else:
                ch = chosen[(c, *key)]
                tw = tile_weights(c, tag, wmaps)
                tile, isz = parse_scale(ch["scale"])
                m = models.get(tw, isz)
                tiles_pp, sahi_pp = [None] * len(paths), [None] * len(paths)
                for i, p in enumerate(paths):
                    im = load(p)
                    t = tile_preds(m, im, tile, float(ch["overlap"]))[0]
                    if ch["route"] == "sahi":
                        sahi_pp[i] = run_pp(ch["pp"], t + full_preds(m, im))
                    else:
                        tiles_pp[i] = run_pp(ch["pp"], t)
                preds = [route_preds(ch["route"], tiles_pp[i], sahi_pp[i], full_a[i])
                         for i in range(len(paths))]
                note = ch
            dt = to_coco(preds)
            res = evaluate(gt, dt)
            with contextlib.redirect_stdout(io.StringIO()):
                per_image = evaluate_per_image(paths, dt, iou_thr=0.1)
            append_rows(img_csv[c], [{"config": f"{c}_config", "repeat": key[0], "fold": key[1],
                                      "image": im, "AP": ap} for im, ap in per_image.items()])
            append_rows(fold_csv[c], [{"config": f"{c}_config", "repeat": key[0],
                                       "fold": key[1], **res}])
            if note is not None:
                append_rows(choice_csv, [{"weights": c, "repeat": key[0], "fold": key[1],
                                          "scale": note["scale"], "overlap": note["overlap"],
                                          "pp": note["pp"], "route": note["route"],
                                          "val_mAP@0.1": note["mAP@0.1"],
                                          "test_mAP@0.1": res["mAP@0.1"]}])
            print(f"[{tag}] {c}: teste mAP@0.1={res['mAP@0.1']:.4f}"
                  + (f"  ({note['scale']} {note['pp']} {note['route']})" if note else ""))
        models.clear()


# --------------------------------------------------------------------------- check
def stage_check(args, splits, wmaps):
    """Compara a reimplementação (rota 'sahi') com o get_sliced_prediction original."""
    split = next(s for s in splits if f"r{s['repeat']}_f{s['fold']}" in wmaps["full"])
    tag = f"r{split['repeat']}_f{split['fold']}"
    models = Models(args.conf, args.device)
    cases = [("B", wmaps["full"].get(tag), 512, 1024)]
    if tag in wmaps["C"]:
        cases.append(("C", wmaps["C"][tag], 512, 512))
    ok_all = True
    for kind, w, tile, isz in cases:
        m = models.get(w, isz)
        for p in img_paths(split["val"][:args.check]):
            pil = load(p)
            ref = get_sliced_prediction(
                pil, m, slice_height=tile, slice_width=tile,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5, perform_standard_pred=True, verbose=0,
            ).object_prediction_list
            mine = run_pp("GREEDYNMM:IOS:0.5",
                          tile_preds(m, pil, tile, args.overlap)[0] + full_preds(m, pil))

            def key(ops):
                return sorted((int(o.category.id), *(round(float(v), 1) for v in o.bbox.to_xywh()),
                               round(float(o.score.value), 4)) for o in ops)
            same = key(ref) == key(mine)
            ok_all &= same
            print(f"  [{kind} {tile}@{isz}] {Path(p).name}: SAHI={len(ref)} reimpl={len(mine)} "
                  f"-> {'IGUAL' if same else 'DIFERENTE'}")
    print("[check] OK: reimplementação equivale ao get_sliced_prediction" if ok_all else
          "[check] DIFERENÇAS: não rode o --stage val antes de entender (versão do SAHI?)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"])
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="compara com get_sliced_prediction em N imagens de val e sai")
    ap.add_argument("--model-suffix", default="",
                    help="sufixo dos runs do 03 (ex.: _yolo26n_musgd); vazio = yolov8m")
    ap.add_argument("--weights", nargs="+", default=["B", "C"], choices=["B", "C"],
                    help="B = pesos de imagem cheia nos tiles; C = pesos de tiles")
    ap.add_argument("--scales", default=DEFAULT_SCALES, help="'tile@imgsz,...'")
    ap.add_argument("--pp", default=DEFAULT_PP, help="'TIPO:MÉTRICA:LIMIAR,...'")
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.1, help="fixo p/ A, B e C (não é ajustado)")
    ap.add_argument("--folds", type=int, default=None, metavar="N", help="só os N primeiros folds")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="",
                    help="(test) sufixo extra das saídas, obrigatório se --scales/--pp/--routes "
                         "restringirem a grade (ex.: --routes sahi,sahi-nostd --tag _tiles)")
    args = ap.parse_args()
    if not args.stage and not args.check:
        ap.error("informe --stage val|test ou --check N")
    if args.tag and not args.tag.startswith("_"):
        args.tag = "_" + args.tag
    for pp in args.pp.split(","):
        if pp.split(":")[0] not in PP_CLASSES:
            ap.error(f"pp inválido: {pp}")

    wmaps = weights_map(args.model_suffix)
    if not wmaps["full"]:
        raise SystemExit(f"nenhum best.pt de imagem cheia em runs/nested_cv/AB{args.model_suffix}")
    images, Y = load_manifest()
    splits = make_folds(images, Y)
    if args.folds is not None:
        splits = splits[:args.folds]
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.check:
        stage_check(args, splits, wmaps)
    elif args.stage == "val":
        stage_val(args, splits, wmaps)
    else:
        stage_test(args, splits, wmaps)


if __name__ == "__main__":
    main()
