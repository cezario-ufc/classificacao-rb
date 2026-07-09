"""Etapa 8 — Estatística e figuras: compara as configs (A/B/C) com Wilcoxon pareado.

Consome os CSVs gerados pela Etapa 3 (scripts/03_nested_cv.py):
  results/fold_maps_{A,B,C}.csv        -> comparação por fold (30 mAPs pareados)
  results/ap_per_image_{A,B,C}.csv     -> comparação por imagem (mais poder)

Faz, para cada par de configs disponível:
  (1) Wilcoxon pareado sobre os 30 mAPs por fold  (+ teste-t corrigido de Nadeau-Bengio,
      que ajusta a correlação do k-fold repetido);
  (2) Wilcoxon pareado sobre o AP por imagem (média das repetições -> 1 valor por imagem).
Aplica correção de múltiplas comparações (Holm-Bonferroni) e reporta o tamanho de efeito.

Figuras em outputs/stats/: barras de mAP por config (erro entre folds) e AP por classe.

Uso:
    python scripts/04_stats_and_plots.py                 # resultados reais
    python scripts/04_stats_and_plots.py --suffix _smoke # sobre CSVs de smoke/teste
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "outputs" / "stats"
CONFIGS = ["A", "B", "C"]
CONFIG_LABEL = {"A": "A (baseline)", "B": "B (SAHI)", "C": "C (SAHI+FT)"}
CLASS_NAMES = ["MA", "EX", "SE", "HE"]

# fração de teste/treino do k-fold externo (10 folds -> teste=1/10, treino=9/10)
NB_N_TEST, NB_N_TRAIN = 1, 9


def holm_bonferroni(pvals):
    """Correção de Holm-Bonferroni. Retorna p-valores ajustados na ordem original."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)      # garante monotonicidade
        adj[idx] = min(running, 1.0)
    return adj


def nadeau_bengio_ttest(diffs, n_test=NB_N_TEST, n_train=NB_N_TRAIN):
    """Teste-t pareado corrigido (Nadeau & Bengio, 2003) para CV repetido.

    Corrige a variância subestimada por reuso de dados entre folds. Retorna (t, p) bilateral.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    corrected_var = (1.0 / n + n_test / n_train) * var
    t = mean / np.sqrt(corrected_var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def wilcoxon_paired(a, b):
    """Wilcoxon pareado + tamanho de efeito r = Z / sqrt(N). Retorna dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return {"stat": float("nan"), "p": 1.0, "r": 0.0, "n": len(a)}
    res = stats.wilcoxon(a, b, method="approx")
    z = res.zstatistic
    r = abs(z) / np.sqrt(len(a))
    return {"stat": float(res.statistic), "p": float(res.pvalue), "r": float(r), "n": len(a)}


def load(kind, suffix):
    """kind = 'fold_maps' ou 'ap_per_image'. Retorna {config: DataFrame} dos que existem."""
    out = {}
    for c in CONFIGS:
        path = RESULTS_DIR / f"{kind}_{c}{suffix}.csv"
        if path.exists():
            out[c] = pd.read_csv(path)
    return out


def fold_vector(df):
    """Vetor de mAP@0.1 por fold, ordenado por (repeat, fold) para parear entre configs."""
    return df.sort_values(["repeat", "fold"])["mAP@0.1"].to_numpy()


def image_vector(df):
    """AP médio por imagem (média das repetições), como Series indexada por imagem."""
    return df.groupby("image")["AP"].mean()


def compare(fold_data, image_data):
    """Roda as comparações pareadas para todos os pares de configs disponíveis."""
    pairs = [p for p in itertools.combinations(CONFIGS, 2)
             if p[0] in fold_data and p[1] in fold_data]
    results = []
    raw_p_fold, raw_p_img = [], []
    for x, y in pairs:
        a, b = fold_vector(fold_data[x]), fold_vector(fold_data[y])
        wf = wilcoxon_paired(a, b)
        t_nb, p_nb = nadeau_bengio_ttest(a - b)
        row = {"par": f"{x} vs {y}",
               "mean_x": a.mean(), "mean_y": b.mean(),
               "fold_W_p": wf["p"], "fold_r": wf["r"], "nb_t": t_nb, "nb_p": p_nb}
        raw_p_fold.append(wf["p"])
        if x in image_data and y in image_data:
            sx, sy = image_vector(image_data[x]), image_vector(image_data[y])
            common = sx.index.intersection(sy.index)
            wi = wilcoxon_paired(sx.loc[common].to_numpy(), sy.loc[common].to_numpy())
            row.update({"img_W_p": wi["p"], "img_r": wi["r"], "img_n": wi["n"]})
            raw_p_img.append(wi["p"])
        else:
            row.update({"img_W_p": np.nan, "img_r": np.nan, "img_n": 0})
            raw_p_img.append(np.nan)
        results.append(row)

    # Holm-Bonferroni por nível (família = os pares)
    if raw_p_fold:
        for r, adj in zip(results, holm_bonferroni(raw_p_fold)):
            r["fold_W_p_holm"] = adj
    valid_img = [p for p in raw_p_img if not np.isnan(p)]
    if valid_img:
        adj_img = holm_bonferroni([p if not np.isnan(p) else 1.0 for p in raw_p_img])
        for r, adj in zip(results, adj_img):
            r["img_W_p_holm"] = adj
    return pd.DataFrame(results)


def plot_map_bars(fold_data, suffix):
    """Barras de mAP@0.1 por config, com erro = desvio entre folds."""
    configs = list(fold_data.keys())
    means = [fold_vector(fold_data[c]).mean() for c in configs]
    stds = [fold_vector(fold_data[c]).std() for c in configs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([CONFIG_LABEL[c] for c in configs], means, yerr=stds, capsize=6,
           color=["#888", "#3a7", "#37a"][:len(configs)])
    ax.set_ylabel("mAP@0.1 (média ± desvio entre folds)")
    ax.set_title("Desempenho por configuração")
    fig.tight_layout()
    path = OUT_DIR / f"map_by_config{suffix}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_per_class(fold_data, suffix):
    """Barras agrupadas: AP@0.1 por classe, por config (média entre folds)."""
    configs = list(fold_data.keys())
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(configs):
        cols = [f"AP@0.1_{n}" for n in CLASS_NAMES]
        vals = [fold_data[c][col].mean() if col in fold_data[c] else 0 for col in cols]
        ax.bar(x + i * w, vals, w, label=CONFIG_LABEL[c])
    ax.set_xticks(x + w * (len(configs) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("AP@0.1 (média entre folds)")
    ax.set_title("AP por classe de lesão")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / f"ap_per_class{suffix}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="sufixo dos CSVs (ex.: _smoke)")
    args = ap.parse_args()

    fold_data = load("fold_maps", args.suffix)
    image_data = load("ap_per_image", args.suffix)
    if not fold_data:
        raise SystemExit(f"Nenhum results/fold_maps_*{args.suffix}.csv encontrado.")
    print("Configs encontradas:", ", ".join(fold_data))

    table = compare(fold_data, image_data)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Comparações pareadas ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"stats_summary{args.suffix}.csv", index=False)
    p1 = plot_map_bars(fold_data, args.suffix)
    p2 = plot_per_class(fold_data, args.suffix)
    print(f"\nFiguras: {p1}\n         {p2}")
    print(f"Tabela : {RESULTS_DIR / f'stats_summary{args.suffix}.csv'}")


if __name__ == "__main__":
    main()
