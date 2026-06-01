import argparse
import json

import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid, train_test_split
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    EPOCHS_FINAL,
    EPOCHS_GRIDSEARCH,
    IMG_SIZE,
    NUM_WORKERS,
    OUTPUTS_DIR,
    PARAM_GRID,
    SEED,
    USE_AUGMENT,
    VAL_RATIO,
)
from src.data.build_dataset import (
    build_ddr_dataframe,
    build_idrid_dataframe,
    build_messidor_dataframe,
)
from src.data.dataset import RetinopathyDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.builders import MODEL_BUILDERS
from src.training.train import evaluate, train_one_epoch

# esquema unificado 0-3 apos remap de DDR/IDRID (4 -> 3)
NUM_CLASSES_EXTERNAL = 4

DATASET_BUILDERS = {
    "ddr": build_ddr_dataframe,
    "messidor": build_messidor_dataframe,
    "idrid": build_idrid_dataframe,
}


def make_loader(df, transform, shuffle):
    ds = RetinopathyDataset(df, transform=transform)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def run_pair(model_name, train_name, test_name, device):
    print(f"\n[{model_name}] === {train_name.upper()} -> {test_name.upper()} ===")

    train_full_df = DATASET_BUILDERS[train_name]()
    test_df = DATASET_BUILDERS[test_name]()

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=VAL_RATIO,
        stratify=train_full_df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(
        f"  n_train={len(train_df)}  n_val={len(val_df)}"
        f"  n_test_externo={len(test_df)}"
    )

    train_tf = get_train_transforms(IMG_SIZE, augment=USE_AUGMENT)
    eval_tf = get_eval_transforms(IMG_SIZE)

    train_loader = make_loader(train_df, train_tf, shuffle=True)
    val_loader = make_loader(val_df, eval_tf, shuffle=False)

    f1s_val = []
    val_metrics_all = []
    par = []

    for params in ParameterGrid(PARAM_GRID):
        print(f"  params={params}")
        model = MODEL_BUILDERS[model_name](num_classes=NUM_CLASSES_EXTERNAL).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        for _ in range(EPOCHS_GRIDSEARCH):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_metrics = evaluate(
            model, val_loader, device, num_classes=NUM_CLASSES_EXTERNAL
        )
        print(
            f"    val_acc={val_metrics['accuracy']:.4f}"
            f" f1={val_metrics['f1_macro']:.4f}"
            f" auc={val_metrics['auc_roc_macro_ovr']:.4f}"
        )
        f1s_val.append(val_metrics["f1_macro"])
        val_metrics_all.append(val_metrics)
        par.append(params)

        del model, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_idx = f1s_val.index(max(f1s_val))
    best_params = par[best_idx]
    best_val_metrics = val_metrics_all[best_idx]
    print(
        f"  Melhores hiperparametros: {best_params}"
        f"  val_f1={best_val_metrics['f1_macro']:.4f}"
    )

    full_train_loader = make_loader(train_full_df, train_tf, shuffle=True)
    test_loader = make_loader(test_df, eval_tf, shuffle=False)

    model_best = MODEL_BUILDERS[model_name](num_classes=NUM_CLASSES_EXTERNAL).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model_best.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    for _ in range(EPOCHS_FINAL):
        train_one_epoch(model_best, full_train_loader, criterion, optimizer, device)

    test_metrics = evaluate(
        model_best, test_loader, device, num_classes=NUM_CLASSES_EXTERNAL
    )
    print(
        f"  ext_test ({test_name}):"
        f" acc={test_metrics['accuracy']:.4f}"
        f" precision={test_metrics['precision_macro']:.4f}"
        f" recall={test_metrics['recall_macro']:.4f}"
        f" f1={test_metrics['f1_macro']:.4f}"
        f" auc={test_metrics['auc_roc_macro_ovr']:.4f}"
    )

    del model_best, optimizer, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_params": best_params,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }


def run_external_validation(model_name, datasets):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device={device}  datasets={datasets}")

    pair_results = []
    for train_name in datasets:
        for test_name in datasets:
            if train_name == test_name:
                continue
            pair_results.append(run_pair(model_name, train_name, test_name, device))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": model_name,
        "num_classes": NUM_CLASSES_EXTERNAL,
        "datasets": datasets,
        "pairs": pair_results,
    }
    out_path = OUTPUTS_DIR / f"external_validation_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[{model_name}] Resultados salvos em: {out_path}")

    print(f"\n[{model_name}] === Resumo (test externo) ===")
    for r in pair_results:
        m = r["test_metrics"]
        print(
            f"  {r['train_dataset']:>8} -> {r['test_dataset']:<8}"
            f"  acc={m['accuracy']:.4f}"
            f"  f1={m['f1_macro']:.4f}"
            f"  auc={m['auc_roc_macro_ovr']:.4f}"
        )

    return pair_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_BUILDERS.keys()) + ["all"],
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASET_BUILDERS.keys()),
        choices=list(DATASET_BUILDERS.keys()),
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in MODEL_BUILDERS.keys():
            run_external_validation(name, args.datasets)
    else:
        run_external_validation(args.model, args.datasets)
