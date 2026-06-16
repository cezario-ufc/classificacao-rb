import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    EPOCHS_FINAL,
    EPOCHS_GRIDSEARCH,
    IMG_SIZE,
    N_SPLITS,
    NUM_WORKERS,
    OUTPUTS_DIR,
    PARAM_GRID,
    SEED,
    USE_AUGMENT,
)
from src.data.build_dataset_ddr import build_classification_dataframe_ddr
from src.data.build_dataset_mesidor import build_classification_dataframe_mesidor
from src.data.dataset import RetinopathyDataset
from src.data.splits_kfold import get_fold_dfs, make_kfold_splits
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.builders import MODEL_BUILDERS
from src.training.train import evaluate, train_one_epoch


def get_dataset_config(dataset: str):
    if dataset == "ddr":
        return build_classification_dataframe_ddr, 5
    if dataset == "mesidor":
        return build_classification_dataframe_mesidor, 4
    raise ValueError(f"Dataset desconhecido: {dataset}")


def make_loader(df, transform, shuffle):
    ds = RetinopathyDataset(df, transform=transform)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def run_pipeline(model_name: str, dataset: str = "ddr"):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{dataset}/{model_name}"
    print(f"[{tag}] device={device}")

    builder, num_classes = get_dataset_config(dataset)
    df = builder()
    splits_df = make_kfold_splits(df)

    train_tf = get_train_transforms(IMG_SIZE, augment=USE_AUGMENT)
    eval_tf = get_eval_transforms(IMG_SIZE)

    test_scores = []
    fold_results = []

    for fold_id in range(N_SPLITS):
        print(f"\n[{tag}] === Fold {fold_id} ===")
        train_df, val_df, test_df = get_fold_dfs(splits_df, fold_id)

        trainDivided_loader = make_loader(train_df, train_tf, shuffle=True)
        val_loader = make_loader(val_df, eval_tf, shuffle=False)

        f1s_val = []
        val_metrics_all = []
        par = []

        for params in ParameterGrid(PARAM_GRID):
            print(f"  params={params}")
            model = MODEL_BUILDERS[model_name](num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )

            for _ in range(EPOCHS_GRIDSEARCH):
                train_one_epoch(model, trainDivided_loader, criterion, optimizer, device)

            val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)
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
            f"  val_acc={best_val_metrics['accuracy']:.4f}"
        )

        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        full_train_loader = make_loader(full_train_df, train_tf, shuffle=True)
        test_loader = make_loader(test_df, eval_tf, shuffle=False)

        model_best = MODEL_BUILDERS[model_name](num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model_best.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        for _ in range(EPOCHS_FINAL):
            train_one_epoch(model_best, full_train_loader, criterion, optimizer, device)

        test_metrics = evaluate(model_best, test_loader, device, num_classes=num_classes)
        print(
            f"  test_acc={test_metrics['accuracy']:.4f}"
            f" precision={test_metrics['precision_macro']:.4f}"
            f" recall={test_metrics['recall_macro']:.4f}"
            f" f1={test_metrics['f1_macro']:.4f}"
            f" auc={test_metrics['auc_roc_macro_ovr']:.4f}"
        )
        test_scores.append(test_metrics["accuracy"])
        fold_results.append({
            "fold": fold_id,
            "best_params": best_params,
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        })

        del model_best, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_acc = sum(test_scores) / len(test_scores)
    scalar_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_roc_macro_ovr",
    ]
    mean_test_metrics = {
        k: sum(f["test_metrics"][k] for f in fold_results) / len(fold_results)
        for k in scalar_keys
    }
    print(f"\n[{tag}] Metricas medias nos conjuntos de teste:")
    for k, v in mean_test_metrics.items():
        print(f"  {k}={v:.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": dataset,
        "model": model_name,
        "num_classes": num_classes,
        "test_scores": test_scores,
        "mean_test_acc": mean_acc,
        "mean_test_metrics": mean_test_metrics,
        "folds": fold_results,
    }
    out_path = OUTPUTS_DIR / f"results_{dataset}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] Resultados salvos em: {out_path}")

    return mean_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_BUILDERS.keys()) + ["all"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ddr",
        choices=["ddr", "mesidor"],
    )
    args = parser.parse_args()

    if args.model == "all":
        results = {}
        for name in MODEL_BUILDERS.keys():
            results[name] = run_pipeline(name, dataset=args.dataset)
        print(f"\n=== Resumo final ({args.dataset}) ===")
        for name, acc in results.items():
            print(f"{name}: {acc:.4f}")
    else:
        run_pipeline(args.model, dataset=args.dataset)
