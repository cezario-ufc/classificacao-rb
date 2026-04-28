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
)
from src.data.build_dataset_ddr import build_classification_dataframe_ddr
from src.data.dataset import RetinopathyDataset
from src.data.splits_kfold import get_fold_dfs, make_kfold_splits
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.builders import MODEL_BUILDERS
from src.training.train import evaluate, train_one_epoch


def make_loader(df, transform, shuffle):
    ds = RetinopathyDataset(df, transform=transform)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def run_pipeline(model_name: str):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device={device}")

    df = build_classification_dataframe_ddr()
    splits_df = make_kfold_splits(df)

    train_tf = get_train_transforms(IMG_SIZE)
    eval_tf = get_eval_transforms(IMG_SIZE)

    test_scores = []
    fold_results = []

    for fold_id in range(N_SPLITS):
        print(f"\n[{model_name}] === Fold {fold_id} ===")
        train_df, val_df, test_df = get_fold_dfs(splits_df, fold_id)

        trainDivided_loader = make_loader(train_df, train_tf, shuffle=True)
        val_loader = make_loader(val_df, eval_tf, shuffle=False)

        accs_val = []
        par = []

        for params in ParameterGrid(PARAM_GRID):
            print(f"  params={params}")
            model = MODEL_BUILDERS[model_name]().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )

            for _ in range(EPOCHS_GRIDSEARCH):
                train_one_epoch(model, trainDivided_loader, criterion, optimizer, device)

            acc = evaluate(model, val_loader, device)
            print(f"    val_acc={acc:.4f}")
            accs_val.append(acc)
            par.append(params)

            del model, optimizer, criterion
            if device.type == "cuda":
                torch.cuda.empty_cache()

        best_idx = accs_val.index(max(accs_val))
        best_params = par[best_idx]
        print(f"  Melhores hiperparametros: {best_params}  val_acc={accs_val[best_idx]:.4f}")

        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        full_train_loader = make_loader(full_train_df, train_tf, shuffle=True)
        test_loader = make_loader(test_df, eval_tf, shuffle=False)

        model_best = MODEL_BUILDERS[model_name]().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model_best.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        for _ in range(EPOCHS_FINAL):
            train_one_epoch(model_best, full_train_loader, criterion, optimizer, device)

        test_acc = evaluate(model_best, test_loader, device)
        print(f"  test_acc={test_acc:.4f}")
        test_scores.append(test_acc)
        fold_results.append({
            "fold": fold_id,
            "best_params": best_params,
            "val_acc": accs_val[best_idx],
            "test_acc": test_acc,
        })

        del model_best, optimizer, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_acc = sum(test_scores) / len(test_scores)
    print(f"\n[{model_name}] Acuracia media nos conjuntos de testes: {mean_acc:.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": model_name,
        "test_scores": test_scores,
        "mean_test_acc": mean_acc,
        "folds": fold_results,
    }
    out_path = OUTPUTS_DIR / f"results_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{model_name}] Resultados salvos em: {out_path}")

    return mean_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_BUILDERS.keys()) + ["all"],
    )
    args = parser.parse_args()

    if args.model == "all":
        results = {}
        for name in MODEL_BUILDERS.keys():
            results[name] = run_pipeline(name)
        print("\n=== Resumo final ===")
        for name, acc in results.items():
            print(f"{name}: {acc:.4f}")
    else:
        run_pipeline(args.model)
