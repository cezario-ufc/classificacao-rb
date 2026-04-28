import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import N_SPLITS, OUTPUTS_DIR, SEED, SPLITS_CSV, VAL_RATIO
from src.data.build_dataset_ddr import build_classification_dataframe_ddr


def make_kfold_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold_id, (_, test_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[test_idx, "fold"] = fold_id

    return df


def get_fold_dfs(splits_df: pd.DataFrame, fold_id: int):
    test_df = splits_df[splits_df["fold"] == fold_id].reset_index(drop=True)
    train_pool = splits_df[splits_df["fold"] != fold_id].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_pool,
        test_size=VAL_RATIO,
        stratify=train_pool["label"],
        random_state=SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df,
    )


if __name__ == "__main__":
    df = build_classification_dataframe_ddr()
    splits_df = make_kfold_splits(df)

    print(f"Total: {len(splits_df)} imagens (DDR)\n")

    print("Distribuicao geral por classe:")
    print(splits_df["label"].value_counts(normalize=True).sort_index().round(3))

    print("\n=== Folds ===")
    for fold_id in range(N_SPLITS):
        train_df, val_df, test_df = get_fold_dfs(splits_df, fold_id)
        test_dist = test_df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        val_dist = val_df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(
            f"Fold {fold_id}: "
            f"train={len(train_df):>5}  val={len(val_df):>5}  test={len(test_df):>5}  "
            f"val_dist={val_dist}  test_dist={test_dist}"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(SPLITS_CSV, index=False)
    print(f"\nSplits salvos em: {SPLITS_CSV}")
