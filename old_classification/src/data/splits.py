import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.config import SEED, VAL_RATIO, N_SPLITS, OUTPUTS_DIR, SPLITS_CSV
from src.data.build_dataset import build_classification_dataframe


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    pool_idx, val_idx = train_test_split(
        df.index,
        test_size=VAL_RATIO,
        stratify=df["label"],
        random_state=SEED,
    )

    df["is_validation"] = False
    df.loc[val_idx, "is_validation"] = True
    df["fold"] = -1

    pool_df = df.loc[pool_idx]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold_id, (_, test_local_pos) in enumerate(skf.split(pool_df, pool_df["label"])):
        global_test_idx = pool_df.index[test_local_pos]
        df.loc[global_test_idx, "fold"] = fold_id

    return df


def get_validation_df(splits_df: pd.DataFrame) -> pd.DataFrame:
    return splits_df[splits_df["is_validation"]].reset_index(drop=True)


def get_fold_dfs(splits_df: pd.DataFrame, fold_id: int):
    pool = splits_df[~splits_df["is_validation"]]
    test_df = pool[pool["fold"] == fold_id].reset_index(drop=True)
    train_df = pool[pool["fold"] != fold_id].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    df = build_classification_dataframe()
    splits_df = make_splits(df)

    print(f"Total: {len(splits_df)} imagens\n")

    print("=== Holdout (validacao) ===")
    val_df = get_validation_df(splits_df)
    print(f"Tamanho: {len(val_df)} ({len(val_df)/len(splits_df):.1%})")
    print("Distribuicao por classe:")
    print(val_df["label"].value_counts(normalize=True).sort_index().round(3))

    print("\n=== Pool (vai pro K-Fold) ===")
    pool_df = splits_df[~splits_df["is_validation"]]
    print(f"Tamanho: {len(pool_df)} ({len(pool_df)/len(splits_df):.1%})")
    print("Distribuicao por classe:")
    print(pool_df["label"].value_counts(normalize=True).sort_index().round(3))

    print("\n=== Folds ===")
    for fold_id in range(N_SPLITS):
        train_df, test_df = get_fold_dfs(splits_df, fold_id)
        test_dist = test_df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f"Fold {fold_id}: train={len(train_df):>5}  test={len(test_df):>5}  test_dist={test_dist}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(SPLITS_CSV, index=False)
    print(f"\nSplits salvos em: {SPLITS_CSV}")
