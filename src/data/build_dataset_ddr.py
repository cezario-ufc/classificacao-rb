from pathlib import Path

import pandas as pd

from src.config import (
    CLASSES_TO_DROP,
    CLASSIFICATION_INDEX_CSV,
    DDR_DIR,
    OUTPUTS_DIR,
)


def _parse_ddr_label_file(txt_path: Path, images_dir: Path, split_name: str) -> pd.DataFrame:
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, label = line.rsplit(None, 1)
            rows.append({
                "path": str(images_dir / filename),
                "label": int(label),
                "dataset": "ddr",
                "split_origem": split_name,
            })
    return pd.DataFrame(rows)


def build_classification_dataframe_ddr() -> pd.DataFrame:
    parts = []
    for split in ["train", "test", "valid"]:
        txt = DDR_DIR / f"{split}.txt"
        img_dir = DDR_DIR / split
        parts.append(_parse_ddr_label_file(txt, img_dir, split))

    df = pd.concat(parts, ignore_index=True)
    df = df[~df["label"].isin(CLASSES_TO_DROP)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_classification_dataframe_ddr()

    print(f"Total de imagens (DDR): {len(df)}\n")

    print("Por classe:")
    print(df["label"].value_counts().sort_index())

    print("\nPor split de origem:")
    print(df["split_origem"].value_counts())

    print("\nExemplo:")
    print(df.iloc[0].to_dict())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLASSIFICATION_INDEX_CSV, index=False)
    print(f"\nIndice salvo em: {CLASSIFICATION_INDEX_CSV}")
