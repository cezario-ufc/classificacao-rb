from pathlib import Path

import pandas as pd

from src.config import (
    DDR_DIR,
    MESSIDOR_DIR,
    IDRID_DIR,
    CLASSES_TO_DROP,
    DDR_LABEL_REMAP,
    IDRID_LABEL_REMAP,
    OUTPUTS_DIR,
    CLASSIFICATION_INDEX_CSV,
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


def build_ddr_dataframe() -> pd.DataFrame:
    parts = []
    for split in ["train", "test", "valid"]:
        txt = DDR_DIR / f"{split}.txt"
        img_dir = DDR_DIR / split
        parts.append(_parse_ddr_label_file(txt, img_dir, split))

    df = pd.concat(parts, ignore_index=True)
    df = df[~df["label"].isin(CLASSES_TO_DROP)].reset_index(drop=True)

    for old, new in DDR_LABEL_REMAP.items():
        df.loc[df["label"] == old, "label"] = new

    return df


def _load_messidor_base(base_dir: Path) -> pd.DataFrame:
    xls_files = list(base_dir.rglob("*.xls"))
    if len(xls_files) != 1:
        raise RuntimeError(
            f"Esperava 1 arquivo .xls em {base_dir}, encontrei {len(xls_files)}"
        )
    xls = xls_files[0]
    images_dir = xls.parent

    df = pd.read_excel(xls)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    return pd.DataFrame({
        "path": df["Image name"].apply(lambda fn: str(images_dir / fn)),
        "label": df["Retinopathy grade"].astype(int),
        "dataset": "messidor",
        "split_origem": base_dir.name,
    })


def build_messidor_dataframe() -> pd.DataFrame:
    parts = []
    for base_dir in sorted(MESSIDOR_DIR.glob("Base*")):
        if base_dir.is_dir():
            parts.append(_load_messidor_base(base_dir))
    return pd.concat(parts, ignore_index=True)


def _load_idrid_split(csv_path: Path, images_dir: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["Image name", "Retinopathy grade"])
    df = df.dropna(subset=["Retinopathy grade"])

    return pd.DataFrame({
        "path": df["Image name"].apply(lambda fn: str(images_dir / f"{fn}.jpg")),
        "label": df["Retinopathy grade"].astype(int),
        "dataset": "idrid",
        "split_origem": split_name,
    })


def build_idrid_dataframe() -> pd.DataFrame:
    images_root = IDRID_DIR / "1. Original Images"
    labels_root = IDRID_DIR / "2. Groundtruths"

    splits = [
        ("train",
         images_root / "a. Training Set",
         labels_root / "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("test",
         images_root / "b. Testing Set",
         labels_root / "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ]

    parts = [_load_idrid_split(csv, img_dir, name) for name, img_dir, csv in splits]
    df = pd.concat(parts, ignore_index=True)

    for old, new in IDRID_LABEL_REMAP.items():
        df.loc[df["label"] == old, "label"] = new

    return df


def build_classification_dataframe() -> pd.DataFrame:
    df_ddr = build_ddr_dataframe()
    df_messidor = build_messidor_dataframe()
    df_idrid = build_idrid_dataframe()
    return pd.concat([df_ddr, df_messidor, df_idrid], ignore_index=True)


if __name__ == "__main__":
    df = build_classification_dataframe()

    print(f"Total de imagens: {len(df)}\n")

    print("Por dataset:")
    print(df["dataset"].value_counts())

    print("\nPor classe (apos remap DDR 4->3):")
    print(df["label"].value_counts().sort_index())

    print("\nDataset x classe:")
    print(pd.crosstab(df["dataset"], df["label"], margins=True))

    print("\nExemplo DDR:")
    print(df[df["dataset"] == "ddr"].iloc[0].to_dict())
    print("\nExemplo Messidor:")
    print(df[df["dataset"] == "messidor"].iloc[0].to_dict())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLASSIFICATION_INDEX_CSV, index=False)
    print(f"\nIndice salvo em: {CLASSIFICATION_INDEX_CSV}")
