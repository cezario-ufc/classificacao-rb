from pathlib import Path

import pandas as pd

from src.config import (
    CLASSIFICATION_INDEX_CSV,
    MESSIDOR_DIR,
    OUTPUTS_DIR,
)


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


def build_classification_dataframe_mesidor() -> pd.DataFrame:
    parts = []
    for base_dir in sorted(MESSIDOR_DIR.glob("Base*")):
        if base_dir.is_dir():
            parts.append(_load_messidor_base(base_dir))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    df = build_classification_dataframe_mesidor()

    print(f"Total de imagens (Messidor): {len(df)}\n")

    print("Por classe:")
    print(df["label"].value_counts().sort_index())

    print("\nPor base de origem:")
    print(df["split_origem"].value_counts().sort_index())

    print("\nExemplo:")
    print(df.iloc[0].to_dict())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLASSIFICATION_INDEX_CSV, index=False)
    print(f"\nIndice salvo em: {CLASSIFICATION_INDEX_CSV}")
