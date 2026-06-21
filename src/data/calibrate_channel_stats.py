import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import IMG_SIZE, NUM_WORKERS, OUTPUTS_DIR
from src.data.build_dataset_ddr import build_classification_dataframe_ddr
from src.data.build_dataset_mesidor import build_classification_dataframe_mesidor
from src.data.channel_pipeline import ChannelDecomposition
from src.data.dataset import RetinopathyDataset

DATASET_BUILDERS = {
    "ddr": build_classification_dataframe_ddr,
    "mesidor": build_classification_dataframe_mesidor,
}


def build_stats_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        ChannelDecomposition(),
        transforms.ToTensor(),
    ])


def compute_stats(dataset_name: str, batch_size: int = 32):
    df = DATASET_BUILDERS[dataset_name]()
    print(f"[{dataset_name}] n={len(df)}")

    tf = build_stats_transform(IMG_SIZE)
    ds = RetinopathyDataset(df, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    n_pixels = 0
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)

    for i, (x, _) in enumerate(loader, start=1):
        b, c, h, w = x.shape
        x = x.to(torch.float64)
        n_pixels += b * h * w
        sum_c += x.sum(dim=(0, 2, 3))
        sum_sq_c += (x ** 2).sum(dim=(0, 2, 3))
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}")

    mean = (sum_c / n_pixels).tolist()
    var = (sum_sq_c / n_pixels) - torch.tensor(mean, dtype=torch.float64) ** 2
    std = torch.sqrt(var.clamp(min=0)).tolist()

    return {"mean": mean, "std": std, "n_images": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASET_BUILDERS.keys()) + ["all"],
    )
    args = parser.parse_args()

    targets = list(DATASET_BUILDERS.keys()) if args.dataset == "all" else [args.dataset]
    results = {}
    for name in targets:
        stats = compute_stats(name)
        results[name] = stats
        print(f"\n[{name}] mean={stats['mean']}")
        print(f"[{name}] std={stats['std']}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "channel_decomp_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvo em: {out_path}")
    print("Copie os valores para CHANNEL_DECOMP_STATS em src/config.py.")


if __name__ == "__main__":
    main()
