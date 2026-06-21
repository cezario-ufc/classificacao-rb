import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import IMG_SIZE, OUTPUTS_DIR
from src.data.build_dataset_ddr import build_classification_dataframe_ddr
from src.data.channel_pipeline import ChannelDecomposition
from src.data.dataset import RetinopathyDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_train_transforms


def denormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def preview_batch(augment: bool, n: int = 8, seed: int = 0):
    df = build_classification_dataframe_ddr().sample(n=n, random_state=seed)
    tf = get_train_transforms(IMG_SIZE, augment=augment)
    loader = DataLoader(RetinopathyDataset(df, transform=tf), batch_size=n)
    imgs, labels = next(iter(loader))

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for ax, img, y in zip(axes, imgs, labels):
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"label={int(y)}")
        ax.axis("off")
    fig.suptitle(f"augment={augment}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"preview_batch_augment_{augment}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def preview_channel_decomp(n: int = 6, seed: int = 0):
    """Mostra original + cada canal decomposto separadamente para validacao visual."""
    df = build_classification_dataframe_ddr().sample(n=n, random_state=seed)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ChannelDecomposition(),
        transforms.ToTensor(),
    ])
    orig_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ds_orig = RetinopathyDataset(df, transform=orig_tf)
    ds_dec = RetinopathyDataset(df, transform=tf)
    orig_loader = DataLoader(ds_orig, batch_size=n)
    dec_loader = DataLoader(ds_dec, batch_size=n)

    orig_imgs, labels = next(iter(orig_loader))
    dec_imgs, _ = next(iter(dec_loader))

    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    row_titles = ["original RGB", "R: iluminacao", "G: CLAHE", "B: top-hat"]
    for j in range(n):
        axes[0, j].imshow(orig_imgs[j].permute(1, 2, 0).numpy())
        axes[0, j].set_title(f"label={int(labels[j])}")
        axes[0, j].axis("off")
        for i in range(3):
            axes[i + 1, j].imshow(dec_imgs[j, i].numpy(), cmap="gray")
            axes[i + 1, j].axis("off")
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=10)
    fig.suptitle("Decomposicao de canais para realce de exsudatos")
    fig.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "preview_channel_decomp.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    preview_batch(augment=False)
    preview_batch(augment=True)
    preview_channel_decomp()
