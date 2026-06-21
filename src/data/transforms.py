from torchvision import transforms

from src.data.channel_pipeline import ChannelDecomposition

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int,
    augment: bool = True,
    use_channel_decomp: bool = False,
    mean=None,
    std=None,
):
    if not augment:
        return get_eval_transforms(
            img_size,
            use_channel_decomp=use_channel_decomp,
            mean=mean,
            std=std,
        )

    mean = mean if mean is not None else IMAGENET_MEAN
    std = std if std is not None else IMAGENET_STD

    steps = [transforms.Resize((img_size, img_size))]
    if use_channel_decomp:
        steps.append(ChannelDecomposition())
    steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    if not use_channel_decomp:
        steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(steps)


def get_eval_transforms(
    img_size: int,
    use_channel_decomp: bool = False,
    mean=None,
    std=None,
):
    mean = mean if mean is not None else IMAGENET_MEAN
    std = std if std is not None else IMAGENET_STD

    steps = [transforms.Resize((img_size, img_size))]
    if use_channel_decomp:
        steps.append(ChannelDecomposition())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(steps)
