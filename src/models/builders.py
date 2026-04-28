import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_mobilenet_v3():
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_efficientnet_b0():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_resnet152():
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vgg19():
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


MODEL_BUILDERS = {
    "mobilenet_v3": build_mobilenet_v3,
    "efficientnet_b0": build_efficientnet_b0,
    "resnet152": build_resnet152,
    "vgg19": build_vgg19,
}
