"""Backbones (torchvision, pesos do ImageNet) e as cabeças MLP.

Cada backbone vira Net = features (o backbone sem a última camada linear) + fc (linear nova
para as classes do DDR). O embedding é a saída de features: a entrada da camada de
classificação original.

Entrada: 512 px (o cache), como no old_classification; o ViT-B/16 usa os pesos SWAG E2E, treinados
em 384 px (o torchvision fixa image_size=384 com eles). Grade de lr por família (lr_grid): as
CNNs em torno do 1e-4 do old_classification; os transformers, uma ordem abaixo, receita usual
para não destruir o pré-treino. batch e weight decay têm a mesma grade para todos (07).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

_CNN_LR, _VIT_LR = [1e-4, 3e-5], [3e-5, 1e-5]
BACKBONES = {
    "vgg19":           {"input": 512, "lr_grid": _CNN_LR},
    "resnet152":       {"input": 512, "lr_grid": _CNN_LR},
    "efficientnet_b0": {"input": 512, "lr_grid": _CNN_LR},
    "mobilenet_v3":    {"input": 512, "lr_grid": _CNN_LR},
    "vit_b_16":        {"input": 384, "lr_grid": _VIT_LR},
    "swin_t":          {"input": 512, "lr_grid": _VIT_LR},
}


def _ensure_writable_torch_home():
    """O /app da imagem pode não ser gravável (o Ultralytics já avisa isso): baixa os pesos do
    torchvision em runs/ (volume gravável) se TORCH_HOME não aceitar escrita."""
    home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    try:
        home.mkdir(parents=True, exist_ok=True)
        ok = os.access(home, os.W_OK)
    except OSError:
        ok = False
    if not ok:
        alt = Path(__file__).resolve().parents[1] / "runs" / ".torch_cache"
        alt.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(alt / "hub"))


def _backbone(name):
    """(módulo sem a última linear, dimensão do embedding)."""
    from torchvision import models as tvm

    _ensure_writable_torch_home()
    if name == "vgg19":
        m = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        d, m.classifier[-1] = m.classifier[-1].in_features, nn.Identity()
    elif name == "resnet152":
        m = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V2)
        d, m.fc = m.fc.in_features, nn.Identity()
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        d, m.classifier[-1] = m.classifier[-1].in_features, nn.Identity()
    elif name == "mobilenet_v3":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        d, m.classifier[-1] = m.classifier[-1].in_features, nn.Identity()
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)   # 384 px
        d, m.heads.head = m.heads.head.in_features, nn.Identity()
    elif name == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1)
        d, m.head = m.head.in_features, nn.Identity()
    else:
        raise ValueError(f"backbone desconhecido: {name}")
    return m, d


class Net(nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        self.features, self.emb_dim = _backbone(name)
        self.fc = nn.Linear(self.emb_dim, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


class MLPHead(nn.Module):
    """Cabeça sobre embeddings congelados. lesion_dim=0 -> só o embedding (controle);
    >0 -> o vetor de lesões passa por uma camada pequena e é concatenado ao embedding."""

    def __init__(self, emb_dim, num_classes, lesion_dim=0, hidden=256, lesion_hidden=64,
                 dropout=0.3):
        super().__init__()
        self.emb = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, hidden), nn.ReLU())
        self.les = (nn.Sequential(nn.Linear(lesion_dim, lesion_hidden), nn.ReLU())
                    if lesion_dim else None)
        self.out = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(hidden + (lesion_hidden if lesion_dim else 0),
                                           num_classes))

    def forward(self, emb, les=None):
        h = self.emb(emb)
        if self.les is not None:
            h = torch.cat([h, self.les(les)], dim=1)
        return self.out(h)
