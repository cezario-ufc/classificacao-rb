"""Metadados de checkpoints do Ultralytics (best.pt/last.pt) — para retomada e auditoria.

O Ultralytics só "finaliza" um checkpoint quando o treino TERMINA (épocas esgotadas ou
early-stopping): strip_optimizer grava epoch=-1 e optimizer=None. Durante o treino, best.pt
é regravado a cada melhora com epoch>=0 e o estado do otimizador. Logo, um best.pt não
finalizado é de um treino interrompido e NÃO deve ser reaproveitado.

A leitura não exige torch/ultralytics: decodifica só o data.pkl do zip, trocando tensores e
classes por um stub (funciona em máquinas sem GPU/torch). Se falhar (formato inesperado),
cai para torch.load quando o torch estiver disponível.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path


class _Stub:
    """Substitui classes do torch/ultralytics ao despicklar: só queremos os metadados."""

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _MetaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("builtins", "collections", "datetime", "_codecs"):
            return super().find_class(module, name)
        return _Stub

    def persistent_load(self, pid):  # tensores: não carrega
        return None


def read_ckpt_meta(path) -> dict:
    """Dict do checkpoint com metadados utilizáveis (epoch, date, optimizer, train_args)."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as z:
            pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
            return _MetaUnpickler(z.open(pkl)).load()
    except Exception:  # noqa: BLE001  formato inesperado -> tenta o caminho oficial
        import torch
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch antigo sem weights_only
            return torch.load(path, map_location="cpu")


def meta_is_finalized(ck: dict) -> bool:
    """True se os metadados (de read_ckpt_meta) são de um checkpoint finalizado.

    epoch=-1 e optimizer=None são gravados juntos pelo strip_optimizer; checkpoints de
    meio de treino sempre têm epoch>=0 e o estado do otimizador. Aceita qualquer um dos
    dois sinais para não depender de um detalhe de versão do Ultralytics.
    """
    return ck.get("epoch") == -1 or ck.get("optimizer") is None


def is_finalized(path) -> bool:
    """True se o checkpoint em `path` foi finalizado pelo fim normal do treino."""
    return meta_is_finalized(read_ckpt_meta(path))
