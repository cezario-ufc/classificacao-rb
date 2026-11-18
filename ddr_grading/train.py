"""Fine-tuning do backbone por fold, extração de embeddings, cabeças MLP e métricas.

Critério de seleção em tudo (early-stopping e escolha da combinação da grade): AUC-ROC macro
one-vs-rest no val interno do fold. O teste do fold não entra em nenhuma decisão.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ddr_grading.data import CACHE_DIR, CACHE_SIZE
from ddr_grading.models import MLPHead

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fine-tuning: Adam e CE simples como no old_classification; lr/batch/weight decay vêm da grade
# (scripts/07_grading_cv.py); nº de épocas por early-stopping na AUC do val.
# micro_batch: pedaço que cabe na GPU (VGG19/ResNet152 em 512 px não cabem com 32 numa 3090);
# batches maiores acumulam gradiente até o batch da grade.
FT = {"max_epochs": 30, "patience": 5, "workers": 8, "micro_batch": 16}
# Cabeças MLP sobre embeddings (segundos por combinação).
MLP = {"max_epochs": 300, "patience": 30}
SELECT = "auc_ovr_macro"


# --------------------------------------------------------------------------- dados
def _resize(size):
    return [] if size == CACHE_SIZE else [transforms.Resize((size, size))]


def train_tf(size):
    return transforms.Compose(_resize(size) + [
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_tf(size):
    return transforms.Compose(_resize(size) + [
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class CachedImages(Dataset):
    def __init__(self, names, labels, tf):
        self.names, self.labels, self.tf = list(names), list(labels), tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(CACHE_DIR / self.names[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


def loader(names, labels, tf, shuffle, batch=None, generator=None):
    return DataLoader(CachedImages(names, labels, tf), batch_size=batch or FT["micro_batch"],
                      shuffle=shuffle, generator=generator, num_workers=FT["workers"],
                      pin_memory=True, drop_last=False)


# --------------------------------------------------------------------------- métricas
def metrics(y_true, prob, num_classes):
    """AUC-ROC, sensibilidade, F1, precisão e especificidade com média macro entre as classes
    (cada classe contra as demais), e acurácia. Especificidade da classe c = TN/(TN+FP)."""
    y_true = np.asarray(y_true)
    y_pred = prob.argmax(1)
    labels = list(range(num_classes))
    try:
        auc = roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(1) - fp
    spec = np.divide(tn, tn + fp, out=np.zeros(num_classes), where=(tn + fp) > 0)
    kw = dict(average="macro", labels=labels, zero_division=0)
    return {
        "auc_ovr_macro": auc,
        "sensitivity_macro": recall_score(y_true, y_pred, **kw),
        "f1_macro": f1_score(y_true, y_pred, **kw),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "specificity_macro": float(spec.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


# --------------------------------------------------------------------------- fine-tuning
def _amp(device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_probs(net, dl, device):
    net.eval()
    out = []
    for x, _ in dl:
        with _amp(device):
            out.append(torch.softmax(net(x.to(device, non_blocking=True)).float(), 1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed(net, dl, device):
    net.eval()
    out = []
    for x, _ in dl:
        with _amp(device):
            out.append(net.features(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def finetune(net, tr_names, tr_labels, va_dl, va_labels, num_classes, hp, input_size, device,
             ckpt_dir: Path, log=print):
    """Treina uma combinação hp={'lr','batch','weight_decay'} com early-stopping na AUC do val.
    Devolve o conteúdo do done.json ({best_epoch, val_auc_ovr_macro, val_f1_macro});
    best.pt = pesos da melhor época.

    Retomável: last.pt (pesos, otimizador, scaler, época, melhor até ali, paciência) a cada
    época; best.pt ao melhorar; done.json ao terminar (aí last.pt é apagado)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt, last_pt, done = ckpt_dir / "best.pt", ckpt_dir / "last.pt", ckpt_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    micro = min(hp["batch"], FT["micro_batch"])
    if hp["batch"] % micro:
        raise ValueError(f"batch {hp['batch']} não é múltiplo do micro-batch {micro}")
    accum = hp["batch"] // micro
    gen = torch.Generator()
    tr_dl = loader(tr_names, tr_labels, train_tf(input_size), shuffle=True, batch=micro,
                   generator=gen)

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss()
    state = {"epoch": -1, "best": -1.0, "best_epoch": -1, "best_f1": None, "bad": 0}
    if last_pt.exists():
        ck = torch.load(last_pt, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"    [resume] continua da época {state['epoch'] + 2} "
            f"(melhor AUC val {state['best']:.4f} na {state['best_epoch'] + 1})")

    for epoch in range(state["epoch"] + 1, FT["max_epochs"]):
        if state["bad"] >= FT["patience"]:
            break
        torch.manual_seed(42 + epoch)       # augmentação/embaralhamento reprodutíveis na retomada
        gen.manual_seed(42 + epoch)
        net.train()
        tot, n, nb = 0.0, 0, len(tr_dl)
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _amp(device):
                loss = crit(net(x), y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == nb:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot, n = tot + loss.item() * len(y), n + len(y)
        m = metrics(va_labels, predict_probs(net, va_dl, device), num_classes)
        state["epoch"] = epoch
        if m[SELECT] > state["best"]:
            state.update(best=m[SELECT], best_epoch=epoch, best_f1=m["f1_macro"], bad=0)
            torch.save(net.state_dict(), best_pt)
        else:
            state["bad"] += 1
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, last_pt)
        log(f"    época {epoch + 1}: loss {tot / n:.4f} | val AUC {m[SELECT]:.4f} "
            f"F1 {m['f1_macro']:.4f} (melhor AUC {state['best']:.4f} na {state['best_epoch'] + 1})")

    out = {"best_epoch": state["best_epoch"] + 1, "val_auc_ovr_macro": state["best"],
           "val_f1_macro": state["best_f1"]}
    done.write_text(json.dumps(out))
    last_pt.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- cabeças MLP
def _std(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train, *others)]


def _t(arrays, device):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]


def fit_mlp(emb, les, y, num_classes, hp, device, seed=42):
    """emb/les/y: dicts {'train','val','test'} (les=None -> só embedding); hp={'lr','batch',
    'weight_decay'}. Padroniza com o treino, treina com early-stopping na AUC do val e devolve
    (probabilidades do teste, {best_epoch, val_auc_ovr_macro, val_f1_macro})."""
    torch.manual_seed(seed)
    e_tr, e_va, e_te = _t(_std(emb["train"], emb["val"], emb["test"]), device)
    if les is not None:
        l_tr, l_va, l_te = _t(_std(les["train"], les["val"], les["test"]), device)
    else:
        l_tr = l_va = l_te = None
    y_tr = torch.tensor(np.asarray(y["train"]), device=device)
    head = MLPHead(e_tr.shape[1], num_classes,
                   lesion_dim=0 if les is None else l_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = nn.CrossEntropyLoss()

    def probs(e, l):
        head.eval()
        with torch.no_grad():
            return torch.softmax(head(e, l), 1).cpu().numpy()

    best, best_val, best_state, bad = -1.0, None, None, 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(MLP["max_epochs"]):
        head.train()
        perm = torch.randperm(len(y_tr), generator=g).to(device)
        for i in range(0, len(perm), hp["batch"]):
            b = perm[i:i + hp["batch"]]
            opt.zero_grad(set_to_none=True)
            loss = crit(head(e_tr[b], None if l_tr is None else l_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        m = metrics(y["val"], probs(e_va, l_va), num_classes)
        if m[SELECT] > best:
            best, bad = m[SELECT], 0
            best_val = {"best_epoch": epoch + 1, "val_auc_ovr_macro": m[SELECT],
                        "val_f1_macro": m["f1_macro"]}
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP["patience"]:
                break
    head.load_state_dict(best_state)
    return probs(e_te, l_te), best_val
