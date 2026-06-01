import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import NUM_CLASSES


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_prob.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)
    labels = list(range(NUM_CLASSES))

    try:
        auc_roc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels
        )
    except ValueError:
        auc_roc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "auc_roc_macro_ovr": auc_roc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }