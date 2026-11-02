"""Evaluation metrics. The ISIC 2018 Task 3 official metric is balanced multiclass accuracy (mean recall)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

from .data import CLASSES, NUM_CLASSES


def compute_metrics(y_true, probs) -> dict:
    """Metrics from integer labels (N,) and class probabilities (N, C)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    labels = np.arange(NUM_CLASSES)

    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Specificity for melanoma: true negatives / all non-melanoma cases.
    mel = CLASSES.index("MEL")
    non_mel = y_true != mel
    mel_specificity = float((y_pred[non_mel] != mel).mean()) if non_mel.any() else float("nan")

    present = np.unique(y_true)
    try:
        macro_auc = roc_auc_score(y_true, probs[:, present] / probs[:, present].sum(1, keepdims=True),
                                  multi_class="ovr", average="macro", labels=present) if len(present) > 1 else float("nan")
    except ValueError:
        macro_auc = float("nan")

    return {
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_auc": float(macro_auc),
        "mel_sensitivity": float(recalls[mel]),
        "mel_specificity": mel_specificity,
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, recalls)},
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def format_metrics(m: dict, name: str = "") -> str:
    head = f"{name:<14s} " if name else ""
    line = (f"{head}BalAcc {m['balanced_acc']:.4f} | Acc {m['accuracy']:.4f} | F1 {m['macro_f1']:.4f} | "
            f"AUC {m['macro_auc']:.4f} | MEL sens {m['mel_sensitivity']:.4f} spec {m['mel_specificity']:.4f}")
    recalls = "  ".join(f"{c}:{r:.2f}" for c, r in m["per_class_recall"].items())
    return f"{line}\n{' ' * len(head)}recall  {recalls}"
