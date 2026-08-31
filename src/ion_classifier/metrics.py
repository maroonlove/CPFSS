"""Metrics for hierarchical classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_macro_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision_macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
    }


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 9) -> pd.DataFrame:
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()

    rows = []
    for cls in labels:
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = total - tp - fp - fn

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class": int(cls),
                "support": int(cm[cls, :].sum()),
                "accuracy_one_vs_rest": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def overall_metrics_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "precision_micro", "value": precision_score(y_true, y_pred, average="micro", zero_division=0)},
        {"metric": "recall_micro", "value": recall_score(y_true, y_pred, average="micro", zero_division=0)},
        {"metric": "f1_micro", "value": f1_score(y_true, y_pred, average="micro", zero_division=0)},
        {"metric": "precision_macro", "value": precision_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "recall_macro", "value": recall_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "f1_macro", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "precision_weighted", "value": precision_score(y_true, y_pred, average="weighted", zero_division=0)},
        {"metric": "recall_weighted", "value": recall_score(y_true, y_pred, average="weighted", zero_division=0)},
        {"metric": "f1_weighted", "value": f1_score(y_true, y_pred, average="weighted", zero_division=0)},
    ]
    return pd.DataFrame(rows)
