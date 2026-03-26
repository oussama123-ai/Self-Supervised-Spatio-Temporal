"""
Evaluation metrics for SSS-TT (Section 4.6 of paper):
  - Accuracy
  - Quadratic Weighted Kappa (QWK) — primary metric
  - Mean Absolute Error (MAE)
  - AUC (per-class OvR)
  - F1-score (macro)
  - Per-class precision / recall / F1
  - Clinical alert metrics (sensitivity, specificity, PPV, NPV)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, mean_absolute_error,
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray,
                num_classes: int = 4) -> float:
    """Quadratic Weighted Kappa (sklearn wrapper with weights='quadratic')."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(num_classes)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 4,
    pain_threshold: int = 2,
) -> dict:
    """
    Compute all metrics reported in the paper.

    Args:
        y_true:         (N,) ground truth labels
        y_pred:         (N,) predicted labels
        y_prob:         (N, C) class probabilities (for AUC)
        num_classes:    number of classes (4)
        pain_threshold: threshold for binary clinical alert metrics (2)
    Returns:
        dict of metric names → values
    """
    metrics = {}

    # Primary metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['qwk'] = compute_qwk(y_true, y_pred, num_classes)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)

    # AUC (OvR, requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr',
                labels=list(range(num_classes))
            )
        except ValueError:
            metrics['auc'] = float('nan')

    # Per-class metrics
    per_class = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=[f'Level_{k}' for k in range(num_classes)],
        output_dict=True, zero_division=0
    )
    metrics['per_class'] = per_class

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )

    # Clinical alert metrics (binary: pain ≥ threshold vs < threshold)
    y_binary_true = (y_true >= pain_threshold).astype(int)
    y_binary_pred = (y_pred >= pain_threshold).astype(int)

    tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
    tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
    fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
    fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))

    metrics['sensitivity'] = tp / (tp + fn + 1e-8)
    metrics['specificity'] = tn / (tn + fp + 1e-8)
    metrics['ppv'] = tp / (tp + fp + 1e-8)
    metrics['npv'] = tn / (tn + fn + 1e-8)

    if y_prob is not None:
        try:
            y_binary_prob = y_prob[:, pain_threshold:].sum(axis=1)
            metrics['auc_binary'] = roc_auc_score(y_binary_true, y_binary_prob)
        except ValueError:
            metrics['auc_binary'] = float('nan')

    return metrics


def print_metrics(metrics: dict, prefix: str = '') -> None:
    """Pretty-print metrics dict."""
    print(f"\n{'='*50}")
    print(f"{prefix} Evaluation Results")
    print('='*50)
    print(f"Accuracy:    {metrics.get('accuracy', 0):.2f}%")
    print(f"QWK:         {metrics.get('qwk', 0):.4f}")
    print(f"MAE:         {metrics.get('mae', 0):.4f}")
    print(f"F1 (macro):  {metrics.get('f1_macro', 0):.4f}")
    if 'auc' in metrics:
        print(f"AUC (OvR):   {metrics.get('auc', 0):.4f}")
    print(f"\nClinical Alert (pain≥2):")
    print(f"  Sensitivity: {metrics.get('sensitivity', 0)*100:.1f}%")
    print(f"  Specificity: {metrics.get('specificity', 0)*100:.1f}%")
    print(f"  PPV:         {metrics.get('ppv', 0)*100:.1f}%")
    print(f"  NPV:         {metrics.get('npv', 0)*100:.1f}%")
    if 'auc_binary' in metrics:
        print(f"  AUC:         {metrics.get('auc_binary', 0):.4f}")
    print('='*50)


class MetricTracker:
    """Accumulates predictions across batches for epoch-level evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._y_true = []
        self._y_pred = []
        self._y_prob = []
        self._losses = []

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor,
               y_prob: torch.Tensor | None = None,
               loss: float = 0.0):
        self._y_true.extend(y_true.cpu().numpy().tolist())
        self._y_pred.extend(y_pred.cpu().numpy().tolist())
        if y_prob is not None:
            self._y_prob.extend(y_prob.cpu().numpy().tolist())
        self._losses.append(loss)

    def compute(self) -> dict:
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        y_prob = np.array(self._y_prob) if self._y_prob else None
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = np.mean(self._losses)
        return metrics
