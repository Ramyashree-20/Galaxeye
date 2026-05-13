"""Evaluation metrics for binary segmentation/change detection."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def _to_binary_predictions(
    preds: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """Convert model outputs to binary predictions (0/1)."""
    if from_logits:
        preds = torch.sigmoid(preds)
    return (preds >= threshold).to(torch.int64)


def _to_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    """Convert targets to binary integer mask (0/1)."""
    return (targets > 0.5).to(torch.int64)


def _flatten_binary_tensors(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert predictions/targets to flattened binary tensors."""
    pred_bin = _to_binary_predictions(preds, threshold=threshold, from_logits=from_logits)
    target_bin = _to_binary_targets(targets)

    return pred_bin.reshape(-1), target_bin.reshape(-1)


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix for binary segmentation.

    Returns ndarray with shape [2, 2] in sklearn order:
    [[TN, FP],
     [FN, TP]]
    """
    pred_flat, target_flat = _flatten_binary_tensors(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )

    y_pred = pred_flat.detach().cpu().numpy()
    y_true = target_flat.detach().cpu().numpy()

    return sk_confusion_matrix(y_true, y_pred, labels=[0, 1])


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-7,
) -> float:
    """Compute Intersection-over-Union (IoU / Jaccard) for binary masks."""
    pred_flat, target_flat = _flatten_binary_tensors(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )

    pred_flat = pred_flat.float()
    target_flat = target_flat.float()

    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat) - intersection

    return float((intersection + eps) / (union + eps))


def precision_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-7,
) -> float:
    """Compute precision = TP / (TP + FP)."""
    pred_flat, target_flat = _flatten_binary_tensors(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )

    tp = torch.sum((pred_flat == 1) & (target_flat == 1)).float()
    fp = torch.sum((pred_flat == 1) & (target_flat == 0)).float()

    return float((tp + eps) / (tp + fp + eps))


def recall_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-7,
) -> float:
    """Compute recall = TP / (TP + FN)."""
    pred_flat, target_flat = _flatten_binary_tensors(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )

    tp = torch.sum((pred_flat == 1) & (target_flat == 1)).float()
    fn = torch.sum((pred_flat == 0) & (target_flat == 1)).float()

    return float((tp + eps) / (tp + fn + eps))


def f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-7,
) -> float:
    """Compute F1 score = 2 * (precision * recall) / (precision + recall)."""
    precision = precision_score(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
        eps=eps,
    )
    recall = recall_score(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
        eps=eps,
    )

    return float((2.0 * precision * recall + eps) / (precision + recall + eps))


def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Dict[str, object]:
    """
    Compute all binary segmentation metrics in one call.

    Returns keys:
    - iou
    - precision
    - recall
    - f1
    - confusion_matrix
    """
    cm = confusion_matrix(
        preds=preds,
        targets=targets,
        threshold=threshold,
        from_logits=from_logits,
    )

    return {
        "iou": iou_score(preds, targets, threshold=threshold, from_logits=from_logits),
        "precision": precision_score(preds, targets, threshold=threshold, from_logits=from_logits),
        "recall": recall_score(preds, targets, threshold=threshold, from_logits=from_logits),
        "f1": f1_score(preds, targets, threshold=threshold, from_logits=from_logits),
        "confusion_matrix": cm,
    }
