"""Custom loss functions for binary segmentation/change detection."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Args:
        smooth: Smoothing term to avoid division by zero.
        from_logits: If True, applies sigmoid to inputs before computing Dice.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        targets = targets.float()

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class BCEWithLogitsBinaryLoss(nn.Module):
    """
    BCEWithLogits loss with optional class-imbalance handling.

    Args:
        pos_weight: Manual positive-class weight (scalar). If provided,
            positives are weighted more when changes are sparse.
        reduction: Reduction mode for BCE loss.
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_pos_weight",
            torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None,
            persistent=False,
        )
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        pos_weight = None
        if self._pos_weight is not None:
            pos_weight = self._pos_weight.to(device=inputs.device, dtype=inputs.dtype)

        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


class CombinedDiceBCELoss(nn.Module):
    """
    Weighted combination of Dice loss and BCEWithLogits loss.

    Supports class imbalance via:
    - fixed `pos_weight`
    - optional dynamic per-batch `auto_balance`

    Args:
        dice_weight: Weight for Dice term.
        bce_weight: Weight for BCE term.
        smooth: Dice smoothing term.
        pos_weight: Fixed positive-class weight for BCE.
        auto_balance: If True, computes pos_weight = neg/pos from current batch.
        max_auto_pos_weight: Caps dynamic pos_weight to avoid instability.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
        auto_balance: bool = False,
        max_auto_pos_weight: float = 100.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.auto_balance = auto_balance
        self.max_auto_pos_weight = max_auto_pos_weight

        self.dice = DiceLoss(smooth=smooth, from_logits=True)
        self.bce = BCEWithLogitsBinaryLoss(pos_weight=pos_weight)

    @staticmethod
    def _compute_auto_pos_weight(targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        positive = targets.sum()
        total = torch.tensor(targets.numel(), device=targets.device, dtype=targets.dtype)
        negative = total - positive
        pos_weight = negative / (positive + eps)
        return pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        if self.auto_balance:
            auto_pw = self._compute_auto_pos_weight(targets).clamp(max=self.max_auto_pos_weight)
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs,
                targets,
                pos_weight=auto_pw,
                reduction="mean",
            )
        else:
            bce_loss = self.bce(inputs, targets)

        dice_loss = self.dice(inputs, targets)

        total_weight = self.dice_weight + self.bce_weight
        if total_weight <= 0:
            raise ValueError("dice_weight + bce_weight must be > 0")

        return (self.dice_weight * dice_loss + self.bce_weight * bce_loss) / total_weight


def build_loss(
    name: str = "dice_bce",
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    smooth: float = 1.0,
    pos_weight: Optional[float] = None,
    auto_balance: bool = False,
) -> nn.Module:
    """
    Loss factory for binary segmentation.

    Supported names:
        - "dice"
        - "bce"
        - "dice_bce"
    """
    name = name.lower()

    if name == "dice":
        return DiceLoss(smooth=smooth, from_logits=True)
    if name == "bce":
        return BCEWithLogitsBinaryLoss(pos_weight=pos_weight)
    if name in {"dice_bce", "combined", "combo"}:
        return CombinedDiceBCELoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            smooth=smooth,
            pos_weight=pos_weight,
            auto_balance=auto_balance,
        )

    raise ValueError(f"Unsupported loss name: {name}")
