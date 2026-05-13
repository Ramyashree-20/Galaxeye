"""Model definitions for binary change detection."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import segmentation_models_pytorch as smp


class ChangeDetectionUNetPP(nn.Module):
    """
    Unet++ model for binary change detection.

    Configuration:
    - Encoder: resnet34
    - Encoder weights: ImageNet
    - Input channels: 4 (3 EO RGB + 1 SAR intensity)
    - Output classes: 1 (binary change mask)
    - Final activation: None (returns raw logits — pair with BCEWithLogitsLoss)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 4,
        classes: int = 1,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    def forward(self, x):
        return self.model(x)


def build_change_detection_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 4,
    classes: int = 1,
    activation: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to build a change detection Unet++ model.
    """
    return ChangeDetectionUNetPP(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
