"""Visualization utilities for EO-SAR binary change detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Must match the stats used by ChangeDetectionDataset.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAR_MEAN = 0.5
SAR_STD = 0.25


def _eo_chw_to_hwc(eo_chw: torch.Tensor) -> np.ndarray:
    """Denormalize a 3-channel EO tensor (CHW) back to displayable HWC in [0,1]."""
    arr = eo_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0.0, 1.0)


def _sar_chw_to_hw(sar_chw: torch.Tensor) -> np.ndarray:
    """Denormalize a 1-channel SAR tensor (CHW or HW) back to HW in [0,1]."""
    arr = sar_chw.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr * SAR_STD + SAR_MEAN
    return np.clip(arr, 0.0, 1.0)


def _to_mask_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    arr = mask_tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return (arr > 0.5).astype(np.uint8)


def save_qualitative_result(
    eo_image: torch.Tensor,
    sar_image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str | Path,
    title: Optional[str] = None,
) -> None:
    """
    Save a single 4-panel qualitative visualization.

    Panels:
        1) pre-event EO image (RGB)
        2) post-event SAR image (grayscale)
        3) ground-truth mask
        4) predicted mask
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    eo = _eo_chw_to_hwc(eo_image)
    sar = _sar_chw_to_hw(sar_image)
    gt = _to_mask_numpy(gt_mask)
    pred = _to_mask_numpy(pred_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(eo)
    axes[0].set_title("Pre-event (EO)")
    axes[0].axis("off")

    axes[1].imshow(sar, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Post-event (SAR)")
    axes[1].axis("off")

    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_batch_qualitative_results(
    images_4ch: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    sample_ids: Optional[Iterable[str]] = None,
    output_dir: str | Path = "outputs/visualizations",
    prefix: str = "sample",
) -> None:
    """
    Save qualitative visualizations for a batch.

    Args:
        images_4ch: Tensor [B, 4, H, W]  (channels 0..2 = EO RGB, channel 3 = SAR)
        gt_masks:   Tensor [B, 1, H, W] or [B, H, W]
        pred_masks: Tensor [B, 1, H, W] or [B, H, W]
        sample_ids: Optional iterable of names used in file names
        output_dir: Directory to save panels
        prefix:     Fallback file prefix when sample_ids is not provided
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = images_4ch.shape[0]
    ids = list(sample_ids) if sample_ids is not None else [f"{prefix}_{i:05d}" for i in range(batch_size)]

    if len(ids) != batch_size:
        raise ValueError("Length of sample_ids must match batch size.")

    for i in range(batch_size):
        image = images_4ch[i]
        eo = image[:3]
        sar = image[3:4]

        save_path = output_dir / f"{ids[i]}.png"
        save_qualitative_result(
            eo_image=eo,
            sar_image=sar,
            gt_mask=gt_masks[i],
            pred_mask=pred_masks[i],
            save_path=save_path,
            title=ids[i],
        )
