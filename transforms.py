"""Albumentations transforms for EO-SAR change detection.

Geometric augmentations (resize + flips + 90-degree rotations) are applied
*jointly* to the EO image, the SAR image, and the mask so they stay
co-registered. Per-modality intensity normalization is intentionally NOT done
here — it lives in ``ChangeDetectionDataset`` so EO and SAR can use different
statistics (ImageNet mean/std for EO, dataset-derived for SAR).
"""

from __future__ import annotations

import albumentations as A


# ``additional_targets`` tells Albumentations to apply the same spatial
# transforms to extra inputs. SAR is declared as an "image" so it goes through
# the same Resize / Flip / Rotate as the EO image.
_ADDITIONAL_TARGETS = {"sar": "image"}


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """Training geometric transforms applied jointly to EO, SAR, and mask."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets=_ADDITIONAL_TARGETS,
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """Deterministic validation/test transform (resize only)."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
        ],
        additional_targets=_ADDITIONAL_TARGETS,
    )
