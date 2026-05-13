"""Dataset utilities for binary EO-SAR change detection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Mandatory label remap (GalaxEye assignment Section 2.2):
#   0 (Background)  -> 0 (No-Change)
#   1 (Intact)      -> 0 (No-Change)
#   2 (Damaged)     -> 1 (Change)
#   3 (Destroyed)   -> 1 (Change)
CHANGE_LABEL_VALUES = (2, 3)

# EO is ImageNet-pretrained-backbone friendly; SAR gets its own normalization.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAR_MEAN = 0.5
SAR_STD = 0.25


class ChangeDetectionDataset(Dataset):
    """
    PyTorch Dataset for binary EO-SAR change detection.

    Expected directory layout:
        data_root/
            pre-event/      # EO, 3-channel uint8 GeoTIFF (RGB)
            post-event/     # SAR, 1-channel uint8 GeoTIFF (intensity)
            target/         # 4-class uint8 mask {0,1,2,3}, remapped to binary

    Samples are formed from files that share the same filename in all 3 folders.

    Output image tensor shape: [4, H, W]  (3 EO channels + 1 SAR channel)
    Output mask tensor shape:  [1, H, W]  (binary, change class only)
    """

    def __init__(self, data_root: str, transform: Optional[Callable] = None) -> None:
        self.data_root = Path(data_root)
        self.pre_dir = self.data_root / "pre-event"
        self.post_dir = self.data_root / "post-event"
        self.mask_dir = self.data_root / "target"
        self.transform = transform

        self._validate_dirs()
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(f"No valid samples found under: {self.data_root}")

    def _validate_dirs(self) -> None:
        for folder in (self.pre_dir, self.post_dir, self.mask_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing required directory: {folder}")

    def _build_samples(self) -> list[Tuple[Path, Path, Path]]:
        samples: list[Tuple[Path, Path, Path]] = []
        for pre_path in sorted(self.pre_dir.iterdir()):
            if pre_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            post_path = self.post_dir / pre_path.name
            mask_path = self.mask_dir / pre_path.name
            if not post_path.exists():
                raise FileNotFoundError(f"Missing post-event file: {post_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing target mask file: {mask_path}")
            samples.append((pre_path, post_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_eo_image(path: Path) -> np.ndarray:
        """Read pre-event EO image as HxWx3 float32 in [0, 1] (RGB order)."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read EO image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _read_sar_image(path: Path) -> np.ndarray:
        """Read post-event SAR image as HxW float32 in [0, 1] (single channel)."""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read SAR image: {path}")
        if image.ndim == 3:
            # Defensive: collapse to single-channel by taking the first channel.
            image = image[..., 0]
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _read_remap_mask(path: Path) -> np.ndarray:
        """Read mask and apply mandatory 4-class -> binary remap (Section 2.2)."""
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = np.isin(mask, CHANGE_LABEL_VALUES)
        return binary.astype(np.float32)

    @staticmethod
    def _normalize_eo(eo: np.ndarray) -> np.ndarray:
        """Apply ImageNet mean/std to 3-channel EO (HxWx3, already in [0,1])."""
        return (eo - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def _normalize_sar(sar: np.ndarray) -> np.ndarray:
        """Standardize single-channel SAR (HxW, already in [0,1])."""
        return (sar - SAR_MEAN) / SAR_STD

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_path, post_path, mask_path = self.samples[index]

        eo = self._read_eo_image(pre_path)        # [H, W, 3], float32 [0,1]
        sar = self._read_sar_image(post_path)     # [H, W],    float32 [0,1]
        mask = self._read_remap_mask(mask_path)   # [H, W],    float32 {0,1}

        # Apply joint geometric/resize transforms across EO + SAR + mask.
        # Normalization is applied below (not inside the Compose) so EO and SAR
        # can use different statistics.
        if self.transform is not None:
            augmented = self.transform(image=eo, sar=sar, mask=mask)
            eo = augmented["image"]
            sar = augmented["sar"]
            mask = augmented["mask"]

        # Per-modality normalization.
        eo = self._normalize_eo(eo)
        sar = self._normalize_sar(sar)

        # To tensors (CHW for the image, 1HW for the mask).
        eo_t = torch.from_numpy(np.ascontiguousarray(eo.transpose(2, 0, 1))).float()
        sar_t = torch.from_numpy(np.ascontiguousarray(sar[None, ...])).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()

        image = torch.cat([eo_t, sar_t], dim=0)   # [4, H, W]

        if image.shape[0] != 4:
            raise RuntimeError(f"Expected image with 4 channels, got {tuple(image.shape)}")
        if mask_t.shape[0] != 1:
            raise RuntimeError(f"Expected mask with 1 channel, got {tuple(mask_t.shape)}")

        return image, mask_t
