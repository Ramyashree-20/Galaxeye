"""Evaluation pipeline for binary EO-SAR change detection.

CLI follows the GalaxEye spec (Section 5.1.1):

    python eval.py --data_path /path/to/test --weights /path/to/checkpoint.pth

``--data_path`` should point at a directory containing ``pre-event/``,
``post-event/`` and ``target/`` subfolders. ``--weights`` may be either a raw
state_dict or a checkpoint dict produced by ``train.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from metrics import confusion_matrix as batch_confusion_matrix
from model import build_change_detection_model
from transforms import get_val_transforms

# Must match the stats used in ChangeDetectionDataset (per modality).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAR_MEAN = 0.5
SAR_STD = 0.25


def _safe_div(num: float, den: float, eps: float = 1e-7) -> float:
    return float(num / (den + eps))


def _compute_scores_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def _eo_to_rgb_uint8(eo_chw: torch.Tensor) -> np.ndarray:
    arr = eo_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _sar_to_rgb_uint8(sar_chw: torch.Tensor) -> np.ndarray:
    arr = sar_chw.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr * SAR_STD + SAR_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _mask_to_rgb_uint8(mask_hw: np.ndarray) -> np.ndarray:
    gray = (mask_hw > 0).astype(np.uint8) * 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _save_visualization(
    out_path: Path,
    eo_rgb: np.ndarray,
    sar_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
) -> None:
    panel = np.concatenate([eo_rgb, sar_rgb, gt_rgb, pred_rgb], axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _load_model_state_dict(weights_path: str, device: torch.device) -> Tuple[Dict, Dict]:
    """Load weights from either a raw state_dict or a wrapped checkpoint.

    Returns (state_dict, config_or_empty_dict).
    """
    obj = torch.load(weights_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj.get("config", {})
    # Raw state_dict.
    return obj, {}


class SegmentationEvaluator:
    def __init__(self, weights_path: str, config: Dict, device: torch.device) -> None:
        state_dict, ckpt_config = _load_model_state_dict(weights_path, device)
        # Prefer the model config baked into the checkpoint when available
        # (this lets us evaluate older checkpoints without a separate config).
        model_cfg = (ckpt_config.get("model") if ckpt_config else None) or config.get("model", {})

        self.device = device
        self.model = build_change_detection_model(
            encoder_name=model_cfg.get("encoder", model_cfg.get("encoder_name", "resnet34")),
            encoder_weights=None,
            in_channels=model_cfg.get("in_channels", 4),
            classes=model_cfg.get("classes", 1),
            activation=None,
        ).to(device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float,
        save_vis_dir: Path | None,
    ) -> Tuple[Dict[str, float], np.ndarray, List[Dict[str, float]]]:
        global_cm = np.zeros((2, 2), dtype=np.int64)
        rows: List[Dict[str, float]] = []

        if save_vis_dir is not None:
            save_vis_dir.mkdir(parents=True, exist_ok=True)

        sample_index = 0
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            batch_cm = batch_confusion_matrix(preds, masks, threshold=0.5, from_logits=False)
            global_cm += batch_cm

            for i in range(images.shape[0]):
                sample_pred = preds[i : i + 1]
                sample_mask = masks[i : i + 1]
                sample_cm = batch_confusion_matrix(sample_pred, sample_mask, threshold=0.5, from_logits=False)
                sample_scores = _compute_scores_from_cm(sample_cm)
                tn, fp, fn, tp = sample_cm.ravel()
                rows.append({
                    "sample_index": sample_index,
                    "iou": sample_scores["iou"],
                    "precision": sample_scores["precision"],
                    "recall": sample_scores["recall"],
                    "f1": sample_scores["f1"],
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                })

                if save_vis_dir is not None:
                    image_4ch = images[i].detach().cpu()
                    eo_rgb = _eo_to_rgb_uint8(image_4ch[:3])
                    sar_rgb = _sar_to_rgb_uint8(image_4ch[3:4])
                    gt_rgb = _mask_to_rgb_uint8(sample_mask[0, 0].detach().cpu().numpy())
                    pred_rgb = _mask_to_rgb_uint8(sample_pred[0, 0].detach().cpu().numpy())
                    vis_path = save_vis_dir / f"sample_{sample_index:05d}.png"
                    _save_visualization(vis_path, eo_rgb, sar_rgb, gt_rgb, pred_rgb)

                sample_index += 1

        summary = _compute_scores_from_cm(global_cm)
        return summary, global_cm, rows


def build_loader(data_path: str, config: Dict, batch_size: int) -> DataLoader:
    dataset_cfg = config.get("dataset", {})
    image_size = dataset_cfg.get("image_size", 256)
    dataset = ChangeDetectionDataset(
        data_root=data_path,
        transform=get_val_transforms(image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=bool(dataset_cfg.get("pin_memory", False)),
    )


def save_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(out_dir: Path, summary: Dict[str, float], cm: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["iou", summary["iou"]])
        writer.writerow(["precision", summary["precision"]])
        writer.writerow(["recall", summary["recall"]])
        writer.writerow(["f1", summary["f1"]])
        writer.writerow(["tn", int(cm[0, 0])])
        writer.writerow(["fp", int(cm[0, 1])])
        writer.writerow(["fn", int(cm[1, 0])])
        writer.writerow(["tp", int(cm[1, 1])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary EO-SAR change detection model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Absolute or relative path to a split directory containing "
                             "pre-event/, post-event/, target/ subfolders")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.pt or .pth, raw state_dict or wrapped checkpoint)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (used for image_size / model defaults if checkpoint is bare)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="outputs/eval")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save 4-panel qualitative visualizations under <out-dir>/visualizations/")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data path:    {args.data_path}")
    print(f"Weights:      {args.weights}")

    loader = build_loader(data_path=args.data_path, config=config, batch_size=args.batch_size)
    evaluator = SegmentationEvaluator(weights_path=args.weights, config=config, device=device)

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations" if args.save_vis else None

    summary, cm, rows = evaluator.evaluate(
        dataloader=loader,
        threshold=args.threshold,
        save_vis_dir=vis_dir,
    )

    save_csv(rows, out_dir / "per_sample_metrics.csv")
    save_summary(out_dir, summary, cm)

    print()
    print("Evaluation complete")
    print(f"  IoU       : {summary['iou']:.4f}")
    print(f"  Precision : {summary['precision']:.4f}")
    print(f"  Recall    : {summary['recall']:.4f}")
    print(f"  F1        : {summary['f1']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
