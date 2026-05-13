"""Training pipeline for binary change detection."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChangeDetectionDataset
from losses import CombinedDiceBCELoss
from model import build_change_detection_model
from transforms import get_train_transforms, get_val_transforms


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        """Return True when training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class CorpusIoU:
    """Streaming binary-IoU accumulator over an epoch (corpus-level)."""

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) >= threshold).to(torch.int64)
        tgt = (targets > 0.5).to(torch.int64)
        self.tp += int(((preds == 1) & (tgt == 1)).sum().item())
        self.fp += int(((preds == 1) & (tgt == 0)).sum().item())
        self.fn += int(((preds == 0) & (tgt == 1)).sum().item())

    def compute(self) -> float:
        denom = self.tp + self.fp + self.fn
        return (self.tp + self.eps) / (denom + self.eps)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(config: Dict) -> torch.device:
    requested = config.get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(config: Dict, seed: int) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    data_root = Path(dataset_cfg["data_root"])

    train_dir = data_root / dataset_cfg.get("train_split", "train/train")
    val_dir = data_root / dataset_cfg.get("val_split", "val/val")

    image_size = dataset_cfg.get("image_size", 256)

    train_dataset = ChangeDetectionDataset(
        data_root=str(train_dir),
        transform=get_train_transforms(image_size=image_size),
    )
    val_dataset = ChangeDetectionDataset(
        data_root=str(val_dir),
        transform=get_val_transforms(image_size=image_size),
    )

    batch_size = config["training"].get("batch_size", 4)
    num_workers = dataset_cfg.get("num_workers", 4)
    pin_memory = bool(dataset_cfg.get("pin_memory", True))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def create_optimizer(config: Dict, model: torch.nn.Module) -> Optimizer:
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})

    name = optimizer_cfg.get("name", "adam").lower()
    lr = training_cfg.get("learning_rate", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=optimizer_cfg.get("eps", 1e-8),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=optimizer_cfg.get("eps", 1e-8),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer name: {name}")


def create_scheduler(config: Dict, optimizer: Optimizer, num_epochs: int):
    scheduler_cfg = config.get("scheduler", {})
    name = scheduler_cfg.get("name", "cosine").lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", num_epochs),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    if name == "step":
        return StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            min_lr=scheduler_cfg.get("min_lr", 1e-7),
        )

    raise ValueError(f"Unsupported scheduler name: {name}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: CombinedDiceBCELoss,
    device: torch.device,
    grad_clip_norm: Optional[float],
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    iou_meter = CorpusIoU()

    progress = tqdm(dataloader, desc="Train", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        iou_meter.update(logits.detach(), masks)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_meter.compute():.4f}")

    num_batches = max(len(dataloader), 1)
    return {
        "loss": running_loss / num_batches,
        "iou": iou_meter.compute(),
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CombinedDiceBCELoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    iou_meter = CorpusIoU()

    progress = tqdm(dataloader, desc="Val", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        iou_meter.update(logits, masks)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_meter.compute():.4f}")

    num_batches = max(len(dataloader), 1)
    return {
        "loss": running_loss / num_batches,
        "iou": iou_meter.compute(),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler,
    best_val_iou: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_iou": best_val_iou,
            "config": config,
        },
        path,
    )


def run_training(config: Dict) -> Dict[str, list[float]]:
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True)
    device = get_device(config)

    model_cfg = config.get("model", {})
    model = build_change_detection_model(
        encoder_name=model_cfg.get("encoder", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 6),
        classes=model_cfg.get("classes", 1),
        activation=None,
    ).to(device)

    loss_cfg = config.get("loss", {})
    criterion = CombinedDiceBCELoss(
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        pos_weight=loss_cfg.get("pos_weight"),
        auto_balance=loss_cfg.get("auto_balance", False),
    ).to(device)

    train_cfg = config["training"]
    num_epochs = train_cfg.get("epochs", train_cfg.get("num_epochs", 50))
    grad_clip_norm = train_cfg.get("gradient_clip_norm")
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, num_epochs=num_epochs)

    train_loader, val_loader = create_dataloaders(config, seed=seed)

    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.0),
    )

    ckpt_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    best_ckpt_path = ckpt_dir / "best_model.pt"
    last_ckpt_path = ckpt_dir / "last_model.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
        "lr": [],
    }

    best_val_iou = -float("inf")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    print(f"Model params: {n_params:,}")
    print(f"Mixed precision: {use_amp} | Grad-clip norm: {grad_clip_norm}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip_norm=grad_clip_norm, use_amp=use_amp, scaler=scaler,
        )
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["lr"].append(current_lr)

        print(
            f"train_loss={train_metrics['loss']:.4f} | train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_iou={val_metrics['iou']:.4f} | lr={current_lr:.2e}"
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["iou"])
        else:
            scheduler.step()

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, scheduler, best_val_iou, config)
            print(f"Best model updated at epoch {epoch} (val_iou={best_val_iou:.4f})")

        save_checkpoint(last_ckpt_path, epoch, model, optimizer, scheduler, best_val_iou, config)

        if es_enabled and early_stopper.step(val_metrics["iou"]):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    history_path = ckpt_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    _save_history_plot(history, ckpt_dir / "training_history.png")

    print(f"Training complete. Best val_iou={best_val_iou:.4f}")
    print(f"Saved best checkpoint to: {best_ckpt_path}")
    return history


def _save_history_plot(history: Dict[str, list[float]], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], label="train")
    axes[1].plot(epochs, history["val_iou"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU")
    axes[1].legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary change detection model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
