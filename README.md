# Binary Change Detection on EO-SAR Image Pairs

> **GalaxEye AI Research Intern — Technical Assignment**
> Binary change detection on co-registered pre-event (EO) / post-event (SAR) image pairs.

---

## Project Title & Description

**Binary EO-SAR Change Detection.** Given a co-registered pre-event EO (RGB) image and a post-event SAR (single-channel intensity) image of the same scene, predict a per-pixel binary mask: `1 = change`, `0 = no-change`.

The pipeline is built around a UNet++ segmentation network with a ResNet-34 ImageNet-pretrained encoder, adapted for **4-channel input** (3 EO + 1 SAR). The 4-semantic-class annotations (Background / Intact / Damaged / Destroyed) are remapped to binary {No-Change, Change} as required by Section 2.2 of the assignment.

---

## Requirements

- **Python:** 3.10 or newer
- **PyTorch:** 2.0 or newer (CUDA build recommended; CPU also supported)
- Pinned dependencies are in [`requirements.txt`](requirements.txt)

PyTorch is intentionally not pinned in `requirements.txt`. Install the build that matches your hardware first (see the install instructions below).

---

## Environment Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd galaxeye_change_detection

# 2. Create environment (venv example)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

# 3. Install PyTorch (pick one matching your hardware)
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install the rest
pip install -r requirements.txt
```

A conda-based setup works analogously:

```bash
conda create -n galaxeye python=3.10 -y
conda activate galaxeye
# (then steps 3 and 4 as above)
```

---

## Dataset Structure

The dataset must be placed under `data/` with the **fixed train / val / test split** provided by GalaxEye:

```text
data/
  train/train/
    pre-event/      # 3-channel EO GeoTIFF (RGB)
    post-event/     # 1-channel SAR GeoTIFF (intensity)
    target/         # uint8 mask with raw values {0, 1, 2, 3}
  val/val/
    pre-event/  post-event/  target/
  test/test/
    pre-event/  post-event/  target/
```

For each sample, the same filename must exist across `pre-event/`, `post-event/`, and `target/`.

### Modalities

| Source | Channels | dtype | Notes |
|---|---|---|---|
| `pre-event/`  | 3 (RGB) | uint8 | Electro-optical (EO) imagery |
| `post-event/` | 1       | uint8 | Synthetic-aperture radar (SAR) intensity |
| `target/`     | 1       | uint8 | 4 semantic classes (see remapping below) |

### Mandatory label remapping (Section 2.2)

This is applied inside the dataset loader — you do **not** need to remap the files on disk.

| Original value | Original class | Remapped value | Remapped class |
|---:|---|---:|---|
| 0 | Background | 0 | No-Change |
| 1 | Intact     | 0 | No-Change |
| 2 | Damaged    | 1 | Change    |
| 3 | Destroyed  | 1 | Change    |

### Split statistics (as shipped)

| Split | Samples | Note |
|-------|--------:|------|
| Train | 2,781   | Full train set |
| Val   | 334     | |
| Test  | 77      | Per Section 2.3, this is the **visible 50%**; the remaining 50% is GalaxEye's blind holdout |

---

## Training

```bash
python train.py --config config.yaml
```

All hyperparameters live in [`config.yaml`](config.yaml): random seed, image size, batch size, epochs, optimizer, learning rate, scheduler, loss weights, augmentations, early stopping, and checkpointing. The full resolved config used for any run is dumped to `outputs/checkpoints/config_used.yaml` for reproducibility.

Per-epoch training artifacts (under `outputs/checkpoints/`):

| File | Description |
|---|---|
| `best_model.pt`         | Best by validation IoU (recommended weights to submit) |
| `last_model.pt`         | Most recent epoch |
| `history.json`          | Per-epoch train/val loss + IoU + LR |
| `training_history.png`  | Loss and IoU curves |
| `config_used.yaml`      | Exact config used |

---

## Evaluation

The CLI follows the spec in Section 5.1.1:

```bash
python eval.py \
  --data_path /path/to/test/test \
  --weights   /path/to/checkpoint.pth \
  --config    config.yaml \
  --batch-size 4 \
  --threshold 0.5 \
  --out-dir   outputs/eval/test \
  --save-vis
```

`--data_path` should point at a split directory containing `pre-event/`, `post-event/`, and `target/`. `--weights` accepts both raw `state_dict` files and the wrapped checkpoints produced by `train.py`.

To get the validation-split metrics that Section 4 asks for, run the same command against the val directory:

```bash
python eval.py --data_path data/val/val --weights outputs/checkpoints/best_model.pt \
               --config config.yaml --out-dir outputs/eval/val --save-vis
```

Output files (per run, under `--out-dir`):

- `summary.csv`            — global IoU / Precision / Recall / F1 / TN / FP / FN / TP
- `per_sample_metrics.csv` — per-sample metrics
- `visualizations/*.png`   — 4-panel qualitative panels (EO / SAR / GT / Pred) when `--save-vis` is set

---

## Model Weights

> Public download link: **`<TODO: paste your Google Drive / HuggingFace Hub link here once trained>`**

Once you have a final checkpoint, upload `outputs/checkpoints/best_model.pt` to Google Drive (anyone-with-the-link / public), Hugging Face Hub, or equivalent, and replace the placeholder above. GalaxEye explicitly requires a public link in the README — do not attach the file by email.

---

## Results

> Re-run `python eval.py ...` on both `data/val/val` and `data/test/test` after training, then fill in this table.

| Split | IoU | Precision | Recall | F1 |
|-------|----:|----------:|-------:|---:|
| Val   | `TBD` | `TBD` | `TBD` | `TBD` |
| Test  | `TBD` | `TBD` | `TBD` | `TBD` |

Confusion matrix layout used throughout this repo:

```text
[[TN, FP],
 [FN, TP]]
```

A discussion of the error profile (where the model fails and why) lives in the technical report under `report/`.

---

## Architecture and Design Decisions (Summary)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | UNet++ (`segmentation_models_pytorch`) | Strong dense-prediction backbone with skip connections; well-supported with custom `in_channels`. |
| Encoder | `resnet34`, ImageNet-pretrained | SMP duplicates the conv1 filters for `in_channels=4`, so EO benefits from pretraining; SAR begins from a sensible initialization. |
| Input | 4 channels: `[EO_R, EO_G, EO_B, SAR]` | The provided data has 3-channel EO pre-event and 1-channel SAR post-event. Stacking both modalities into a single tensor lets the encoder learn EO-SAR fusion end-to-end. |
| Output | 1 channel of raw logits (no in-model activation) | Pairs correctly with `BCEWithLogitsLoss`; sigmoid is applied only at loss and eval time. |
| Loss | Combined Dice + BCEWithLogitsLoss | Dice handles overlap quality; BCE provides a stable per-pixel signal. Optional per-batch `auto_balance` to handle class imbalance. |
| Class imbalance | Dice term + optional `pos_weight` / `auto_balance` | Disaster change pixels are rare. Both fixed and dynamic weighting are configurable in `config.yaml`. |
| EO normalization | ImageNet mean/std | Matches the pretrained encoder. |
| SAR normalization | `(x − 0.5) / 0.25` (uint8 scaled to [0,1] then standardized) | Avoids inappropriate ImageNet stats on single-channel SAR; can be replaced with dataset-derived stats. |
| Augmentation | HFlip / VFlip / RandomRotate90 | Geometry-only augs applied jointly to EO + SAR + mask so the triplet stays co-registered. |
| Optimizer | Adam, lr 1e-3, wd 1e-5 | Standard for segmentation; AdamW / SGD also selectable in `config.yaml`. |
| Scheduler | CosineAnnealingLR | Smooth decay across epochs; alternatives (`step`, `plateau`) are configurable. |
| Reproducibility | Seeded RNGs, seeded DataLoader workers, `cudnn.deterministic = True` | Run-to-run repeatability on the same hardware. |

---

## Project Layout

```text
.
├── config.yaml             # all hyperparameters
├── dataset.py              # EO + SAR + mask Dataset, label remap, per-modality normalization
├── transforms.py           # Albumentations geometric augmentations (joint across modalities)
├── model.py                # UNet++ wrapper (smp), in_channels=4
├── losses.py               # Dice / BCE / Dice+BCE
├── metrics.py              # IoU / P / R / F1 / confusion matrix
├── visualization.py        # 4-panel qualitative panels (EO / SAR / GT / Pred)
├── train.py                # training loop
├── eval.py                 # evaluation loop (--data_path, --weights)
├── requirements.txt
├── README.md
├── notebooks/
│   └── data_exploration.ipynb
├── report/                 # technical report PDF goes here
├── data/                   # not tracked in git
└── outputs/                # not tracked in git
```

---

## Citation / References

1. Zhou, Z. et al. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA.
2. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR.
4. Iakubovskii, P. *segmentation_models.pytorch.* https://github.com/qubvel/segmentation_models.pytorch
5. Buslaev, A. et al. *Albumentations: Fast and Flexible Image Augmentations.* Information, 2020.
6. Daudt, R. C. et al. (2018). *Fully Convolutional Siamese Networks for Change Detection.* ICIP.
7. Chen, H. & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing.

---

