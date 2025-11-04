"""
UNet Semantic Segmentation Training Pipeline
================================================

This script is structured into small, self-contained steps so it can be
copied directly into Google Colab cells. The pipeline assumes the dataset
will be fetched from Roboflow using their provided download snippet and will
run on a Colab runtime backed by a T4 GPU.

How to use in Colab:
1. Create a new Colab notebook, set the runtime to GPU (T4).
2. Copy each STEP section below into its own notebook cell (in order).
3. Execute the cells sequentially, making sure to update the Roboflow
   snippet with your workspace/project/version information.
"""

# =============================================================================
# STEP 0: Optional GPU & environment sanity check
# =============================================================================
import os
import sys
import subprocess


def _in_colab() -> bool:
    """Detect whether the current runtime is Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


if _in_colab():
    print("✅ Detected Google Colab runtime")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA GPU not available. Check Runtime → Change runtime type → GPU.")
    except ImportError:
        print("⚠️ PyTorch not yet installed; it will be installed in STEP 1.")
else:
    print("ℹ️ Non-Colab environment detected. The script will still run if dependencies are present.")


# =============================================================================
# STEP 1: Install dependencies (run only once per runtime)
# =============================================================================
REQUIRED_PACKAGES = [
    "roboflow",
    "albumentations==1.3.1",
    "opencv-python",
    "torchmetrics",
]


def install_dependencies(packages):
    """Install required Python packages quietly."""
    print("📦 Installing dependencies ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
    print("✅ Dependencies installed")


if _in_colab():
    install_dependencies(REQUIRED_PACKAGES)
else:
    print("ℹ️ Skipping automatic pip install outside Colab. Ensure dependencies are available.")


# =============================================================================
# STEP 2: Imports & global configuration
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from albumentations import Compose, HorizontalFlip, Normalize, RandomBrightnessContrast, Resize
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover - handled by STEP 1
    raise RuntimeError(
        "Albumentations is missing. Run STEP 1 to install dependencies first."
    ) from exc


@dataclass
class Config:
    """Configuration container for training hyperparameters and paths."""

    num_classes: int = 1  # Update this after inspecting the Roboflow dataset
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_amp: bool = True
    grad_accum_steps: int = 1
    class_weights: Sequence[float] | None = None
    work_dir: Path = Path("unet_runs")
    checkpoint_name: str = "unet_best.pt"
    seed: int = 42


cfg = Config()


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic-ish behaviour for reproducibility."""

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


cfg.work_dir.mkdir(parents=True, exist_ok=True)
set_seed(cfg.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🏁 Using device: {DEVICE}")


# =============================================================================
# STEP 3: Download the dataset from Roboflow (replace placeholders!)
# =============================================================================
# NOTE: Replace <YOUR_API_KEY>, <WORKSPACE>, <PROJECT>, and VERSION_NUMBER with
# values from your Roboflow dataset page. The download target should be a
# semantic segmentation format such as "png-mask-semantic" or "coco-segmentation".

DATASET_ROOT: Path | None = None

try:
    from roboflow import Roboflow

    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "<YOUR_API_KEY>")
    if ROBOFLOW_API_KEY and ROBOFLOW_API_KEY != "<YOUR_API_KEY>":
        print("🔑 Using Roboflow API key from environment variable ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "<WORKSPACE>")
    ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "<PROJECT>")
    ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))  # Update to match your dataset version
    ROBOFLOW_FORMAT = os.getenv("ROBOFLOW_FORMAT", "png-mask-semantic")

    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download(ROBOFLOW_FORMAT)
    DATASET_ROOT = Path(dataset.location)
    print(f"📂 Dataset downloaded to: {DATASET_ROOT}")
except Exception as download_exc:
    print(
        "⚠️ Roboflow dataset download skipped. Update STEP 3 with valid credentials or manually set DATASET_ROOT."
    )
    print(f"↪︎ Details: {download_exc}")


# If you already have the dataset on disk (e.g., mounted Google Drive), you can set:
# DATASET_ROOT = Path("/content/drive/MyDrive/path_to_dataset_folder")

if DATASET_ROOT is None:
    # Placeholder to avoid attribute errors later. Update this path before training.
    DATASET_ROOT = Path("/content/roboflow_dataset")
    print(f"ℹ️ Placeholder DATASET_ROOT set to: {DATASET_ROOT}. Update this before training.")


# =============================================================================
# STEP 4: Data utilities – pair images with masks & build datasets
# =============================================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _find_split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    """Infer the images and masks directories for a dataset split."""

    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    image_dir = split_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images directory at: {image_dir}")

    for candidate in ("masks", "labels", "annotations"):
        mask_dir = split_dir / candidate
        if mask_dir.exists():
            return image_dir, mask_dir

    raise FileNotFoundError(
        f"Unable to locate masks directory inside {split_dir}. Expected one of 'masks', 'labels', or 'annotations'."
    )


def _list_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    """Match image/mask files by stem."""

    masks_by_stem: Dict[str, Path] = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs: List[Tuple[Path, Path]] = []

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for image: {image_path.name}")
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    return pairs


def make_transforms(cfg: Config) -> Tuple[Callable, Callable]:
    """Build training and validation transforms using Albumentations."""

    train_transform = Compose(
        [
            Resize(cfg.image_size[0], cfg.image_size[1]),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
            ToTensorV2(),
        ]
    )

    valid_transform = Compose(
        [
            Resize(cfg.image_size[0], cfg.image_size[1]),
            Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
            ToTensorV2(),
        ]
    )

    return train_transform, valid_transform


class RoboflowSegmentationDataset(Dataset):
    """Simple paired image/mask dataset for Roboflow exports."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        transform: Callable,
        num_classes: int,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:  # noqa: D401
        return len(self.pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {path}")
        if mask.ndim == 3:
            # Convert RGB mask to class indices via argmax
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = np.argmax(mask, axis=-1)
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"]

        if self.num_classes == 1:
            mask_tensor = (mask_tensor > 0).float().unsqueeze(0)
        else:
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.squeeze(0)
            mask_tensor = mask_tensor.long()

        return {"image": image_tensor, "mask": mask_tensor}


def build_dataloaders(dataset_root: Path, cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/valid/test dataloaders from the dataset root."""

    train_img_dir, train_mask_dir = _find_split_dirs(dataset_root, "train")
    valid_img_dir, valid_mask_dir = _find_split_dirs(dataset_root, "valid")

    train_pairs = _list_pairs(train_img_dir, train_mask_dir)
    valid_pairs = _list_pairs(valid_img_dir, valid_mask_dir)

    test_pairs: List[Tuple[Path, Path]] = []
    try:
        test_img_dir, test_mask_dir = _find_split_dirs(dataset_root, "test")
        test_pairs = _list_pairs(test_img_dir, test_mask_dir)
    except FileNotFoundError:
        print("⚠️ Test split not found. Proceeding without test dataloader.")

    train_tfms, valid_tfms = make_transforms(cfg)

    train_ds = RoboflowSegmentationDataset(train_pairs, transform=train_tfms, num_classes=cfg.num_classes)
    valid_ds = RoboflowSegmentationDataset(valid_pairs, transform=valid_tfms, num_classes=cfg.num_classes)
    test_ds = RoboflowSegmentationDataset(test_pairs, transform=valid_tfms, num_classes=cfg.num_classes) if test_pairs else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True) if test_ds else None

    return train_loader, valid_loader, test_loader


# =============================================================================
# STEP 5: Model definition – UNet building blocks
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# =============================================================================
# STEP 6: Losses, metrics, and helpers
# =============================================================================

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss supporting binary and multi-class segmentation."""

    if num_classes == 1:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    else:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.num_classes == 1:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    else:
        class_weights = None
        if cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights, device=logits.device, dtype=torch.float32)
        bce = F.cross_entropy(logits, targets.long(), weight=class_weights)

    return bce + dice_loss(logits, targets, cfg.num_classes)


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, threshold: float = 0.5) -> float:
    """Compute mean IoU for binary or multi-class predictions."""

    if num_classes == 1:
        preds = (torch.sigmoid(logits) > threshold).float()
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

    preds = torch.argmax(logits, dim=1)
    targets = targets.long()
    ious: List[torch.Tensor] = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum(dim=(1, 2)).float()
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
        valid = union > 0
        if valid.any():
            class_iou = ((intersection + 1e-6) / (union + 1e-6))[valid]
            ious.append(class_iou)

    if not ious:
        return float("nan")

    return torch.cat(ious).mean().item()


# =============================================================================
# STEP 7: Training and evaluation loops
# =============================================================================

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Config,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    for step, batch in enumerate(loader):
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(images)
            loss = combined_loss(logits, masks, cfg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_iou = compute_iou(logits, masks, cfg.num_classes)

        loss_meter.update(loss.item(), images.size(0))
        if not np.isnan(batch_iou):
            iou_meter.update(batch_iou, images.size(0))

        if (step + 1) % 10 == 0:
            print(f"  ➤ Step {step + 1}/{len(loader)} | loss: {loss_meter.avg:.4f} | mIoU: {iou_meter.avg:.4f}")

    return loss_meter.avg, iou_meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: Config) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = combined_loss(logits, masks, cfg)
        batch_iou = compute_iou(logits, masks, cfg.num_classes)

        loss_meter.update(loss.item(), images.size(0))
        if not np.isnan(batch_iou):
            iou_meter.update(batch_iou, images.size(0))

    return loss_meter.avg, iou_meter.avg


def run_training(train_loader: DataLoader, valid_loader: DataLoader, cfg: Config) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = UNet(n_channels=3, n_classes=cfg.num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    history: Dict[str, List[float]] = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    best_iou = -float("inf")
    checkpoint_path = cfg.work_dir / cfg.checkpoint_name

    for epoch in range(cfg.num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.num_epochs} =====")
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        val_loss, val_iou = evaluate(model, valid_loader, cfg)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_mIoU={train_iou:.4f} | val_mIoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "epoch": epoch + 1}, checkpoint_path)
            print(f"💾 New best model saved to {checkpoint_path} (val mIoU={best_iou:.4f})")

    return model, history


# =============================================================================
# STEP 8: Visualisation utilities
# =============================================================================

import matplotlib.pyplot as plt


def denormalize(images: np.ndarray) -> np.ndarray:
    """Undo normalization applied during preprocessing."""

    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 3, 1, 1)
    return (images * std) + mean


@torch.no_grad()
def infer_batch(model: nn.Module, loader: DataLoader, cfg: Config, num_samples: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    images, masks, preds = [], [], []

    for batch in loader:
        image_batch = batch["image"].to(DEVICE)
        mask_batch = batch["mask"].to(DEVICE)
        logits = model(image_batch)

        if cfg.num_classes == 1:
            pred_masks = (torch.sigmoid(logits) > 0.5).float()
            mask_batch = mask_batch.float()
        else:
            pred_masks = torch.argmax(logits, dim=1)
            mask_batch = mask_batch.long()

        images.append(image_batch.cpu().numpy())
        masks.append(mask_batch.cpu().numpy())
        preds.append(pred_masks.cpu().numpy())

        if len(images) * image_batch.size(0) >= num_samples:
            break

    images_np = np.concatenate(images, axis=0)[:num_samples]
    masks_np = np.concatenate(masks, axis=0)[:num_samples]
    preds_np = np.concatenate(preds, axis=0)[:num_samples]

    if cfg.num_classes == 1:
        preds_np = preds_np[:, 0]
        masks_np = masks_np[:, 0]

    return images_np, masks_np, preds_np


def plot_predictions(images: np.ndarray, masks: np.ndarray, preds: np.ndarray, cfg: Config) -> None:
    images = denormalize(images)
    images = np.clip(images, 0.0, 1.0)
    images = np.transpose(images, (0, 2, 3, 1))  # to HWC
    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))

    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (img, gt, pred) in enumerate(zip(images, masks, preds)):
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title("Image")
        axes[idx, 1].imshow(gt, cmap="viridis")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 2].imshow(pred, cmap="viridis")
        axes[idx, 2].set_title("Prediction")

        for ax in axes[idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 9: Orchestrate the training run (update DATASET_ROOT before executing)
# =============================================================================

def main() -> None:
    if not DATASET_ROOT or not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "DATASET_ROOT is not set to a valid directory. Complete STEP 3 or point DATASET_ROOT to your dataset."
        )

    print(f"🚀 Starting training with data at {DATASET_ROOT}")
    train_loader, valid_loader, test_loader = build_dataloaders(DATASET_ROOT, cfg)
    model, history = run_training(train_loader, valid_loader, cfg)

    if test_loader is not None:
        print("\n===== Evaluating on test set =====")
        test_loss, test_iou = evaluate(model, test_loader, cfg)
        print(f"Test loss: {test_loss:.4f} | Test mIoU: {test_iou:.4f}")

    print("\n===== Sample Predictions =====")
    sample_loader = test_loader or valid_loader
    images, masks, preds = infer_batch(model, sample_loader, cfg, num_samples=3)
    plot_predictions(images, masks, preds, cfg)


if __name__ == "__main__":
    if os.getenv("RUN_TRAINING", "0") == "1":
        main()
    else:
        print(
            "✋ Notebook mode: copy each STEP section into separate Colab cells. "
            "Set RUN_TRAINING=1 to execute end-to-end in a script context."
        )

