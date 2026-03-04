# data.py - Improved DIV2K SR dataset + dataloaders (robust, Y-channel optional)
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


VALID_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def _list_images(folder: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS]
    else:
        files = [p for p in folder.glob("*") if p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    # cv2.imread returns BGR by default; caller should pass BGR -> convert to RGB
    if img is None:
        raise ValueError("Loaded image is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _convert_rgb_to_y(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image in uint8 (H,W,3) to Y channel float32 in [0,1].
    Uses OpenCV YCrCb conversion (Y is first channel).
    """
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0].astype(np.float32) / 255.0
    return y


def _safe_crop_coords(h: int, w: int, crop_h: int, crop_w: int) -> Tuple[int, int]:
    if h == crop_h:
        top = 0
    else:
        top = random.randint(0, h - crop_h)
    if w == crop_w:
        left = 0
    else:
        left = random.randint(0, w - crop_w)
    return top, left


def _make_contiguous_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def worker_init_fn(worker_id: int) -> None:
    # ensures different seed per worker but reproducible across runs if torch.manual_seed set
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class SRDataset(Dataset):
    """
    HR-only dataset that produces (lr_upscaled_tensor, hr_tensor).
    If y_channel=True returns single-channel tensors (1,H,W).
    Otherwise returns 3-channel tensors (3,H,W).
    """
    def __init__(
        self,
        hr_dir: str,
        scale: int = 2,
        patch_size: int = 96,
        augment: bool = True,
        recursive: bool = False,
        y_channel: bool = False,
    ):
        self.hr_dir = Path(hr_dir)
        if not self.hr_dir.exists():
            raise ValueError(f"hr_dir does not exist: {hr_dir}")
        self.scale = int(scale)
        # enforce patch_size divisible by scale
        if patch_size % self.scale != 0:
            patch_size = patch_size + (self.scale - (patch_size % self.scale))
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.recursive = bool(recursive)
        self.y_channel = bool(y_channel)

        self.hr_images = _list_images(self.hr_dir, recursive=self.recursive)
        if len(self.hr_images) == 0:
            raise ValueError(f"No images found in {hr_dir} (extensions: {VALID_EXTS})")
        print(f"[SRDataset] Found {len(self.hr_images)} images in {hr_dir}; "
              f"patch_size={self.patch_size}, scale={self.scale}, y_channel={self.y_channel}")

    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int):
        path = self.hr_images[idx]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = _ensure_rgb(img_bgr)

        # Extract HR patch (height/width are patch_size)
        h, w = img_rgb.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            # upscale small images to patch_size
            hr_patch = cv2.resize(img_rgb, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        else:
            top, left = _safe_crop_coords(h, w, self.patch_size, self.patch_size)
            hr_patch = img_rgb[top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            hr_patch = self._augment(hr_patch)

        # Create LR via bicubic downscale then bicubic upscale (SRCNN-style)
        lr_h, lr_w = self.patch_size // self.scale, self.patch_size // self.scale
        lr_small = cv2.resize(hr_patch, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_upscaled = cv2.resize(lr_small, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        if self.y_channel:
            # convert to Y channel (float32 [0,1]), keep single channel
            hr_y = _convert_rgb_to_y(hr_patch)  # (H,W) float32
            lr_y = _convert_rgb_to_y(lr_upscaled)
            hr_tensor = _make_contiguous_tensor(hr_y[np.newaxis, ...])  # (1,H,W)
            lr_tensor = _make_contiguous_tensor(lr_y[np.newaxis, ...])
        else:
            # normal RGB flow, convert to float32 [0,1] and CHW
            hr_f = hr_patch.astype(np.float32) / 255.0
            lr_f = lr_upscaled.astype(np.float32) / 255.0
            # HWC -> CHW
            hr_chw = hr_f.transpose(2, 0, 1).copy()
            lr_chw = lr_f.transpose(2, 0, 1).copy()
            hr_tensor = _make_contiguous_tensor(hr_chw)
            lr_tensor = _make_contiguous_tensor(lr_chw)

        return lr_tensor, hr_tensor, str(path)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # inplace-safe augmentations: flips + 90-degree rotations
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
        return img.copy()


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 16,
    patch_size: int = 96,
    num_workers: int = 4,
    scale: int = 2,
    y_channel: bool = False,
    recursive: bool = False,
    pin_memory: bool = True,
):
    train_ds = SRDataset(train_dir, scale=scale, patch_size=patch_size, augment=True,
                         recursive=recursive, y_channel=y_channel)
    val_ds = SRDataset(val_dir, scale=scale, patch_size=patch_size, augment=False,
                       recursive=recursive, y_channel=y_channel)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # quick smoke test (no errors)
    train_loader, val_loader = create_dataloaders(
        train_dir="./data/DIV2K_train_HR",
        val_dir="./data/DIV2K_valid_HR",
        batch_size=4,
        patch_size=96,
        num_workers=2,
        scale=2,
        y_channel=False,
        recursive=False,
    )

    # iterate one batch
    for lr, hr, fname in train_loader:
        print("LR:", lr.shape, "HR:", hr.shape, "sample path:", fname[0])
        break