"""
dataset_loader.py
-----------------
Custom dataset loader for deepfake_3 folder structure.

Handles:
  celebdf/real  → REAL (label=0)
  celebdf/fake  → FAKE (label=1)
  ffpp/original_sequences/**  → REAL (label=0)
  ffpp/manipulated_sequences/**  → FAKE (label=1)

Outputs 5-frame sequences (for temporal branch) + single frames.
"""

import os
import io
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Compression Augmentation
# ──────────────────────────────────────────────

def simulate_compression(pil_img: Image.Image) -> Image.Image:
    """
    Simulate JPEG compression artifacts by re-encoding at random quality.
    Helps model generalize to real-world compressed deepfakes
    (WhatsApp, Instagram, Google Meet, etc.).
    """
    quality = random.randint(30, 85)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


class SimulateCompression:
    """
    Picklable callable wrapper around simulate_compression.
    Replaces the lambda so num_workers > 0 works on Windows
    (Windows uses 'spawn' which requires all objects to be picklable;
    lambdas defined inside functions are NOT picklable).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        return simulate_compression(img)


# ──────────────────────────────────────────────
# Supported video extensions
# ──────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ──────────────────────────────────────────────
# Helper: Collect (video_path, label) pairs
# ──────────────────────────────────────────────

def collect_video_paths(dataset_root: str) -> List[Tuple[str, int]]:
    """
    Dynamically map the deepfake_3 folder structure to binary labels.
    Returns list of (absolute_video_path, label) where label: 0=real, 1=fake.
    """
    root = Path(dataset_root)
    samples: List[Tuple[str, int]] = []

    # ── CelebDF ──────────────────────────────
    celebdf_root = root / "datasets" / "celebdf"
    if celebdf_root.exists():
        for path in (celebdf_root / "real").rglob("*"):
            if path.suffix.lower() in VIDEO_EXTS:
                samples.append((str(path), 0))
        for path in (celebdf_root / "fake").rglob("*"):
            if path.suffix.lower() in VIDEO_EXTS:
                samples.append((str(path), 1))
        logger.info(f"[CelebDF] Loaded from {celebdf_root}")
    else:
        logger.warning(f"[CelebDF] Not found at {celebdf_root}")

    # ── FaceForensics++ ──────────────────────
    ffpp_root = root / "datasets" / "ffpp"
    if ffpp_root.exists():
        orig_root = ffpp_root / "original_sequences"
        if orig_root.exists():
            for path in orig_root.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTS:
                    samples.append((str(path), 0))

        manip_root = ffpp_root / "manipulated_sequences"
        if manip_root.exists():
            for path in manip_root.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTS:
                    samples.append((str(path), 1))
        logger.info(f"[FF++] Loaded from {ffpp_root}")
    else:
        logger.warning(f"[FF++] Not found at {ffpp_root}")

    # ── Test Videos (WhatsApp-compressed) ─────
    # Adds real/fake WhatsApp videos to improve generalization
    # on compressed, low-quality sources.
    test_root = root / "test_videos"
    if test_root.exists():
        real_names = ["real", "real_videos", "Real", "reals"]
        fake_names = ["fake", "deepfake_videos", "fake_videos", "Fake", "fakes", "deepfake"]

        real_dir = next((test_root / n for n in real_names if (test_root / n).exists()), None)
        fake_dir = next((test_root / n for n in fake_names if (test_root / n).exists()), None)

        tv_real = tv_fake = 0
        if real_dir:
            for path in real_dir.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTS:
                    samples.append((str(path), 0))
                    tv_real += 1
        if fake_dir:
            for path in fake_dir.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTS:
                    samples.append((str(path), 1))
                    tv_fake += 1
        logger.info(f"[TestVideos] Loaded {tv_real} real + {tv_fake} fake from {test_root}")
    else:
        logger.info(f"[TestVideos] Not found at {test_root} (skipping)")

    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    logger.info(f"Total samples -> Real: {real_count} | Fake: {fake_count} | Total: {len(samples)}")
    return samples


# ──────────────────────────────────────────────
# Frame sampler
# ──────────────────────────────────────────────

def sample_frames(video_path: str, num_frames: int = 5,
                  stride: int = 3, max_attempts: int = 3) -> Optional[List[np.ndarray]]:
    """
    Sample `num_frames` evenly-spaced frames from a video.
    stride=3 → every 3rd frame (reduces redundancy, maintains temporal spread).
    Returns list of BGR numpy arrays (H,W,3) or None on failure.
    """
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < num_frames * stride:
            stride = max(1, total // num_frames)

        max_start = max(0, total - num_frames * stride - 1)
        start = random.randint(0, max_start) if max_start > 0 else 0

        indices = [start + i * stride for i in range(num_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        if len(frames) == num_frames:
            return frames

    return None


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_transforms(split: str = "train", size: int = 256):
    """Return torchvision transforms for train/val/test splits."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            # SimulateCompression is a top-level picklable class — works with num_workers > 0
            transforms.RandomApply([SimulateCompression()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────
# Main Dataset Class
# ──────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Multi-modal deepfake dataset.

    Returns:
        frames_tensor : (T, C, H, W)  ← temporal branch (T=5 frames)
        single_frame  : (C, H, W)     ← spatial / textural / spectral branches
        label         : float (0=real, 1=fake)
        valid         : bool
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split: str = "train",
        num_frames: int = 5,
        frame_stride: int = 3,
        img_size: int = 256,
    ):
        self.samples = samples
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = get_transforms(split, img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = sample_frames(video_path, self.num_frames, self.frame_stride)

        if frames is None:
            T, C, H, W = self.num_frames, 3, 256, 256
            return (
                torch.zeros(T, C, H, W),
                torch.zeros(C, H, W),
                torch.tensor(label, dtype=torch.float32),
                False,
            )

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        transformed = [self.transform(f) for f in rgb_frames]
        frames_tensor = torch.stack(transformed, dim=0)
        single_frame = transformed[self.num_frames // 2]

        return (
            frames_tensor,
            single_frame,
            torch.tensor(label, dtype=torch.float32),
            True,
        )


# ──────────────────────────────────────────────
# Collate function (filters invalid samples)
# ──────────────────────────────────────────────

def collate_fn(batch):
    valid = [(ft, sf, lbl) for ft, sf, lbl, ok in batch if ok]
    if not valid:
        return None
    frames_tensors, single_frames, labels = zip(*valid)
    return (
        torch.stack(frames_tensors),
        torch.stack(single_frames),
        torch.stack(labels),
    )


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def build_dataloaders(
    dataset_root: str,
    batch_size: int = 8,
    val_split: float = 0.15,
    test_split: float = 0.05,
    num_workers: int = 2,
    num_frames: int = 5,
    frame_stride: int = 3,
    img_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test dataloaders with:
    - Video-level splitting (no data leakage)
    - Weighted random sampler for class balance
    """
    all_samples = collect_video_paths(dataset_root)
    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_test  = max(1, int(n * test_split))
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    logger.info(f"Split → Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    labels       = [l for _, l in train_samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights      = [1.0 / class_counts[l] for l in labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = DeepfakeDataset(train_samples, "train", num_frames, frame_stride, img_size)
    val_ds   = DeepfakeDataset(val_samples,   "val",   num_frames, frame_stride, img_size)
    test_ds  = DeepfakeDataset(test_samples,  "test",  num_frames, frame_stride, img_size)

    # persistent_workers=True avoids re-spawning workers each epoch (Windows speedup)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    train_loader, val_loader, test_loader = build_dataloaders(root, batch_size=4)
    batch = next(iter(train_loader))
    if batch:
        ft, sf, labels = batch
        print(f"frames_tensor : {ft.shape}")
        print(f"single_frame  : {sf.shape}")
        print(f"labels        : {labels}")