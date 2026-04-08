"""
train.py
--------
Training pipeline for the 4-branch deepfake detector.

Features:
  ✓ Focal Loss (class imbalance handling)
  ✓ Mixed Precision (AMP fp16) for RTX 3050
  ✓ Gradient Accumulation (effective batch size scaling)
  ✓ Staged unfreezing of MobileNetV3 backbone
  ✓ LR scheduler (CosineAnnealingLR)
  ✓ Checkpoint save/resume
  ✓ tqdm progress bar (per-batch, real-time)
  ✓ Per-batch CSV flushing (instantly visible logs)
  ✓ Early stopping
"""

import os
import csv
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, classification_report
import numpy as np
import torch.serialization
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

from dataset_loader import build_dataloaders
from models import DeepfakeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════
# Focal Loss
# ════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2    → down-weights easy examples (standard)
    alpha=0.75 → weights minority class (Real) higher for 1:4 imbalance

    Uses register_buffer so alpha tensors are pre-allocated on the correct
    device — avoids the Windows CUDA illegal memory access bug caused by
    calling torch.tensor(..., device=...) inside forward().
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha_pos', torch.tensor(alpha))
        self.register_buffer('alpha_neg', torch.tensor(1.0 - alpha))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), targets, reduction="none"
        )
        probs   = torch.sigmoid(logits.squeeze())
        pt      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════

def compute_metrics(all_labels, all_probs, threshold=None):
    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    # Auto-find best threshold via Youden's J statistic
    if threshold is None:
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            threshold = float(thresholds[best_idx])
        except ValueError:
            threshold = 0.5

    preds = (probs >= threshold).astype(int)
    acc   = (preds == labels).mean()

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    f1 = f1_score(labels, preds, zero_division=0)
    return {"acc": acc, "auc": auc, "f1": f1, "threshold": round(threshold, 4)}


# ════════════════════════════════════════════════════
# Training Loop
# ════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, loss_fn,
                    device, accumulation_steps=2, epoch=0,
                    batch_csv_writer=None, batch_csv_file=None):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []
    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"  Train E{epoch+1:02d}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for step, batch in pbar:
        if batch is None:
            continue

        frame_seq, single_frame, labels = batch
        frame_seq    = frame_seq.to(device, non_blocking=True)
        single_frame = single_frame.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        # ── Forward (AMP disabled due to Windows PyTorch bug) ──────
        with autocast(device_type=DEVICE_STR, enabled=False):
            logits = model(single_frame, frame_seq)
            loss   = loss_fn(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        # Skip corrupted batches that produce NaN/inf loss
        if not torch.isfinite(loss):
            print(f"[Warning] Non-finite loss at step {step}, skipping")
            optimizer.zero_grad()
            continue

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Metrics ───────────────────────────────────────────────
        step_loss = loss.item() * accumulation_steps
        total_loss += step_loss
        probs = torch.sigmoid(logits.detach()).squeeze().cpu().numpy()
        batch_probs  = probs.tolist() if probs.ndim > 0 else [float(probs)]
        batch_labels = labels.cpu().numpy().tolist()
        all_probs.extend(batch_probs)
        all_labels.extend(batch_labels)

        # Running accuracy for tqdm display
        batch_preds = (np.array(batch_probs) >= 0.5).astype(int)
        batch_acc   = (batch_preds == np.array(batch_labels)).mean()

        pbar.set_postfix({
            "loss": f"{step_loss:.4f}",
            "acc":  f"{batch_acc:.3f}",
            "vram": f"{torch.cuda.memory_reserved() / 1e9:.1f}GB"
                    if device.type == "cuda" else "cpu",
        }, refresh=True)

        # ── Per-batch CSV flush ────────────────────────────────────
        if batch_csv_writer is not None:
            batch_csv_writer.writerow([
                epoch + 1, step + 1,
                round(step_loss, 5),
                round(float(batch_acc), 4),
            ])
            batch_csv_file.flush()

    pbar.close()
    metrics  = compute_metrics(all_labels, all_probs)
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc="  Val      ", unit="batch",
                dynamic_ncols=True, leave=True)

    for batch in pbar:
        if batch is None:
            continue

        frame_seq, single_frame, labels = batch
        frame_seq    = frame_seq.to(device, non_blocking=True)
        single_frame = single_frame.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        with autocast(device_type=DEVICE_STR, enabled=False):
            logits = model(single_frame, frame_seq)
            loss   = loss_fn(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
        all_labels.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=True)

    pbar.close()
    metrics  = compute_metrics(all_labels, all_probs)
    avg_loss = total_loss / max(len(loader), 1)

    # Per-class recall report (shows if model gave up on "Real")
    preds  = (np.array(all_probs) >= metrics["threshold"]).astype(int)
    report = classification_report(
        np.array(all_labels).astype(int), preds,
        target_names=["Real", "Fake"], zero_division=0
    )
    logger.info(f"\n{report}")

    return avg_loss, metrics


# ════════════════════════════════════════════════════
# Main Training Script
# ════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=5,
        frame_stride=3,
        img_size=256,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = DeepfakeDetector(n_segment=5, freeze_backbone_layers=10).to(device)
    logger.info(f"Model params: {model.count_params():,}")

    # ── Loss & Optimizer ──────────────────────────────────────────
    # .to(device) ensures register_buffer tensors land on CUDA
    loss_fn   = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(device=DEVICE_STR, enabled=False)

    # ── Resume checkpoint ─────────────────────────────────────────
    start_epoch = 0
    best_auc    = 0.0
    best_thresh = 0.5
    ckpt_path   = Path(args.ckpt_dir) / "best_model.pth"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── Epoch-level CSV ───────────────────────────────────────────
    log_csv    = open(Path(args.ckpt_dir) / "training_log.csv", "w",
                      newline="", encoding="utf-8")
    csv_writer = csv.writer(log_csv)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "train_auc",
                         "val_loss", "val_acc", "val_auc", "val_f1",
                         "val_threshold", "lr"])

    # ── Batch-level CSV ───────────────────────────────────────────
    batch_log_file   = open(Path(args.ckpt_dir) / "batch_log.csv", "w",
                            newline="", encoding="utf-8")
    batch_csv_writer = csv.writer(batch_log_file)
    batch_csv_writer.writerow(["epoch", "batch", "batch_loss", "batch_acc"])
    batch_log_file.flush()

    # ── Early stopping ────────────────────────────────────────────
    patience_counter = 0

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Stage-2: unfreeze MobileNet backbone after warmup
        if epoch == args.unfreeze_epoch:
            model.unfreeze_backbone()
            optimizer = optim.Adam([
                {"params": model.branch_mobile.features.parameters(), "lr": args.lr * 0.1},
                {"params": [p for n, p in model.named_parameters()
                            if "branch_mobile.features" not in n], "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn,
            device, accumulation_steps=args.accum_steps, epoch=epoch,
            batch_csv_writer=batch_csv_writer,
            batch_csv_file=batch_log_file,
        )
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch+1:3d}/{args.epochs}] "
            f"Train: loss={train_loss:.4f} acc={train_metrics['acc']:.4f} "
            f"auc={train_metrics['auc']:.4f} thresh={train_metrics['threshold']:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_metrics['acc']:.4f} "
            f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} "
            f"thresh={val_metrics['threshold']:.4f} | "
            f"LR={current_lr:.2e} | {elapsed:.1f}s"
        )

        csv_writer.writerow([
            epoch + 1,
            round(train_loss, 5), round(train_metrics['acc'], 5), round(train_metrics['auc'], 5),
            round(val_loss, 5),   round(val_metrics['acc'], 5),   round(val_metrics['auc'], 5),
            round(val_metrics['f1'], 5), val_metrics['threshold'],
            round(current_lr, 8)
        ])
        log_csv.flush()

        # ── Save best checkpoint ───────────────────────────────────
        if val_metrics["auc"] > best_auc:
            best_auc    = val_metrics["auc"]
            best_thresh = val_metrics["threshold"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_auc":    float(best_auc),
                "best_thresh": float(best_thresh),
            }, ckpt_path)
            logger.info(f"  ✓ Best model saved (AUC={best_auc:.4f}, thresh={best_thresh:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Always save latest
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_auc":   float(best_auc),
            "best_thresh": float(val_metrics["threshold"]),
        }, Path(args.ckpt_dir) / "latest.pth")

    log_csv.close()
    batch_log_file.close()
    logger.info(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoints saved to: {args.ckpt_dir}")


# ════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Deepfake Detector")
    p.add_argument("--data_root",      default=".",          help="Root with datasets/")
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=8,
                   help="Per-step batch (effective = batch_size * accum_steps)")
    p.add_argument("--accum_steps",    type=int,   default=2,
                   help="Gradient accumulation steps")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--num_workers",    type=int,   default=0,
                   help="0 = safe on Windows; 4+ on Linux")
    p.add_argument("--unfreeze_epoch", type=int,   default=10,
                   help="Epoch to unfreeze MobileNetV3 backbone")
    p.add_argument("--patience",       type=int,   default=8,
                   help="Early stopping patience (epochs)")
    p.add_argument("--focal_alpha",    type=float, default=0.25)
    p.add_argument("--focal_gamma",    type=float, default=2.0)
    p.add_argument("--resume",         action="store_true",
                   help="Resume from latest checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)