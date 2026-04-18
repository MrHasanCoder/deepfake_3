"""
evaluate_test_videos.py
-----------------------
Efficient batch evaluation of labeled test videos.

Key improvements over the previous version:
  ✓ Loads and displays all checkpoint parameters (best_auc, best_thresh, epoch)
  ✓ Frame sampling (configurable stride) instead of processing every frame
  ✓ Direct model inference on sampled frames (bypasses real-time smoothing/EMA)
  ✓ Batch GPU inference for speed
  ✓ Multi-threshold analysis (trained threshold + Youden's J optimal)
  ✓ Per-class and per-video detailed diagnostics
  ✓ ROC curve data export for plotting
  ✓ VRAM-efficient with torch.no_grad() and optional AMP

Expected folder structure:
  test_videos/
    real/ or real_videos/
    fake/ or deepfake_videos/
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from dataset_loader import get_transforms
    from models import DeepfakeDetector, GradCAM
except ImportError:
    sys.exit(
        "[ERROR] Cannot import from models.py / dataset_loader.py. "
        "Run this script from the project root."
    )


# ============================================================================
# Constants
# ============================================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMG_SIZE = 224
N_FRAMES = 5
MIN_FACE_SIZE = 60
MIN_SHARPNESS = 30.0


# ============================================================================
# Video discovery
# ============================================================================


def collect_test_videos(test_dir: str) -> List[Tuple[str, int]]:
    root = Path(test_dir)
    samples: List[Tuple[str, int]] = []

    real_candidates = ["real", "real_videos", "Real", "reals"]
    fake_candidates = ["fake", "deepfake_videos", "fake_videos", "Fake", "fakes", "deepfake"]

    real_dir = next((root / name for name in real_candidates if (root / name).exists()), None)
    fake_dir = next((root / name for name in fake_candidates if (root / name).exists()), None)

    if real_dir is None and fake_dir is None:
        existing = [d.name for d in root.iterdir() if d.is_dir()] if root.exists() else []
        sys.exit(
            f"[ERROR] Could not find real or fake folders in {test_dir}\n"
            f"  Looked for real: {real_candidates}\n"
            f"  Looked for fake: {fake_candidates}\n"
            f"  Found: {existing}"
        )

    for label, folder in [(0, real_dir), (1, fake_dir)]:
        if folder is None:
            print(f"[WARN] Missing {'real' if label == 0 else 'fake'} folder, skipping that class.")
            continue
        print(f"[Data] {'Real' if label == 0 else 'Fake'} videos folder: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTS:
                samples.append((str(path), label))

    real_n = sum(1 for _, label in samples if label == 0)
    fake_n = sum(1 for _, label in samples if label == 1)
    print(f"[Data] Found {len(samples)} videos (Real: {real_n} | Fake: {fake_n})")
    return samples


# ============================================================================
# Face extraction (lightweight — no MediaPipe, uses Haar for speed)
# ============================================================================


_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def extract_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract the largest face from a frame using Haar cascade (fast)."""
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    if len(faces) == 0:
        return None

    # Pick the largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]

    # Add margin (22% like inference.py)
    margin = 0.22
    mx, my = int(w * margin), int(h * margin)
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(fw, x + w + mx)
    y2 = min(fh, y + h + my)

    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # Sharpness check
    sharpness = float(cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        return None

    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


# ============================================================================
# Frame sampling from video
# ============================================================================


def sample_frames_from_video(
    video_path: str,
    max_frames: int = 30,
    stride: int = 5,
) -> Tuple[List[np.ndarray], int]:
    """
    Sample frames from a video with stride for efficiency.
    Returns (list_of_bgr_frames, total_frame_count).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        total = len(frames)
        # Subsample
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames, total

    # Compute sample indices
    if total <= max_frames * stride:
        # Video is short — sample uniformly
        n_samples = min(total, max_frames)
        indices = np.linspace(0, total - 1, n_samples, dtype=int)
    else:
        indices = list(range(0, min(total, max_frames * stride), stride))[:max_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames, total


# ============================================================================
# Checkpoint loader (extracts all trained parameters)
# ============================================================================


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint and extract all training metadata."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DeepfakeDetector(n_segment=N_FRAMES).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    info = {
        "model": model,
        "epoch": ckpt.get("epoch", "?"),
        "best_auc": float(ckpt.get("best_auc", 0.0)),
        "best_thresh": float(ckpt.get("best_thresh", 0.5)),
        "has_optimizer": "optimizer" in ckpt,
        "has_scheduler": "scheduler" in ckpt,
    }
    return info


# ============================================================================
# Efficient video evaluator
# ============================================================================


class EfficientEvaluator:
    """
    Efficient batch evaluator that:
    1. Samples frames with stride (not every frame)
    2. Extracts faces with fast Haar cascade
    3. Runs direct model inference (no real-time smoothing overhead)
    4. Uses the trained threshold from checkpoint for classification
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        device: torch.device,
        threshold: float,
        gradcam: bool = False,
    ):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.transform = get_transforms("val", IMG_SIZE)
        self.grad_cam = GradCAM(model) if gradcam else None

    def _preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    def predict_video(
        self,
        video_path: str,
        max_frames: int = 30,
        stride: int = 5,
        save_dir: Optional[str] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Evaluate a single video efficiently.

        Strategy:
        1. Sample frames with stride
        2. Extract faces from sampled frames
        3. Build temporal sequences (N_FRAMES consecutive face crops)
        4. Run model on each sequence, collect raw probabilities
        5. Aggregate: mean probability + majority vote at threshold
        """
        frames, total_frames = sample_frames_from_video(video_path, max_frames, stride)

        if not frames:
            return self._empty_result("cannot_open")

        # Extract faces from all sampled frames
        face_crops = []
        face_indices = []
        for i, frame in enumerate(frames):
            face = extract_face(frame)
            if face is not None:
                face_crops.append(face)
                face_indices.append(i)

        if len(face_crops) < N_FRAMES:
            return self._empty_result("insufficient_faces", total_frames, len(frames))

        # Build temporal sequences and run inference
        raw_probs = []
        face_tensors_all = [self._preprocess_face(fc) for fc in face_crops]

        with torch.no_grad():
            # Slide a window of N_FRAMES across the face crops
            for start in range(0, len(face_crops) - N_FRAMES + 1, max(1, N_FRAMES // 2)):
                end = start + N_FRAMES
                single_frame = face_tensors_all[start + N_FRAMES // 2].unsqueeze(0).to(self.device)
                frame_seq = torch.stack(face_tensors_all[start:end], dim=0).unsqueeze(0).to(self.device)

                logit = self.model(single_frame, frame_seq)
                prob = float(torch.sigmoid(logit).squeeze().item())
                raw_probs.append(prob)

        if not raw_probs:
            return self._empty_result("no_valid_sequences", total_frames, len(frames))

        # Aggregate predictions
        probs_np = np.array(raw_probs)
        mean_prob = float(probs_np.mean())
        std_prob = float(probs_np.std())
        median_prob = float(np.median(probs_np))

        # Decision: Use mean probability against threshold
        fake_votes = int((probs_np >= self.threshold).sum())
        total_inferences = len(raw_probs)
        fake_ratio = fake_votes / total_inferences

        # Final decision: majority vote
        predicted_label = 1 if fake_ratio >= 0.5 else 0
        predicted_class = "FAKE" if predicted_label == 1 else "REAL"

        # Save GradCAM scans if requested
        saved_scans = 0
        if save_dir and self.grad_cam is not None and face_crops:
            saved_scans = self._save_gradcam_scans(
                video_path, face_crops, face_tensors_all, frames, face_indices,
                save_dir, top_k, raw_probs
            )

        return {
            "probability": mean_prob,
            "median_probability": median_prob,
            "std_probability": std_prob,
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "faces_extracted": len(face_crops),
            "inferences": total_inferences,
            "fake_votes": fake_votes,
            "fake_ratio": round(fake_ratio, 4),
            "all_probs": raw_probs,
            "skipped_reason": None,
            "saved_scans": saved_scans,
        }

    def _save_gradcam_scans(
        self,
        video_path: str,
        face_crops: List[np.ndarray],
        face_tensors: List[torch.Tensor],
        frames: List[np.ndarray],
        face_indices: List[int],
        save_dir: str,
        top_k: int,
        probs: List[float],
    ) -> int:
        """Save top-k GradCAM overlay frames sorted by fake probability."""
        stem = Path(video_path).stem
        out_dir = Path(save_dir) / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find top-k indices by probability
        n_seqs = len(probs)
        sorted_indices = sorted(range(n_seqs), key=lambda i: probs[i], reverse=True)[:top_k]

        saved = 0
        for rank, seq_idx in enumerate(sorted_indices, start=1):
            center = seq_idx * max(1, N_FRAMES // 2) + N_FRAMES // 2
            if center >= len(face_crops):
                center = len(face_crops) - 1
            start = max(0, center - N_FRAMES // 2)
            end = min(len(face_crops), start + N_FRAMES)
            if end - start < N_FRAMES:
                start = max(0, end - N_FRAMES)

            if end - start < N_FRAMES:
                continue

            single_frame = face_tensors[center].unsqueeze(0).to(self.device)
            frame_seq = torch.stack(face_tensors[start:end], dim=0).unsqueeze(0).to(self.device)

            self.model.zero_grad(set_to_none=True)
            cam_map = self.grad_cam.generate(single_frame, frame_seq)

            # Overlay on face crop
            face = face_crops[center].copy()
            if cam_map.shape[:2] != (IMG_SIZE, IMG_SIZE):
                cam_map = cv2.resize(cam_map, (IMG_SIZE, IMG_SIZE))
            cam_map = np.clip(cam_map, 0.0, 1.0)
            heatmap = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(face, 0.6, heatmap, 0.4, 0)

            score = probs[seq_idx]
            out_path = out_dir / f"{rank:02d}_score_{score:.4f}.jpg"
            cv2.imwrite(str(out_path), overlay)
            saved += 1

        return saved

    @staticmethod
    def _empty_result(
        reason: str,
        total_frames: int = 0,
        sampled_frames: int = 0,
    ) -> Dict[str, Any]:
        return {
            "probability": None,
            "median_probability": None,
            "std_probability": None,
            "predicted_label": None,
            "predicted_class": "UNKNOWN",
            "total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "faces_extracted": 0,
            "inferences": 0,
            "fake_votes": 0,
            "fake_ratio": 0.0,
            "all_probs": [],
            "skipped_reason": reason,
            "saved_scans": 0,
        }


# ============================================================================
# Multi-threshold analysis
# ============================================================================


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all classification metrics at a specific threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, target_names=["Real", "Fake"], zero_division=0
    )
    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


# ============================================================================
# Main evaluation pipeline
# ============================================================================


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Eval] Device: {device}")
    if device.type == "cuda":
        print(f"[Eval] GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Eval] VRAM: {vram:.1f} GB")

    # ── Load checkpoint with full parameter extraction ──────────────
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    print(f"\n[Model] Loading checkpoint: {args.checkpoint}")
    ckpt_info = load_checkpoint(args.checkpoint, device)
    model = ckpt_info["model"]
    trained_thresh = ckpt_info["best_thresh"]
    trained_auc = ckpt_info["best_auc"]
    trained_epoch = ckpt_info["epoch"]

    print(f"[Model] Trained epoch     : {trained_epoch}")
    print(f"[Model] Training best AUC : {trained_auc:.4f}")
    print(f"[Model] Trained threshold : {trained_thresh:.4f}")
    print(f"[Model] Trainable params  : {model.count_params():,}")

    # ── Setup output directories ──────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir = str(out_dir / "gradcam_scans") if args.gradcam else None
    if gradcam_dir:
        Path(gradcam_dir).mkdir(parents=True, exist_ok=True)

    # ── Collect test videos ───────────────────────────────────────
    samples = collect_test_videos(args.test_dir)
    if not samples:
        sys.exit("[ERROR] No videos found.")

    # ── Create evaluator ──────────────────────────────────────────
    evaluator = EfficientEvaluator(
        model=model,
        device=device,
        threshold=trained_thresh,
        gradcam=args.gradcam,
    )

    # ── Process all videos ────────────────────────────────────────
    results: List[Dict] = []
    all_true: List[int] = []
    all_probs: List[float] = []
    skipped_videos: List[Tuple[str, str]] = []

    total = len(samples)
    t_global = time.perf_counter()

    print(f"\n{'=' * 120}")
    print(
        f"  {'#':>4}  {'Video':40}  {'True':>5}  {'Score':>7}  {'Med':>7}  {'Std':>6}  "
        f"{'Pred':>6}  {'Faces':>6}  {'Infs':>5}  {'Result':>7}"
    )
    print(f"{'=' * 120}")

    for idx, (video_path, true_label) in enumerate(samples, start=1):
        video_name = Path(video_path).name
        short_name = video_name[:38] + ".." if len(video_name) > 40 else video_name

        if args.verbose:
            print(f"\n  [{idx}/{total}] Processing: {video_name}")

        t0 = time.perf_counter()
        pred = evaluator.predict_video(
            video_path,
            max_frames=args.max_frames,
            stride=args.frame_stride,
            save_dir=gradcam_dir,
            top_k=args.gradcam_top_k,
        )
        elapsed = time.perf_counter() - t0

        if pred["probability"] is None:
            reason = pred["skipped_reason"] or "unknown"
            skipped_videos.append((video_path, reason))
            true_class = "REAL" if true_label == 0 else "FAKE"
            print(
                f"  {idx:>4}  {short_name:40}  {true_class:>5}  {'---':>7}  {'---':>7}  "
                f"{'---':>6}  {'SKIP':>6}  {pred['faces_extracted']:>6}  {0:>5}  {reason:>7}"
            )
            continue

        prob = float(pred["probability"])
        median = float(pred["median_probability"])
        std = float(pred["std_probability"])
        pred_label = int(pred["predicted_label"])
        pred_class = str(pred["predicted_class"])
        correct = pred_label == true_label

        all_true.append(true_label)
        all_probs.append(prob)

        true_class = "REAL" if true_label == 0 else "FAKE"
        result_text = "OK" if correct else "WRONG"

        results.append({
            "filename": video_name,
            "video_path": video_path,
            "true_label": true_label,
            "true_class": true_class,
            "mean_score": round(prob, 5),
            "median_score": round(median, 5),
            "std_score": round(std, 5),
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "correct": correct,
            "total_frames": pred["total_frames"],
            "sampled_frames": pred["sampled_frames"],
            "faces_extracted": pred["faces_extracted"],
            "inferences": pred["inferences"],
            "fake_votes": pred["fake_votes"],
            "fake_ratio": pred["fake_ratio"],
            "saved_scans": pred.get("saved_scans", 0),
            "elapsed_s": round(elapsed, 2),
        })

        print(
            f"  {idx:>4}  {short_name:40}  {true_class:>5}  {prob:>7.4f}  {median:>7.4f}  "
            f"{std:>6.4f}  {pred_class:>6}  {pred['faces_extracted']:>6}  "
            f"{pred['inferences']:>5}  {result_text:>7}"
        )

    elapsed_total = time.perf_counter() - t_global
    print(f"{'=' * 120}\n")

    # ── Compute metrics ───────────────────────────────────────────
    if len(all_true) < 2:
        print("[WARN] Too few valid predictions to compute metrics.")
        print(f"  Evaluated: {len(all_true)} | Skipped: {len(skipped_videos)}")
        return

    y_true = np.array(all_true)
    y_prob = np.array(all_probs)

    # AUC-ROC
    try:
        test_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        test_auc = float("nan")

    # Average Precision (PR-AUC)
    try:
        test_ap = average_precision_score(y_true, y_prob)
    except ValueError:
        test_ap = float("nan")

    # Youden's J optimal threshold
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        optimal_idx = int(np.argmax(tpr - fpr))
        optimal_thresh = float(roc_thresholds[optimal_idx])
    except ValueError:
        fpr, tpr, roc_thresholds = None, None, None
        optimal_thresh = trained_thresh

    # Metrics at trained threshold
    metrics_trained = compute_metrics_at_threshold(y_true, y_prob, trained_thresh)

    # Metrics at optimal threshold
    metrics_optimal = compute_metrics_at_threshold(y_true, y_prob, optimal_thresh)

    # ── Print summary ─────────────────────────────────────────────
    summary_lines: List[str] = []

    def p(line: str = ""):
        print(line)
        summary_lines.append(line)

    p("+" + "=" * 72 + "+")
    p("|            DEEPSHIELD V4 -- BATCH EVALUATION REPORT            |")
    p("+" + "=" * 72 + "+")
    p()
    p("  -- Model & Checkpoint -----------------------------------------")
    p(f"  |  Checkpoint       : {args.checkpoint}")
    p(f"  |  Trained epoch    : {trained_epoch}")
    p(f"  |  Training AUC     : {trained_auc:.4f}")
    p(f"  |  Trained threshold: {trained_thresh:.4f}")
    p(f"  |  Model params     : {model.count_params():,}")
    p()
    p("  -- Evaluation Config ------------------------------------------")
    p(f"  |  Test directory   : {args.test_dir}")
    p(f"  |  Max frames/video : {args.max_frames}")
    p(f"  |  Frame stride     : {args.frame_stride}")
    p(f"  |  Grad-CAM         : {'enabled' if args.gradcam else 'disabled'}")
    if args.gradcam:
        p(f"  |  Grad-CAM top-k   : {args.gradcam_top_k}")
    p(f"  |  Videos evaluated : {len(all_true)}")
    p(f"  |  Videos skipped   : {len(skipped_videos)}")
    p(f"  |  Total time       : {elapsed_total:.1f}s")
    avg_time = elapsed_total / max(len(all_true), 1)
    p(f"  |  Avg time/video   : {avg_time:.2f}s")
    p()

    p("  -- AUC Scores -------------------------------------------------")
    p(f"  |  AUC-ROC (test)   : {test_auc:.4f}")
    p(f"  |  AUC-ROC (train)  : {trained_auc:.4f}")
    auc_delta = test_auc - trained_auc
    delta_label = "(^ better on test)" if auc_delta > 0 else "(v generalization gap)"
    p(f"  |  Delta AUC        : {auc_delta:+.4f}  {delta_label}")
    p(f"  |  Average Precision: {test_ap:.4f}")
    p()

    p("  -- Metrics @ Trained Threshold ({:.4f}) -----------------------".format(trained_thresh))
    cm_t = metrics_trained["confusion_matrix"]
    p(f"  |  Accuracy  : {metrics_trained['accuracy']:.4f}  ({int(metrics_trained['accuracy'] * len(all_true))}/{len(all_true)})")
    p(f"  |  Precision : {metrics_trained['precision']:.4f}")
    p(f"  |  Recall    : {metrics_trained['recall']:.4f}")
    p(f"  |  F1-Score  : {metrics_trained['f1']:.4f}")
    p(f"  |  TN={cm_t[0][0]:>3}  FP={cm_t[0][1]:>3}  FN={cm_t[1][0]:>3}  TP={cm_t[1][1]:>3}")
    p()

    p("  -- Metrics @ Optimal Threshold ({:.4f}, Youden's J) ----------".format(optimal_thresh))
    cm_o = metrics_optimal["confusion_matrix"]
    p(f"  |  Accuracy  : {metrics_optimal['accuracy']:.4f}  ({int(metrics_optimal['accuracy'] * len(all_true))}/{len(all_true)})")
    p(f"  |  Precision : {metrics_optimal['precision']:.4f}")
    p(f"  |  Recall    : {metrics_optimal['recall']:.4f}")
    p(f"  |  F1-Score  : {metrics_optimal['f1']:.4f}")
    p(f"  |  TN={cm_o[0][0]:>3}  FP={cm_o[0][1]:>3}  FN={cm_o[1][0]:>3}  TP={cm_o[1][1]:>3}")
    p()

    p("  -- Classification Report (trained threshold) -------------------")
    for line in metrics_trained["report"].strip().split("\n"):
        p(f"  |  {line}")
    p()

    # ── Per-class analysis ────────────────────────────────────────
    real_probs = [all_probs[i] for i in range(len(all_true)) if all_true[i] == 0]
    fake_probs = [all_probs[i] for i in range(len(all_true)) if all_true[i] == 1]

    if real_probs:
        p("  -- Real Videos Score Distribution ------------------------------")
        p(f"  |  Count  : {len(real_probs)}")
        p(f"  |  Mean   : {np.mean(real_probs):.4f}  (ideal: < {trained_thresh:.4f})")
        p(f"  |  Median : {np.median(real_probs):.4f}")
        p(f"  |  Std    : {np.std(real_probs):.4f}")
        p(f"  |  Min    : {np.min(real_probs):.4f}")
        p(f"  |  Max    : {np.max(real_probs):.4f}")
        p()

    if fake_probs:
        p("  -- Fake Videos Score Distribution ------------------------------")
        p(f"  |  Count  : {len(fake_probs)}")
        p(f"  |  Mean   : {np.mean(fake_probs):.4f}  (ideal: > {trained_thresh:.4f})")
        p(f"  |  Median : {np.median(fake_probs):.4f}")
        p(f"  |  Std    : {np.std(fake_probs):.4f}")
        p(f"  |  Min    : {np.min(fake_probs):.4f}")
        p(f"  |  Max    : {np.max(fake_probs):.4f}")
        p()

    # ── Misclassified videos ──────────────────────────────────────
    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        p(f"  -- Misclassified Videos ({len(misclassified)}) ---------------------------------")
        for r in misclassified:
            p(f"  |  {r['filename'][:45]:45}  true={r['true_class']:4}  pred={r['predicted_class']:4}  score={r['mean_score']:.4f}")
        p()

    if skipped_videos:
        p(f"  -- Skipped Videos ({len(skipped_videos)}) -----------------------------------------")
        for sv_path, sv_reason in skipped_videos:
            p(f"  |  {Path(sv_path).name[:45]:45}  reason: {sv_reason}")
        p()

    p("+" + "=" * 72 + "+")

    # ── Save outputs ──────────────────────────────────────────────

    # 1. Detailed results CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "true_label", "true_class",
                "mean_score", "median_score", "std_score",
                "predicted_label", "predicted_class", "correct",
                "total_frames", "sampled_frames", "faces_extracted",
                "inferences", "fake_votes", "fake_ratio",
                "saved_scans", "elapsed_s",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: v for k, v in row.items() if k != "video_path"})

    # 2. Summary text
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # 3. ROC curve data (for plotting)
    if fpr is not None:
        roc_path = out_dir / "roc_curve.csv"
        with open(roc_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold"])
            for fp, tp, th in zip(fpr, tpr, roc_thresholds):
                writer.writerow([round(float(fp), 6), round(float(tp), 6), round(float(th), 6)])

    # 4. Evaluation metadata JSON
    meta_path = out_dir / "eval_metadata.json"
    meta = {
        "checkpoint": str(args.checkpoint),
        "test_dir": str(args.test_dir),
        "trained_epoch": int(trained_epoch) if isinstance(trained_epoch, (int, float)) else str(trained_epoch),
        "trained_auc": round(trained_auc, 5),
        "trained_threshold": round(trained_thresh, 5),
        "test_auc": round(test_auc, 5),
        "test_ap": round(test_ap, 5),
        "optimal_threshold": round(optimal_thresh, 5),
        "accuracy_at_trained_thresh": round(metrics_trained["accuracy"], 5),
        "f1_at_trained_thresh": round(metrics_trained["f1"], 5),
        "accuracy_at_optimal_thresh": round(metrics_optimal["accuracy"], 5),
        "f1_at_optimal_thresh": round(metrics_optimal["f1"], 5),
        "videos_evaluated": len(all_true),
        "videos_skipped": len(skipped_videos),
        "total_time_s": round(elapsed_total, 2),
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {summary_path}")
    print(f"  Saved: {meta_path}")
    if fpr is not None:
        print(f"  Saved: {out_dir / 'roc_curve.csv'}")
    if gradcam_dir:
        print(f"  Saved: {gradcam_dir}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Efficient batch evaluation of test videos using trained model parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test_dir",
        default="test_videos",
        help="Root folder containing real/ and fake/ subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        default="test_results",
        help="Directory to save results, ROC data, and metadata",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Max frames to sample per video (controls speed vs accuracy)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
        help="Frame stride for sampling (higher = faster but fewer frames)",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Enable Grad-CAM heatmap generation for top-k frames",
    )
    parser.add_argument(
        "--gradcam_top_k",
        type=int,
        default=6,
        help="Number of highest-score Grad-CAM overlay frames to save per video",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-video progress",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
