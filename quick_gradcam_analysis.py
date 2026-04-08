"""
quick_gradcam_analysis.py
--------------------------
Analyzes False Positives (Real videos predicted as Fake) using Grad-CAM.
Saves heatmap images so you can visually inspect WHERE the model is looking.

Usage:
    python quick_gradcam_analysis.py --model_path checkpoints/best_model.pth --data_root .
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from dataset_loader import collect_video_paths, get_transforms, sample_frames
from models import DeepfakeDetector, GradCAM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 256
N_FRAMES = 5
MAX_CASES = 40
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_tf = get_transforms("val", IMG_SIZE)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path):
    model = DeepfakeDetector(n_segment=N_FRAMES).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[!] New layers not in checkpoint (will use random init): {len(missing)} keys")
        print(f"    e.g. {missing[0]}")
        print("    -> This is expected if you added CBAM after the checkpoint was saved.")
        print("    -> GradCAM analysis is still valid for all other branches.")
    if unexpected:
        print(f"[!] Unexpected keys ignored: {len(unexpected)}")
    model.eval()
    print(f"[OK] Model loaded from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Preprocess a video into tensors
# ---------------------------------------------------------------------------
def video_to_tensors(video_path):
    """Returns (single_frame_tensor, seq_tensor) or (None, None) on failure."""
    frames = sample_frames(video_path, num_frames=N_FRAMES, stride=3)
    if frames is None:
        return None, None

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    transformed = [val_tf(f) for f in rgb_frames]
    seq = torch.stack(transformed).unsqueeze(0).to(DEVICE)
    single = transformed[N_FRAMES // 2].unsqueeze(0).to(DEVICE)
    return single, seq


def sanitize_cam(cam: np.ndarray) -> np.ndarray:
    """
    Convert CAM into a clean non-negative normalized map.

    Why this matters:
      - some Grad-CAM implementations return signed maps before ReLU
      - tiny negative totals can make center-of-mass collapse to (0, 0)
      - NaNs/Infs can silently poison the summary stats
    """
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.squeeze(cam)
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

    if cam.ndim != 2:
        raise ValueError(f"Expected CAM to be 2D after squeeze, got shape {cam.shape}")

    cam = cam - cam.min()
    max_val = float(cam.max())
    if max_val > 1e-8:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam


def center_of_mass(cam: np.ndarray) -> Optional[Tuple[float, float]]:
    """Returns (cy, cx) normalized center of mass of the activation map, or None if invalid."""
    cam = sanitize_cam(cam)
    total = float(cam.sum())
    if total <= 1e-8:
        return None

    h, w = cam.shape
    ys = np.arange(h, dtype=np.float32) / max(h, 1)
    xs = np.arange(w, dtype=np.float32) / max(w, 1)
    cy = float((cam.sum(axis=1) * ys).sum() / total)
    cx = float((cam.sum(axis=0) * xs).sum() / total)
    return cy, cx


# ---------------------------------------------------------------------------
# Save one heatmap figure
# ---------------------------------------------------------------------------
def save_heatmap(video_path, label, prob, cam, out_dir, case_type, idx):
    """
    Saves a figure with:
      Left  -> original face frame (middle of sequence)
      Right -> GradCAM heatmap overlay
    """
    frames = sample_frames(video_path, num_frames=N_FRAMES, stride=3)
    if frames is None:
        return None

    face_bgr = frames[N_FRAMES // 2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resz = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    cam_norm = sanitize_cam(cam)
    heatmap = cv2.resize(cam_norm, (IMG_SIZE, IMG_SIZE))
    heatmap_u = (heatmap * 255).astype(np.uint8)
    heatmap_c = cv2.applyColorMap(heatmap_u, cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(face_resz, 0.55, heatmap_c, 0.45, 0)

    fig = plt.figure(figsize=(8, 4), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

    panels = [
        (face_resz, "Original face"),
        (overlay, "Grad-CAM focus"),
    ]
    for ax_idx, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[ax_idx])
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")

    true_label = "Real" if label == 0 else "Fake"
    pred_label = "Fake" if prob >= THRESHOLD else "Real"
    color = "#ff6b6b" if case_type == "FP" else "#ffd93d"

    cam_center = center_of_mass(cam)
    center_text = "invalid CAM" if cam_center is None else f"cam_center=({cam_center[1]:.2f},{cam_center[0]:.2f})"

    fig.suptitle(
        f"{case_type} #{idx+1} | True: {true_label} -> Pred: {pred_label} (p={prob:.3f})\n"
        f"{Path(video_path).name} | {center_text}",
        color=color,
        fontsize=10,
        y=1.02,
    )

    fname = out_dir / f"{case_type}_{idx+1:03d}_p{prob:.2f}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(args):
    out_dir = Path(args.out_dir)
    fp_dir = out_dir / "false_positives"
    fn_dir = out_dir / "false_negatives"
    tp_dir = out_dir / "true_positives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    tp_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path)
    grad_cam = GradCAM(model)

    import random

    all_samples = collect_video_paths(args.data_root)
    random.seed(124)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.15))
    val_samples = all_samples[n - n_val - n_test : n - n_test]

    print(f"[OK] Analysing {len(val_samples)} val videos...")

    fp_count = fn_count = tp_count = total = 0
    invalid_cam_count = 0
    fp_stats = []
    fn_stats = []

    for video_path, label in val_samples:
        if fp_count >= MAX_CASES and fn_count >= MAX_CASES and tp_count >= 20:
            break

        single, seq = video_to_tensors(video_path)
        if single is None:
            continue

        model.zero_grad()
        cam = grad_cam.generate(single, seq)

        with torch.no_grad():
            logit = model(single, seq)
        prob = float(torch.sigmoid(logit).item())
        pred = 1 if prob >= THRESHOLD else 0
        total += 1

        cam_center = center_of_mass(cam)
        cam_center_str = "invalid" if cam_center is None else f"({cam_center[1]:.2f},{cam_center[0]:.2f})"
        if cam_center is None:
            invalid_cam_count += 1

        is_fp = label == 0 and pred == 1
        is_fn = label == 1 and pred == 0
        is_tp = label == 1 and pred == 1 and prob >= 0.7

        if is_fp and fp_count < MAX_CASES:
            save_heatmap(video_path, label, prob, cam, fp_dir, "FP", fp_count)
            if cam_center is not None:
                cy, cx = cam_center
                fp_stats.append({"path": video_path, "prob": prob, "cam_y": cy, "cam_x": cx})
            print(f"  [FP #{fp_count+1}] p={prob:.3f}  cam_center={cam_center_str}  {Path(video_path).name}")
            fp_count += 1

        elif is_fn and fn_count < MAX_CASES:
            save_heatmap(video_path, label, prob, cam, fn_dir, "FN", fn_count)
            if cam_center is not None:
                cy, cx = cam_center
                fn_stats.append({"path": video_path, "prob": prob, "cam_y": cy, "cam_x": cx})
            print(f"  [FN #{fn_count+1}] p={prob:.3f}  cam_center={cam_center_str}  {Path(video_path).name}")
            fn_count += 1

        if is_tp and tp_count < 20:
            save_heatmap(video_path, label, prob, cam, tp_dir, "TP", tp_count)
            print(f"  [TP #{tp_count+1}] p={prob:.3f}  cam_center={cam_center_str}  {Path(video_path).name}")
            tp_count += 1

    grad_cam.remove_hooks()

    print("\n" + "=" * 60)
    print("  GRAD-CAM ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Videos analysed : {total}")
    print(f"  False Positives : {fp_count}  (Real -> predicted Fake)")
    print(f"  False Negatives : {fn_count}  (Fake -> predicted Real)")
    print(f"  Invalid CAMs    : {invalid_cam_count}")

    if fp_stats:
        avg_cy = np.mean([s["cam_y"] for s in fp_stats])
        avg_cx = np.mean([s["cam_x"] for s in fp_stats])
        print(f"\n  FP Grad-CAM center avg -> x={avg_cx:.2f}, y={avg_cy:.2f}")
        if avg_cy < 0.35:
            print("  Warning: model focuses on top of frame (hair/forehead/background)")
            print("  -> Tighten face crop margin: change margin=0.2 to margin=0.08")
        elif avg_cy > 0.65:
            print("  Warning: model focuses on bottom of frame (chin/neck/background)")
            print("  -> Tighten face crop margin: change margin=0.2 to margin=0.08")
        elif avg_cx < 0.3 or avg_cx > 0.7:
            print("  Warning: model focuses on sides (ears/background)")
            print("  -> Tighten face crop margin: change margin=0.2 to margin=0.08")
        else:
            print("  OK: model is focusing on central facial region")
            print("  -> Crop margin is likely fine; tune the classifier instead")

    if fn_stats:
        avg_cy = np.mean([s["cam_y"] for s in fn_stats])
        print(f"\n  FN Grad-CAM center avg -> y={avg_cy:.2f}")
        if avg_cy > 0.5:
            print("  Warning: missing fakes because model looks mostly at mouth/chin")
            print("  -> Eye-region cues may be weak; inspect the temporal branch")

    print(f"\n  True Positives  : {tp_count}  (Fake -> correctly predicted Fake, p>=0.70)")
    print(f"\n  Heatmaps saved to: {out_dir}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="checkpoints/best_model.pth")
    p.add_argument("--data_root", default=".")
    p.add_argument("--out_dir", default="gradcam_outputs")
    p.add_argument("--max_cases", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAX_CASES = args.max_cases
    THRESHOLD = args.threshold
    run_analysis(args)
