"""
quick_gradcam_analysis.py
-------------------------
Runs a richer Grad-CAM error analysis pass on the validation split.

What it improves over the quick version:
  - ranks and saves the most severe FP/FN/TP cases instead of first-seen cases
  - computes Grad-CAM focus metrics (center, spread, border-focus, center-focus, entropy)
  - saves per-video CSV + JSON summaries for later inspection
  - saves summary plots to compare FP/FN/TP behavior at a glance
  - produces more informative heatmap panels for each saved case

Usage:
    python quick_gradcam_analysis.py --model_path checkpoints/best_model.pth --data_root .
"""

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
TP_PROB_THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05
SAMPLE_STRIDE = 3
val_tf = get_transforms("val", IMG_SIZE)


@dataclass
class CamMetrics:
    center_y: Optional[float]
    center_x: Optional[float]
    spread_y: float
    spread_x: float
    border_ratio: float
    center_ratio: float
    entropy: float
    peak_y: Optional[float]
    peak_x: Optional[float]
    peak_value: float
    valid: bool


@dataclass
class CaseRecord:
    video_path: str
    video_name: str
    label: int
    true_label: str
    pred: int
    pred_label: str
    prob: float
    error: float
    confidence_margin: float
    case_type: str
    severity: float
    cam_center_y: Optional[float]
    cam_center_x: Optional[float]
    cam_spread_y: float
    cam_spread_x: float
    cam_border_ratio: float
    cam_center_ratio: float
    cam_entropy: float
    cam_peak_y: Optional[float]
    cam_peak_x: Optional[float]
    cam_peak_value: float
    cam_valid: bool
    saved_heatmap: Optional[str] = None


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str) -> DeepfakeDetector:
    model = DeepfakeDetector(n_segment=N_FRAMES).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[!] New layers not in checkpoint (will use random init): {len(missing)} keys")
        print(f"    e.g. {missing[0]}")
        print("    -> This is expected if you changed the architecture after training.")
    if unexpected:
        print(f"[!] Unexpected keys ignored: {len(unexpected)}")
    model.eval()
    print(f"[OK] Model loaded from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Preprocess a video into tensors
# ---------------------------------------------------------------------------
def load_video_frames(video_path: str) -> Optional[List[np.ndarray]]:
    return sample_frames(video_path, num_frames=N_FRAMES, stride=SAMPLE_STRIDE)


def video_to_tensors(video_path: str) -> Tuple[Optional[List[np.ndarray]], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Returns (frames, single_frame_tensor, seq_tensor) or (None, None, None) on failure."""
    frames = load_video_frames(video_path)
    if frames is None:
        return None, None, None

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    transformed = [val_tf(f) for f in rgb_frames]
    seq = torch.stack(transformed).unsqueeze(0).to(DEVICE)
    single = transformed[N_FRAMES // 2].unsqueeze(0).to(DEVICE)
    return frames, single, seq


def sanitize_cam(cam: np.ndarray) -> np.ndarray:
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


def analyze_cam(cam: np.ndarray) -> CamMetrics:
    cam = sanitize_cam(cam)
    total = float(cam.sum())
    if total <= 1e-8:
        return CamMetrics(
            center_y=None,
            center_x=None,
            spread_y=0.0,
            spread_x=0.0,
            border_ratio=0.0,
            center_ratio=0.0,
            entropy=0.0,
            peak_y=None,
            peak_x=None,
            peak_value=0.0,
            valid=False,
        )

    h, w = cam.shape
    ys = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    xs = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    row_mass = cam.sum(axis=1)
    col_mass = cam.sum(axis=0)
    cy = float((row_mass * ys).sum() / total)
    cx = float((col_mass * xs).sum() / total)
    spread_y = float(np.sqrt(np.maximum(((row_mass * ((ys - cy) ** 2)).sum() / total), 0.0)))
    spread_x = float(np.sqrt(np.maximum(((col_mass * ((xs - cx) ** 2)).sum() / total), 0.0)))

    border_y = max(1, int(round(h * 0.15)))
    border_x = max(1, int(round(w * 0.15)))
    border_mask = np.zeros_like(cam, dtype=bool)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True
    border_ratio = float(cam[border_mask].sum() / total)

    center_mask = np.zeros_like(cam, dtype=bool)
    cy1, cy2 = int(h * 0.25), int(h * 0.75)
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    center_mask[cy1:cy2, cx1:cx2] = True
    center_ratio = float(cam[center_mask].sum() / total)

    prob_map = cam / total
    entropy = float(-(prob_map * np.log(prob_map + 1e-8)).sum() / np.log(cam.size + 1e-8))

    peak_idx = int(np.argmax(cam))
    peak_y_px, peak_x_px = np.unravel_index(peak_idx, cam.shape)
    peak_y = float(peak_y_px / max(h - 1, 1))
    peak_x = float(peak_x_px / max(w - 1, 1))
    peak_value = float(cam[peak_y_px, peak_x_px])

    return CamMetrics(
        center_y=cy,
        center_x=cx,
        spread_y=spread_y,
        spread_x=spread_x,
        border_ratio=border_ratio,
        center_ratio=center_ratio,
        entropy=entropy,
        peak_y=peak_y,
        peak_x=peak_x,
        peak_value=peak_value,
        valid=True,
    )


def case_severity(label: int, pred: int, prob: float) -> float:
    if label == 0 and pred == 1:
        return prob
    if label == 1 and pred == 0:
        return 1.0 - prob
    if label == 1 and pred == 1:
        return prob
    return 1.0 - prob


def classify_focus(metrics: CamMetrics) -> str:
    if not metrics.valid or metrics.center_y is None or metrics.center_x is None:
        return "invalid"
    if metrics.border_ratio >= 0.45:
        return "border-heavy"
    if metrics.center_ratio >= 0.55:
        return "centered"
    if metrics.center_y < 0.35:
        return "top-biased"
    if metrics.center_y > 0.65:
        return "bottom-biased"
    if metrics.center_x < 0.30:
        return "left-biased"
    if metrics.center_x > 0.70:
        return "right-biased"
    return "diffuse"


def make_case_record(video_path: str, label: int, prob: float, pred: int, cam: np.ndarray) -> CaseRecord:
    metrics = analyze_cam(cam)
    true_label = "Real" if label == 0 else "Fake"
    pred_label = "Fake" if pred == 1 else "Real"

    if label == 0 and pred == 1:
        case_type = "FP"
    elif label == 1 and pred == 0:
        case_type = "FN"
    elif label == 1 and pred == 1:
        case_type = "TP"
    else:
        case_type = "TN"

    return CaseRecord(
        video_path=video_path,
        video_name=Path(video_path).name,
        label=label,
        true_label=true_label,
        pred=pred,
        pred_label=pred_label,
        prob=prob,
        error=abs(prob - float(label)),
        confidence_margin=abs(prob - THRESHOLD),
        case_type=case_type,
        severity=case_severity(label, pred, prob),
        cam_center_y=metrics.center_y,
        cam_center_x=metrics.center_x,
        cam_spread_y=metrics.spread_y,
        cam_spread_x=metrics.spread_x,
        cam_border_ratio=metrics.border_ratio,
        cam_center_ratio=metrics.center_ratio,
        cam_entropy=metrics.entropy,
        cam_peak_y=metrics.peak_y,
        cam_peak_x=metrics.peak_x,
        cam_peak_value=metrics.peak_value,
        cam_valid=metrics.valid,
    )


# ---------------------------------------------------------------------------
# Save one heatmap figure
# ---------------------------------------------------------------------------
def save_heatmap(
    frames: List[np.ndarray],
    record: CaseRecord,
    cam: np.ndarray,
    out_dir: Path,
    idx: int,
) -> Path:
    face_bgr = frames[N_FRAMES // 2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resz = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    cam_norm = sanitize_cam(cam)
    heatmap = cv2.resize(cam_norm, (IMG_SIZE, IMG_SIZE))
    heatmap_u = (heatmap * 255).astype(np.uint8)
    heatmap_c = cv2.applyColorMap(heatmap_u, cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(face_resz, 0.58, heatmap_c, 0.42, 0)

    mask = (heatmap >= np.percentile(heatmap, 85)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    focus_outline = face_resz.copy()
    cv2.drawContours(focus_outline, contours, -1, (255, 80, 80), 2)

    if record.cam_center_x is not None and record.cam_center_y is not None:
        cx = int(record.cam_center_x * (IMG_SIZE - 1))
        cy = int(record.cam_center_y * (IMG_SIZE - 1))
        for img in (overlay, focus_outline):
            cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)

    fig = plt.figure(figsize=(12, 7), facecolor="#131722")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.06, hspace=0.15)
    panels = [
        (face_resz, "Original frame"),
        (heatmap, "Normalized CAM", "viridis"),
        (overlay, "Overlay"),
        (focus_outline, f"High-activation mask ({classify_focus(analyze_cam(cam))})"),
    ]

    for ax_idx, panel in enumerate(panels):
        ax = fig.add_subplot(gs[ax_idx])
        img, title = panel[0], panel[1]
        cmap = panel[2] if len(panel) > 2 else None
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")

    subtitle = (
        f"{record.case_type} #{idx + 1} | True: {record.true_label} -> Pred: {record.pred_label} "
        f"(p={record.prob:.3f}, margin={record.confidence_margin:.3f})\n"
        f"{record.video_name}\n"
        f"center=({record.cam_center_x if record.cam_center_x is not None else -1:.2f},"
        f"{record.cam_center_y if record.cam_center_y is not None else -1:.2f})  "
        f"spread=({record.cam_spread_x:.2f},{record.cam_spread_y:.2f})  "
        f"border={record.cam_border_ratio:.2f}  center_mass={record.cam_center_ratio:.2f}  "
        f"entropy={record.cam_entropy:.2f}"
    )
    fig.suptitle(subtitle, color="white", fontsize=10, y=0.98)

    fname = out_dir / f"{record.case_type}_{idx + 1:03d}_sev{record.severity:.3f}_p{record.prob:.3f}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


def save_summary_plot(records: List[CaseRecord], out_path: Path) -> None:
    if not records:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="#131722")
    for ax in axes.ravel():
        ax.set_facecolor("#1b2230")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#5c677d")

    groups = {
        "FP": [r for r in records if r.case_type == "FP"],
        "FN": [r for r in records if r.case_type == "FN"],
        "TP": [r for r in records if r.case_type == "TP"],
        "TN": [r for r in records if r.case_type == "TN"],
    }
    colors = {"FP": "#ff6b6b", "FN": "#ffd166", "TP": "#4ecdc4", "TN": "#7d8597"}

    ax = axes[0, 0]
    for key in ("FP", "FN", "TP", "TN"):
        vals = [r.prob for r in groups[key]]
        if vals:
            ax.hist(vals, bins=12, alpha=0.55, label=key, color=colors[key])
    ax.axvline(THRESHOLD, color="white", linestyle="--", linewidth=1)
    ax.set_title("Probability distribution", color="white")
    ax.set_xlabel("Predicted fake probability", color="white")
    ax.set_ylabel("Count", color="white")
    ax.legend(facecolor="#1b2230", edgecolor="#5c677d", labelcolor="white")

    ax = axes[0, 1]
    for key in ("FP", "FN", "TP"):
        pts = [(r.cam_center_x, r.cam_center_y) for r in groups[key] if r.cam_valid and r.cam_center_x is not None and r.cam_center_y is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=48, alpha=0.8, label=key, color=colors[key])
    ax.axvline(0.5, color="#adb5bd", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="#adb5bd", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("CAM center positions", color="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.legend(facecolor="#1b2230", edgecolor="#5c677d", labelcolor="white")

    ax = axes[1, 0]
    for key in ("FP", "FN", "TP"):
        xs = [r.cam_border_ratio for r in groups[key] if r.cam_valid]
        ys = [r.cam_center_ratio for r in groups[key] if r.cam_valid]
        if xs and ys:
            ax.scatter(xs, ys, s=48, alpha=0.8, label=key, color=colors[key])
    ax.set_title("Border-focus vs center-focus", color="white")
    ax.set_xlabel("Border activation ratio", color="white")
    ax.set_ylabel("Center activation ratio", color="white")
    ax.legend(facecolor="#1b2230", edgecolor="#5c677d", labelcolor="white")

    ax = axes[1, 1]
    labels = ["FP", "FN", "TP", "TN"]
    values = [len(groups[k]) for k in labels]
    ax.bar(labels, values, color=[colors[k] for k in labels], alpha=0.9)
    ax.set_title("Case counts", color="white")
    ax.set_ylabel("Videos", color="white")

    fig.suptitle("Grad-CAM validation summary", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(records: List[CaseRecord], out_path: Path) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(records: List[CaseRecord]) -> Dict[str, Any]:
    if not records:
        return {"count": 0}

    valid = [r for r in records if r.cam_valid]
    summary: Dict[str, Any] = {
        "count": len(records),
        "avg_prob": round(float(np.mean([r.prob for r in records])), 4),
        "avg_severity": round(float(np.mean([r.severity for r in records])), 4),
        "valid_cam_count": len(valid),
        "focus_profile": {},
    }
    if valid:
        summary.update(
            {
                "avg_cam_center_x": round(float(np.mean([r.cam_center_x for r in valid if r.cam_center_x is not None])), 4),
                "avg_cam_center_y": round(float(np.mean([r.cam_center_y for r in valid if r.cam_center_y is not None])), 4),
                "avg_border_ratio": round(float(np.mean([r.cam_border_ratio for r in valid])), 4),
                "avg_center_ratio": round(float(np.mean([r.cam_center_ratio for r in valid])), 4),
                "avg_entropy": round(float(np.mean([r.cam_entropy for r in valid])), 4),
                "avg_spread_x": round(float(np.mean([r.cam_spread_x for r in valid])), 4),
                "avg_spread_y": round(float(np.mean([r.cam_spread_y for r in valid])), 4),
            }
        )
        counts: Dict[str, int] = {}
        for record in valid:
            key = classify_focus(
                CamMetrics(
                    center_y=record.cam_center_y,
                    center_x=record.cam_center_x,
                    spread_y=record.cam_spread_y,
                    spread_x=record.cam_spread_x,
                    border_ratio=record.cam_border_ratio,
                    center_ratio=record.cam_center_ratio,
                    entropy=record.cam_entropy,
                    peak_y=record.cam_peak_y,
                    peak_x=record.cam_peak_x,
                    peak_value=record.cam_peak_value,
                    valid=record.cam_valid,
                )
            )
            counts[key] = counts.get(key, 0) + 1
        summary["focus_profile"] = counts
    return summary


def summary_recommendations(fp_records: List[CaseRecord], fn_records: List[CaseRecord]) -> List[str]:
    notes: List[str] = []

    if fp_records:
        fp_valid = [r for r in fp_records if r.cam_valid]
        if fp_valid:
            avg_border = float(np.mean([r.cam_border_ratio for r in fp_valid]))
            avg_center = float(np.mean([r.cam_center_ratio for r in fp_valid]))
            avg_y = float(np.mean([r.cam_center_y for r in fp_valid if r.cam_center_y is not None]))
            avg_x = float(np.mean([r.cam_center_x for r in fp_valid if r.cam_center_x is not None]))

            if avg_border >= 0.42:
                notes.append("False positives are border-heavy. Tighten the crop or reduce background leakage around the face.")
            if avg_center <= 0.30:
                notes.append("False positives spend little activation on the central face. Inspect detector alignment and crop stability.")
            if avg_y < 0.35:
                notes.append("False positives are top-biased. Hairline or forehead cues may be dominating.")
            elif avg_y > 0.65:
                notes.append("False positives are bottom-biased. Chin, neck, or compression artifacts near the jaw may be driving predictions.")
            if avg_x < 0.30 or avg_x > 0.70:
                notes.append("False positives are side-biased. Ear or background cues may be leaking into the classifier.")

    if fn_records:
        fn_valid = [r for r in fn_records if r.cam_valid]
        if fn_valid:
            avg_entropy = float(np.mean([r.cam_entropy for r in fn_valid]))
            avg_center = float(np.mean([r.cam_center_ratio for r in fn_valid]))
            avg_y = float(np.mean([r.cam_center_y for r in fn_valid if r.cam_center_y is not None]))
            if avg_entropy >= 0.80:
                notes.append("False negatives have diffuse CAMs. The model may not be locking onto a stable forgery cue.")
            if avg_center >= 0.55:
                notes.append("False negatives are centered but still missed. The classifier head or temporal branch may need harder fake examples.")
            if avg_y > 0.55:
                notes.append("False negatives skew low in the frame. Mouth and chin cues may dominate while eye-region cues remain weak.")

    if not notes:
        notes.append("No dominant CAM failure pattern stood out. Review the saved ranked examples and compare them against training data quality.")
    return notes


def save_json_report(records: List[CaseRecord], out_path: Path) -> None:
    grouped = {
        "FP": [r for r in records if r.case_type == "FP"],
        "FN": [r for r in records if r.case_type == "FN"],
        "TP": [r for r in records if r.case_type == "TP"],
        "TN": [r for r in records if r.case_type == "TN"],
    }
    report = {
        "threshold": THRESHOLD,
        "n_frames": N_FRAMES,
        "img_size": IMG_SIZE,
        "summary": {key: summarize_group(value) for key, value in grouped.items()},
        "recommendations": summary_recommendations(grouped["FP"], grouped["FN"]),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(args) -> None:
    out_dir = Path(args.out_dir)
    case_dirs = {
        "FP": out_dir / "false_positives",
        "FN": out_dir / "false_negatives",
        "TP": out_dir / "true_positives",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in case_dirs.values():
        case_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path)
    grad_cam = GradCAM(model)

    all_samples = collect_video_paths(args.data_root)
    random.seed(args.seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_test = max(1, int(n * args.test_fraction))
    n_val = max(1, int(n * args.val_fraction))
    val_samples = all_samples[n - n_val - n_test : n - n_test]
    if args.limit_videos is not None:
        val_samples = val_samples[: args.limit_videos]

    print(f"[OK] Analysing {len(val_samples)} val videos...")
    print(f"[Info] Seed={args.seed} | threshold={THRESHOLD:.3f} | max_cases_per_type={MAX_CASES}")

    total = 0
    skipped = 0
    invalid_cam_count = 0
    records: List[CaseRecord] = []
    cam_cache: Dict[str, Tuple[List[np.ndarray], np.ndarray]] = {}

    for idx, (video_path, label) in enumerate(val_samples, start=1):
        frames, single, seq = video_to_tensors(video_path)
        if single is None or seq is None or frames is None:
            skipped += 1
            continue

        model.zero_grad(set_to_none=True)
        cam = grad_cam.generate(single, seq)
        with torch.no_grad():
            logit = model(single, seq)
        prob = float(torch.sigmoid(logit).reshape(-1)[0].item())
        pred = 1 if prob >= THRESHOLD else 0
        total += 1

        record = make_case_record(video_path, label, prob, pred, cam)
        if not record.cam_valid:
            invalid_cam_count += 1
        records.append(record)
        cam_cache[video_path] = (frames, cam)

        if idx % 25 == 0:
            print(f"  [Progress] processed {idx}/{len(val_samples)} videos...")

    grad_cam.remove_hooks()

    fp_records = sorted((r for r in records if r.case_type == "FP"), key=lambda r: r.severity, reverse=True)[:MAX_CASES]
    fn_records = sorted((r for r in records if r.case_type == "FN"), key=lambda r: r.severity, reverse=True)[:MAX_CASES]
    tp_records = sorted(
        (r for r in records if r.case_type == "TP" and r.prob >= args.tp_threshold),
        key=lambda r: r.severity,
        reverse=True,
    )[: args.tp_cases]

    for bucket_name, bucket in (("FP", fp_records), ("FN", fn_records), ("TP", tp_records)):
        for save_idx, record in enumerate(bucket):
            frames, cam = cam_cache[record.video_path]
            out_path = save_heatmap(frames, record, cam, case_dirs[bucket_name], save_idx)
            record.saved_heatmap = str(out_path)
            focus_kind = classify_focus(
                CamMetrics(
                    center_y=record.cam_center_y,
                    center_x=record.cam_center_x,
                    spread_y=record.cam_spread_y,
                    spread_x=record.cam_spread_x,
                    border_ratio=record.cam_border_ratio,
                    center_ratio=record.cam_center_ratio,
                    entropy=record.cam_entropy,
                    peak_y=record.cam_peak_y,
                    peak_x=record.cam_peak_x,
                    peak_value=record.cam_peak_value,
                    valid=record.cam_valid,
                )
            )
            print(
                f"  [{bucket_name} #{save_idx + 1}] p={record.prob:.3f} "
                f"sev={record.severity:.3f} focus={focus_kind:12} {record.video_name}"
            )

    records_csv = out_dir / "gradcam_cases.csv"
    report_json = out_dir / "gradcam_report.json"
    summary_png = out_dir / "gradcam_summary.png"
    write_csv(records, records_csv)
    save_json_report(records, report_json)
    save_summary_plot(records, summary_png)

    grouped = {
        "FP": [r for r in records if r.case_type == "FP"],
        "FN": [r for r in records if r.case_type == "FN"],
        "TP": [r for r in records if r.case_type == "TP"],
        "TN": [r for r in records if r.case_type == "TN"],
    }
    recommendations = summary_recommendations(grouped["FP"], grouped["FN"])

    print("\n" + "=" * 70)
    print("  GRAD-CAM ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Videos analysed         : {total}")
    print(f"  Videos skipped          : {skipped}")
    print(f"  False Positives         : {len(grouped['FP'])}")
    print(f"  False Negatives         : {len(grouped['FN'])}")
    print(f"  True Positives          : {len(grouped['TP'])}")
    print(f"  True Negatives          : {len(grouped['TN'])}")
    print(f"  Invalid CAMs            : {invalid_cam_count}")
    print(f"  Saved FP heatmaps       : {len(fp_records)}")
    print(f"  Saved FN heatmaps       : {len(fn_records)}")
    print(f"  Saved TP heatmaps       : {len(tp_records)}")

    for key in ("FP", "FN", "TP"):
        summary = summarize_group(grouped[key])
        if summary["count"] == 0:
            continue
        print(f"\n  {key} summary:")
        print(f"    count={summary['count']}  avg_prob={summary['avg_prob']:.3f}  avg_severity={summary['avg_severity']:.3f}")
        if summary.get("valid_cam_count", 0) > 0:
            print(
                f"    center=({summary['avg_cam_center_x']:.2f},{summary['avg_cam_center_y']:.2f})  "
                f"border={summary['avg_border_ratio']:.2f}  center_mass={summary['avg_center_ratio']:.2f}  "
                f"entropy={summary['avg_entropy']:.2f}"
            )
            print(f"    focus_profile={summary['focus_profile']}")

    print("\n  Recommendations:")
    for item in recommendations:
        print(f"    - {item}")

    print(f"\n  CSV report              : {records_csv}")
    print(f"  JSON report             : {report_json}")
    print(f"  Summary plot            : {summary_png}")
    print(f"  Heatmaps saved to       : {out_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank and analyze Grad-CAM behavior on validation videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", default="checkpoints/best_model.pth")
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--out_dir", default="gradcam_outputs")
    parser.add_argument("--max_cases", type=int, default=40, help="Saved FP and FN examples per type")
    parser.add_argument("--tp_cases", type=int, default=20, help="Saved TP examples")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tp_threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--val_fraction", type=float, default=VAL_SPLIT)
    parser.add_argument("--test_fraction", type=float, default=TEST_SPLIT)
    parser.add_argument("--limit_videos", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAX_CASES = args.max_cases
    THRESHOLD = args.threshold
    TP_PROB_THRESHOLD = args.tp_threshold
    run_analysis(args)
