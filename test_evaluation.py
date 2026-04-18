"""
evaluate_test_videos.py
-----------------------
Evaluate labeled test videos using the exact pipeline implemented in
inference.py, while preserving evaluation-style metrics and CSV outputs.

Expected folder structure:
  test_videos/
    real/ or real_videos/
    fake/ or deepfake_videos/
"""

import argparse
import csv
import heapq
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import inference as inference_module
    from inference import DeepfakeInferenceEngine
except ImportError:
    sys.exit(
        "[ERROR] Cannot import DeepfakeInferenceEngine from inference.py. "
        "Run this script from the project root."
    )


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


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


class Evaluator:
    def __init__(self, checkpoint_path: str, source_mode: str, gradcam: bool = False):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            sys.exit(f"[ERROR] Checkpoint not found: {checkpoint_path}")

        self.engine = DeepfakeInferenceEngine(
            model_path=str(checkpoint),
            source=source_mode,
            gradcam=gradcam,
        )
        self.threshold = float(inference_module.FAKE_THRESHOLD)
        self.device = self.engine.device

    def _reset_engine_video_state(self) -> None:
        self.engine.reset_state()
        self.engine.frame_id = 0
        if hasattr(self.engine, "tracker"):
            self.engine.tracker.frame_index = -1
            self.engine.tracker.last_observation = None
        if hasattr(self.engine, "blinker"):
            self.engine.blinker.ear_history.clear()

    def _analyze_video_stream(
        self,
        video_path: str,
        max_frames: int | None = None,
        save_dir: str | None = None,
        top_k: int = 6,
    ) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "probability": None,
                "predicted_label": None,
                "predicted_class": "UNKNOWN",
                "analysis_status": "UNKNOWN",
                "frames_processed": 0,
                "decision_frames": 0,
                "fake_votes": 0,
                "skipped_reason": "cannot_open",
                "saved_scans": 0,
            }

        self._reset_engine_video_state()
        stem = Path(video_path).stem
        frames_processed = 0
        decision_frames = 0
        fake_votes = 0
        prob_sum = 0.0
        analysis_status = "UNKNOWN"
        top_candidates: List[Tuple[float, int, np.ndarray]] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.engine.process_frame(frame)
                frames_processed += 1
                analysis_status = result.analysis_status

                decision_ready = (
                    result.face_state == "ok"
                    and self.engine.buf.ready()
                )
                if decision_ready:
                    score = float(result.smooth_prob)
                    decision_frames += 1
                    prob_sum += score
                    if result.prediction_label == "FAKE":
                        fake_votes += 1

                    if save_dir and result.cam_overlay is not None and result.bbox is not None and top_k > 0:
                        display = self.engine.render_prediction(frame, result)
                        candidate = (score, int(result.frame_id), display.copy())
                        if len(top_candidates) < top_k:
                            heapq.heappush(top_candidates, candidate)
                        elif score > top_candidates[0][0]:
                            heapq.heapreplace(top_candidates, candidate)

                if max_frames is not None and frames_processed >= max_frames:
                    break
        finally:
            cap.release()

        if frames_processed == 0:
            return {
                "probability": None,
                "predicted_label": None,
                "predicted_class": "UNKNOWN",
                "analysis_status": "UNKNOWN",
                "frames_processed": 0,
                "decision_frames": 0,
                "fake_votes": 0,
                "skipped_reason": "no_readable_frames",
                "saved_scans": 0,
            }

        if decision_frames == 0:
            return {
                "probability": None,
                "predicted_label": None,
                "predicted_class": "UNKNOWN",
                "analysis_status": analysis_status,
                "frames_processed": frames_processed,
                "decision_frames": 0,
                "fake_votes": 0,
                "skipped_reason": "no_decision_ready_frames",
                "saved_scans": 0,
            }

        avg_prob = prob_sum / decision_frames
        final_class = "FAKE" if fake_votes >= max(1, (decision_frames + 1) // 2) else "REAL"
        final_label = 1 if final_class == "FAKE" else 0

        saved_scans = 0
        if save_dir and top_candidates:
            out_root = Path(save_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            video_scan_dir = out_root / stem
            video_scan_dir.mkdir(parents=True, exist_ok=True)

            for rank, (score, frame_id, display) in enumerate(
                sorted(top_candidates, key=lambda item: item[0], reverse=True),
                start=1,
            ):
                out_path = video_scan_dir / f"{rank:02d}_frame_{frame_id:05d}_score_{score:.4f}.jpg"
                cv2.imwrite(str(out_path), display)
                saved_scans += 1

        return {
            "probability": avg_prob,
            "predicted_label": final_label,
            "predicted_class": final_class,
            "analysis_status": analysis_status,
            "frames_processed": frames_processed,
            "decision_frames": decision_frames,
            "fake_votes": fake_votes,
            "skipped_reason": None,
            "saved_scans": saved_scans,
        }

    def predict_video(self, video_path: str, max_frames: int | None = None) -> Dict:
        return self._analyze_video_stream(video_path, max_frames=max_frames)

    def predict_video_with_scans(
        self,
        video_path: str,
        max_frames: int | None = None,
        save_dir: str | None = None,
        top_k: int = 6,
    ) -> Dict:
        return self._analyze_video_stream(
            video_path,
            max_frames=max_frames,
            save_dir=save_dir,
            top_k=top_k,
        )


def run_evaluation(args):
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[Eval] Requested device: {requested_device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir = out_dir / "gradcam_scans" if args.gradcam else None
    if gradcam_dir:
        gradcam_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_test_videos(args.test_dir)
    if not samples:
        sys.exit("[ERROR] No videos found.")

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        source_mode=args.source_mode,
        gradcam=args.gradcam,
    )
    print(f"[Eval] Exact inference device: {evaluator.device}")
    print(f"[Eval] Effective threshold from inference.py: {evaluator.threshold:.4f}")

    results: List[Dict] = []
    all_true: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []
    skipped_videos: List[Tuple[str, str]] = []

    total = len(samples)
    t_global = time.perf_counter()

    print(f"\n{'=' * 110}")
    print(f"  {'#':>4}  {'Video':40}  {'True':>5}  {'Score':>7}  {'Pred':>6}  {'Frames':>7}  {'Result':>8}")
    print(f"{'=' * 110}")

    for idx, (video_path, true_label) in enumerate(samples, start=1):
        video_name = Path(video_path).name
        short_name = video_name[:38] + ".." if len(video_name) > 40 else video_name

        if args.verbose:
            print(f"\n  [{idx}/{total}] Processing: {video_name}")

        t0 = time.perf_counter()
        if args.gradcam:
            pred = evaluator.predict_video_with_scans(
                video_path,
                max_frames=args.max_frames,
                save_dir=str(gradcam_dir),
                top_k=args.gradcam_top_k,
            )
        else:
            pred = evaluator.predict_video(video_path, max_frames=args.max_frames)
        elapsed = time.perf_counter() - t0

        if pred["probability"] is None:
            reason = pred["skipped_reason"] or "unknown"
            skipped_videos.append((video_path, reason))
            print(
                f"  {idx:>4}  {short_name:40}  "
                f"{'REAL' if true_label == 0 else 'FAKE':>5}  {'---':>7}  "
                f"{'SKIP':>6}  {0:>7}  {reason:>8}"
            )
            continue

        prob = float(pred["probability"])
        pred_label = int(pred["predicted_label"])
        pred_class = str(pred["predicted_class"])
        correct = pred_label == true_label

        all_true.append(true_label)
        all_probs.append(prob)
        all_preds.append(pred_label)

        true_class = "REAL" if true_label == 0 else "FAKE"
        result_text = "OK" if correct else "WRONG"

        results.append(
            {
                "filename": video_name,
                "video_path": video_path,
                "true_label": true_label,
                "true_class": true_class,
                "predicted_score": round(prob, 5),
                "predicted_label": pred_label,
                "predicted_class": pred_class,
                "correct": correct,
                "frames_processed": pred["frames_processed"],
                "decision_frames": pred["decision_frames"],
                "fake_votes": pred["fake_votes"],
                "analysis_status": pred["analysis_status"],
                "saved_scans": pred.get("saved_scans", 0),
                "elapsed_s": round(elapsed, 2),
            }
        )

        print(
            f"  {idx:>4}  {short_name:40}  {true_class:>5}  {prob:>7.4f}  "
            f"{pred_class:>6}  {pred['frames_processed']:>7}  {result_text:>8}"
        )

    elapsed_total = time.perf_counter() - t_global
    print(f"{'=' * 110}\n")

    if len(all_true) < 2:
        print("[WARN] Too few valid predictions to compute metrics.")
        print(f"  Evaluated: {len(all_true)} | Skipped: {len(skipped_videos)}")
        return

    all_true_np = np.array(all_true)
    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)

    try:
        test_auc = roc_auc_score(all_true_np, all_probs_np)
    except ValueError:
        test_auc = float("nan")

    acc = accuracy_score(all_true_np, all_preds_np)
    prec = precision_score(all_true_np, all_preds_np, zero_division=0)
    rec = recall_score(all_true_np, all_preds_np, zero_division=0)
    f1 = f1_score(all_true_np, all_preds_np, zero_division=0)
    cm = confusion_matrix(all_true_np, all_preds_np, labels=[0, 1])
    report = classification_report(
        all_true_np,
        all_preds_np,
        target_names=["Real", "Fake"],
        zero_division=0,
    )

    try:
        fpr, tpr, thresholds = roc_curve(all_true_np, all_probs_np)
        optimal_thresh = float(thresholds[int(np.argmax(tpr - fpr))])
    except ValueError:
        optimal_thresh = evaluator.threshold

    summary_lines: List[str] = []

    def p(line: str = ""):
        print(line)
        summary_lines.append(line)

    p("+" + "-" * 62 + "+")
    p("|              SUSFACE V4 - EXACT INFERENCE EVAL            |")
    p("+" + "-" * 62 + "+")
    p()
    p(f"  Checkpoint        : {args.checkpoint}")
    p(f"  Test directory    : {args.test_dir}")
    p(f"  Source mode       : {args.source_mode}")
    p(f"  Max frames/video  : {args.max_frames if args.max_frames is not None else 'all'}")
    p(f"  Inference threshold: {evaluator.threshold:.4f}")
    p(f"  Grad-CAM scans    : {'enabled' if args.gradcam else 'disabled'}")
    if args.gradcam:
        p(f"  Grad-CAM top-k    : {args.gradcam_top_k}")
    p(f"  Optimal threshold : {optimal_thresh:.4f}  (Youden's J)")
    p(f"  Videos evaluated  : {len(all_true)}")
    p(f"  Videos skipped    : {len(skipped_videos)}")
    p(f"  Decision frames   : {sum(row['decision_frames'] for row in results)}")
    p(f"  Total time        : {elapsed_total:.1f}s")
    p()
    p("  Metrics:")
    p(f"    AUC-ROC     : {test_auc:.4f}")
    p(f"    Accuracy    : {acc:.4f}  ({int(acc * len(all_true))}/{len(all_true)})")
    p(f"    Precision   : {prec:.4f}")
    p(f"    Recall      : {rec:.4f}")
    p(f"    F1-Score    : {f1:.4f}")
    p()
    p("  Confusion Matrix (rows=true, cols=predicted):")
    p("                  Pred REAL    Pred FAKE")
    p(f"    True REAL       {cm[0][0]:>5}        {cm[0][1]:>5}")
    p(f"    True FAKE       {cm[1][0]:>5}        {cm[1][1]:>5}")
    p()
    p("  Classification Report:")
    for line in report.strip().split("\n"):
        p(f"    {line}")
    p()

    if skipped_videos:
        p(f"  Skipped Videos ({len(skipped_videos)}):")
        for sv_path, sv_reason in skipped_videos:
            p(f"    {Path(sv_path).name:50}  reason: {sv_reason}")
        p()

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "true_label",
                "true_class",
                "predicted_score",
                "predicted_label",
                "predicted_class",
                "correct",
                "frames_processed",
                "decision_frames",
                "fake_votes",
                "analysis_status",
                "saved_scans",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: v for k, v in row.items() if k != "video_path"})

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {summary_path}")
    if gradcam_dir:
        print(f"  Saved: {gradcam_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate videos using the exact inference.py pipeline.",
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
        help="Directory to save results.csv and summary.txt",
    )
    parser.add_argument(
        "--source_mode",
        choices=["video", "webcam"],
        default="video",
        help="Exact source mode passed into DeepfakeInferenceEngine",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on frames processed per video",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Reported requested device; actual execution follows inference.py",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Enable Grad-CAM in the exact inference engine",
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
        help="Print per-video progress",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
