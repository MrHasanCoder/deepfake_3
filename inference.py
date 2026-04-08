"""
inference.py
------------
Real-time deepfake detection pipeline:

  Mode 1: Webcam (OpenCV)
  Mode 2: Video file
  Mode 3: Virtual webcam (Google Meet via pyvirtualcam)

Key improvements:
  - Uses MediaPipe face landmarks to refine the face crop instead of bbox only
  - Computes blink / EAR from actual landmarks
  - Applies webcam threshold compensation only for webcam sources
  - Rejects weak face crops (too small / blurry / invalid landmarks)
  - Smooths bounding boxes and probabilities for more stable predictions
  - Resets temporal state after prolonged face loss to avoid stale decisions
"""

import argparse
import collections
import csv
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from dataset_loader import get_transforms
from models import DeepfakeDetector, GradCAM


# ============================================================================
# Constants
# ============================================================================

FAKE_THRESHOLD = 0.65
IMG_SIZE = 224
N_FRAMES = 5
FACE_DETECT_EVERY = 4
SMOOTH_WINDOW = 15
MAX_MISSED_FACE_FRAMES = 12
MIN_FACE_SIZE = 80
MIN_SHARPNESS = 45.0
EMA_ALPHA = 0.22
STATE_SWITCH_MARGIN = 0.08
MIN_SWITCH_FRAMES = 4
SUSPICIOUS_FACE_LOSS_FRAMES = 8

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)


# ============================================================================
# MediaPipe helpers
# ============================================================================


def ensure_asset(local_path: str, url: str) -> str:
    path = Path(local_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


@dataclass
class FaceObservation:
    bbox: Tuple[int, int, int, int]
    detection_score: float
    landmarks: Optional[List]
    landmark_score: float
    refined_bbox: Tuple[int, int, int, int]


class FaceAnalyzer:
    """
    Detects faces and refines the crop using landmarks.
    Keeps the last valid result and smooths bbox motion to reduce jitter.
    """

    def __init__(
        self,
        detector_model_path: str = "face_detection_short_range.tflite",
        landmarker_model_path: str = "face_landmarker.task",
        detect_every: int = FACE_DETECT_EVERY,
        bbox_smooth_alpha: float = 0.65,
    ):
        detector_model_path = ensure_asset(detector_model_path, FACE_DETECTOR_URL)
        landmarker_model_path = ensure_asset(landmarker_model_path, FACE_LANDMARKER_URL)

        detector_opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=detector_model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        landmarker_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=landmarker_model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.detector = mp_vision.FaceDetector.create_from_options(detector_opts)
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_opts)
        self.detect_every = detect_every
        self.bbox_smooth_alpha = bbox_smooth_alpha
        self.frame_index = -1
        self.last_observation: Optional[FaceObservation] = None

    @staticmethod
    def _to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _clip_bbox(
        bbox: Tuple[int, int, int, int], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(x1 + 1, min(width, int(x2)))
        y2 = max(y1 + 1, min(height, int(y2)))
        return x1, y1, x2, y2

    def _detect_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        h, w = frame_bgr.shape[:2]
        result = self.detector.detect(self._to_mp_image(frame_bgr))
        if not result.detections:
            return None

        best = max(result.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
        bb = best.bounding_box
        bbox = self._clip_bbox((bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height), w, h)
        score = best.categories[0].score if best.categories else 0.0
        return bbox, float(score)

    def _landmarks_from_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[List], float]:
        result = self.landmarker.detect(self._to_mp_image(frame_bgr))
        if not result.face_landmarks:
            return None, 0.0
        landmarks = result.face_landmarks[0]
        presence_score = 1.0
        if result.face_blendshapes:
            presence_score = float(np.mean([c.score for c in result.face_blendshapes[0]]))
        return landmarks, presence_score

    def _refine_bbox(
        self, landmarks: List, frame_shape: Tuple[int, int, int], margin: float = 0.22
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        xs = np.array([lm.x * w for lm in landmarks], dtype=np.float32)
        ys = np.array([lm.y * h for lm in landmarks], dtype=np.float32)

        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        bw = x2 - x1
        bh = y2 - y1
        mx = bw * margin
        my = bh * margin
        return self._clip_bbox((x1 - mx, y1 - my, x2 + mx, y2 + my), w, h)

    def _smooth_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if self.last_observation is None:
            return bbox
        prev = np.array(self.last_observation.refined_bbox, dtype=np.float32)
        cur = np.array(bbox, dtype=np.float32)
        smoothed = self.bbox_smooth_alpha * cur + (1.0 - self.bbox_smooth_alpha) * prev
        return tuple(int(v) for v in smoothed)

    def analyze(self, frame_bgr: np.ndarray) -> Optional[FaceObservation]:
        self.frame_index += 1

        detected = None
        if self.frame_index % self.detect_every == 0 or self.last_observation is None:
            detected = self._detect_bbox(frame_bgr)

        if detected is None and self.last_observation is None:
            return None

        base_bbox, det_score = detected if detected is not None else (
            self.last_observation.bbox,
            self.last_observation.detection_score,
        )
        landmarks, landmark_score = self._landmarks_from_frame(frame_bgr)

        refined_bbox = base_bbox
        if landmarks:
            refined_bbox = self._refine_bbox(landmarks, frame_bgr.shape)
        refined_bbox = self._smooth_bbox(refined_bbox)

        obs = FaceObservation(
            bbox=base_bbox,
            detection_score=det_score,
            landmarks=landmarks,
            landmark_score=landmark_score,
            refined_bbox=refined_bbox,
        )
        self.last_observation = obs
        return obs


# ============================================================================
# Blink Detection
# ============================================================================


class BlinkDetector:
    def __init__(self, history_len: int = 60):
        self.ear_history: Deque[float] = collections.deque(maxlen=history_len)

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def update(self, landmarks: Optional[List], img_w: int, img_h: int) -> Tuple[float, float]:
        if not landmarks:
            return 0.0, 0.0

        def pts(indices: List[int]) -> np.ndarray:
            return np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices], dtype=np.float32)

        left_ear = self._ear(pts(LEFT_EYE_IDX))
        right_ear = self._ear(pts(RIGHT_EYE_IDX))
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        variance = float(np.var(self.ear_history))
        return float(ear), variance


# ============================================================================
# Preprocessing and quality gating
# ============================================================================


_val_transform = get_transforms("val", IMG_SIZE)


def preprocess_face(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return _val_transform(rgb).unsqueeze(0).to(device)


def crop_face(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def face_quality_ok(face_bgr: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> Tuple[bool, str, float]:
    if face_bgr is None:
        return False, "empty_crop", 0.0

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if min(bw, bh) < MIN_FACE_SIZE:
        return False, "small_face", 0.0

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        return False, "blurry", sharpness

    return True, "ok", sharpness


# ============================================================================
# Temporal buffer and smoothing
# ============================================================================


class FrameBuffer:
    def __init__(self, n_frames: int = N_FRAMES):
        self.n_frames = n_frames
        self.buffer: Deque[torch.Tensor] = collections.deque(maxlen=n_frames)

    def push(self, frame_tensor: torch.Tensor) -> None:
        self.buffer.append(frame_tensor.squeeze(0))

    def ready(self) -> bool:
        return len(self.buffer) == self.n_frames

    def reset(self) -> None:
        self.buffer.clear()

    def get_sequence(self) -> torch.Tensor:
        return torch.stack(list(self.buffer), dim=0).unsqueeze(0)


class ProbSmoother:
    def __init__(self, window: int = SMOOTH_WINDOW):
        self.history: Deque[float] = collections.deque(maxlen=window)
        self.ema_value: Optional[float] = None

    def update(self, prob: float) -> float:
        self.history.append(prob)
        median = float(np.median(self.history))
        if self.ema_value is None:
            self.ema_value = median
        else:
            self.ema_value = (EMA_ALPHA * median) + ((1.0 - EMA_ALPHA) * self.ema_value)
        return float(self.ema_value)

    def reset(self) -> None:
        self.history.clear()
        self.ema_value = None


class PredictionStabilizer:
    """
    Uses hysteresis plus minimum consecutive frames before switching label.
    This prevents rapid REAL/FAKE flips when the score hovers near threshold.
    """

    def __init__(
        self,
        threshold: float,
        margin: float = STATE_SWITCH_MARGIN,
        min_switch_frames: int = MIN_SWITCH_FRAMES,
    ):
        self.threshold = threshold
        self.margin = margin
        self.min_switch_frames = min_switch_frames
        self.state = "REAL"
        self.candidate_state: Optional[str] = None
        self.candidate_count = 0

    def update(self, prob: float) -> str:
        if self.state == "REAL":
            desired_state = "FAKE" if prob >= (self.threshold + self.margin) else "REAL"
        else:
            desired_state = "REAL" if prob <= (self.threshold - self.margin) else "FAKE"

        if desired_state == self.state:
            self.candidate_state = None
            self.candidate_count = 0
            return self.state

        if self.candidate_state != desired_state:
            self.candidate_state = desired_state
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= self.min_switch_frames:
            self.state = desired_state
            self.candidate_state = None
            self.candidate_count = 0

        return self.state

    def reset(self) -> None:
        self.state = "REAL"
        self.candidate_state = None
        self.candidate_count = 0


# ============================================================================
# Overlay
# ============================================================================


def render_overlay(
    frame: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    prob: float,
    raw_prob: float,
    prediction_label: str,
    analysis_status: str,
    ear: float,
    ear_var: float,
    face_state: str,
    quality_score: float,
    cam: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = frame.copy()
    is_fake = prediction_label == "FAKE"
    if analysis_status == "SUSPICIOUS_NO_FACE":
        color = (0, 165, 255)
    else:
        color = (0, 0, 220) if is_fake else (0, 200, 0)

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_fake else 2)

        if cam is not None:
            face_h, face_w = y2 - y1, x2 - x1
            if face_h > 0 and face_w > 0:
                heatmap = cv2.resize(cam, (face_w, face_h))
                heatmap = np.clip(heatmap, 0.0, 1.0)
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                roi = out[y1:y2, x1:x2]
                if roi.shape[:2] == heatmap.shape[:2]:
                    out[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)

    h, _ = out.shape[:2]
    headline = "SUSPICIOUS  FACE NOT VISIBLE" if analysis_status == "SUSPICIOUS_NO_FACE" else f"{prediction_label}  {prob:.1%}"
    cv2.putText(out, headline, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(out, f"Raw: {raw_prob:.3f}  Smooth: {prob:.3f}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(out, f"EAR: {ear:.3f}  Var: {ear_var:.5f}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(out, f"Face: {face_state}  Sharpness: {quality_score:.1f}", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    if analysis_status == "SUSPICIOUS_NO_FACE":
        cv2.putText(out, "Prediction paused until a clear face returns", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, f"Threshold: {FAKE_THRESHOLD:.2f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return out


# ============================================================================
# Main loop
# ============================================================================


def is_webcam_source(source: str) -> bool:
    return source == "webcam" or source == "0"


def run_inference(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    model = DeepfakeDetector(n_segment=N_FRAMES).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    global FAKE_THRESHOLD
    saved_thresh = float(ckpt.get("best_thresh", 0.5))
    threshold_bump = 0.15 if is_webcam_source(args.source) else 0.0
    FAKE_THRESHOLD = min(saved_thresh + threshold_bump, 0.80)
    print(
        f"[Inference] Model loaded | Trained thresh: {saved_thresh:.4f} "
        f"| Applied bump: {threshold_bump:.2f} | Final thresh: {FAKE_THRESHOLD:.4f}"
    )
    print(f"[Inference] Best AUC from training: {ckpt.get('best_auc', '?')}")

    grad_cam = GradCAM(model) if args.gradcam else None

    cap_source = 0 if is_webcam_source(args.source) else args.source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = FaceAnalyzer(
        detector_model_path=args.face_detector_model,
        landmarker_model_path=args.face_landmarker_model,
    )
    blinker = BlinkDetector(history_len=60)
    buf = FrameBuffer(n_frames=N_FRAMES)
    smoother = ProbSmoother(window=SMOOTH_WINDOW)
    stabilizer = PredictionStabilizer(threshold=FAKE_THRESHOLD)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "frame_predictions.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "frame_id",
            "timestamp_ms",
            "raw_prob",
            "smooth_prob",
            "prediction",
            "analysis_status",
            "ear",
            "ear_variance",
            "face_state",
            "sharpness",
            "latency_ms",
        ]
    )

    virtual_cam = None
    if args.virtual_cam:
        try:
            import pyvirtualcam

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            virtual_cam = pyvirtualcam.Camera(width=frame_w, height=frame_h, fps=30)
            print(f"[VirtualCam] Sending to: {virtual_cam.device}")
        except ImportError:
            print("[VirtualCam] pyvirtualcam not installed. Skipping.")

    frame_id = 0
    raw_prob = 0.5
    smooth_prob = 0.5
    bbox = None
    ear = 0.0
    ear_var = 0.0
    cam_overlay = None
    face_state = "searching"
    sharpness = 0.0
    missed_face_frames = 0
    prediction_label = "REAL"
    analysis_status = "OK"

    print("[Inference] Running... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.perf_counter()
            observation = tracker.analyze(frame)
            face_tensor = None

            if observation is None:
                missed_face_frames += 1
                bbox = None
                ear = 0.0
                ear_var = 0.0
                face_state = "no_face"
                sharpness = 0.0
            else:
                bbox = observation.refined_bbox
                ear, ear_var = blinker.update(observation.landmarks, frame.shape[1], frame.shape[0])
                face_crop = crop_face(frame, bbox)
                quality_ok, face_state, sharpness = face_quality_ok(face_crop, bbox)

                if quality_ok:
                    face_tensor = preprocess_face(face_crop, device)
                    buf.push(face_tensor)
                    missed_face_frames = 0
                else:
                    missed_face_frames += 1

            analysis_status = "SUSPICIOUS_NO_FACE" if missed_face_frames >= SUSPICIOUS_FACE_LOSS_FRAMES else "OK"

            if missed_face_frames >= MAX_MISSED_FACE_FRAMES:
                buf.reset()
                smoother.reset()
                stabilizer.reset()
                cam_overlay = None
                raw_prob = 0.5
                smooth_prob = 0.5
                prediction_label = "REAL"

            if buf.ready() and face_tensor is not None:
                seq = buf.get_sequence()
                if grad_cam is not None:
                    model.zero_grad(set_to_none=True)
                    logit = model(face_tensor, seq)
                    cam_overlay = grad_cam.generate(face_tensor, seq)
                else:
                    with torch.no_grad():
                        logit = model(face_tensor, seq)
                    cam_overlay = None

                raw_prob = float(torch.sigmoid(logit).reshape(-1)[0].item())
                smooth_prob = smoother.update(raw_prob)
                prediction_label = stabilizer.update(smooth_prob)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if not is_webcam_source(args.source) else int(time.time() * 1000)

            writer.writerow(
                [
                    frame_id,
                    ts_ms,
                    round(raw_prob, 5),
                    round(smooth_prob, 5),
                    prediction_label,
                    analysis_status,
                    round(ear, 5),
                    round(ear_var, 8),
                    face_state,
                    round(sharpness, 2),
                    round(latency_ms, 2),
                ]
            )

            display = render_overlay(
                frame,
                bbox,
                smooth_prob,
                raw_prob,
                prediction_label,
                analysis_status,
                ear,
                ear_var,
                face_state,
                sharpness,
                cam_overlay,
            )
            cv2.putText(
                display,
                f"{latency_ms:.1f}ms",
                (display.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 255, 100),
                1,
            )
            cv2.imshow("DeepfakeDetector", display)

            if virtual_cam is not None:
                virtual_cam.send(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                virtual_cam.sleep_until_next_frame()

            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        csv_file.close()
        if virtual_cam is not None:
            virtual_cam.close()
        cv2.destroyAllWindows()

    print(f"[Inference] Frame log saved to {csv_path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to best_model.pth")
    p.add_argument("--source", default="webcam", help="'webcam', '0', or path to video file")
    p.add_argument("--log_dir", default="inference_logs")
    p.add_argument("--gradcam", action="store_true", help="Enable Grad-CAM overlay")
    p.add_argument("--virtual_cam", action="store_true", help="Feed output to virtual webcam")
    p.add_argument("--face_detector_model", default="face_detection_short_range.tflite")
    p.add_argument("--face_landmarker_model", default="face_landmarker.task")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
