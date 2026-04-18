"""
app.py  —  SusFace web backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uses the SAME inference pipeline as inference.py (process_frame / analyze_video_file).
No custom thresholds — all classification comes directly from the engine.

Endpoints
    GET  /              → Frontend
    POST /predict       → Real-time frame (same as webcam mode)
    POST /predict_video → Upload video (same as analyze_video_file + Grad-CAM)
    GET  /health        → Status
"""

import base64
import io
import os
import sys
import uuid
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── Project root on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    DeepfakeInferenceEngine,
    crop_face,
    preprocess_face,
    face_quality_ok,
    N_FRAMES,
    FAKE_THRESHOLD,
)
from models import GradCAM
from webapp.explainability import ExplainabilityAnalyzer

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
FACE_DETECTOR = str(PROJECT_ROOT / "face_detection_short_range.tflite")
FACE_LANDMARKER = str(PROJECT_ROOT / "face_landmarker.task")
UPLOAD_DIR = Path(__file__).resolve().parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 200
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "webm", "mkv"}
MAX_FRAMES_UPLOAD = 500          # cap to keep analysis under ~60 s
MAX_GRADCAM_FRAMES = 6           # heatmap frames per category

# ============================================================================
# App
# ============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

engine: DeepfakeInferenceEngine = None
engine_lock = threading.Lock()
explainer = ExplainabilityAnalyzer()
model_loaded = False
model_load_error = None


def load_model():
    global engine, model_loaded, model_load_error
    try:
        print(f"[SusFace] Loading model from {MODEL_PATH} ...")
        engine = DeepfakeInferenceEngine(
            model_path=MODEL_PATH,
            source="webapp",
            gradcam=False,
            face_detector_model=FACE_DETECTOR,
            face_landmarker_model=FACE_LANDMARKER,
        )
        model_loaded = True
        print("[SusFace] Model loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        model_loaded = False
        print(f"[SusFace] ERROR loading model: {e}")


# ============================================================================
# Helpers
# ============================================================================


def encode_image_base64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def generate_heatmap_overlay(face_bgr, cam_map, alpha=0.45):
    h, w = face_bgr.shape[:2]
    cam = cv2.resize(np.clip(cam_map, 0, 1), (w, h))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(face_bgr, 1 - alpha, heatmap, alpha, 0)


# ============================================================================
# Routes
# ============================================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "error": model_load_error,
        "device": str(engine.device) if engine else "none",
        "n_frames_required": N_FRAMES,
        "fake_threshold": float(FAKE_THRESHOLD),
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict  — real-time single frame  (identical to webcam process_frame)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    if "frame" not in request.files:
        return jsonify({"error": "No 'frame' field in request"}), 400

    nparr = np.frombuffer(request.files["frame"].read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    with engine_lock:
        result = engine.process_frame(frame)        # exact same call as webcam
        buffer_count = len(engine.buf.buffer)
        buffer_ready = engine.buf.ready()

    # prediction_label comes from PredictionStabilizer (trained threshold)
    status = result.prediction_label                 # "FAKE" or "REAL"
    if result.analysis_status == "SUSPICIOUS_NO_FACE":
        status = "SUSPICIOUS"

    return jsonify({
        "prediction": status,
        "confidence": round(result.smooth_prob, 4),
        "raw_prob": round(result.raw_prob, 4),
        "smooth_prob": round(result.smooth_prob, 4),
        "ear": round(result.ear, 5),
        "ear_var": round(result.ear_var, 8),
        "face_state": result.face_state,
        "sharpness": round(result.sharpness, 2),
        "latency_ms": round(result.latency_ms, 2),
        "frame_id": result.frame_id,
        "analysis_status": result.analysis_status,
        "buffer_count": buffer_count,
        "buffer_ready": buffer_ready,
        "buffer_required": N_FRAMES,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict_video  — upload analysis  (mirrors analyze_video_file logic)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict_video", methods=["POST"])
def predict_video():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    if "video" not in request.files:
        return jsonify({"error": "No 'video' field in request"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported: .{ext}"}), 400

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"

    try:
        file.save(str(temp_path))
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_in_file / fps if fps > 0 else 0

        # ── Process frames exactly like analyze_video_file ──
        # Process every frame sequentially (no skipping) so the temporal
        # buffer fills naturally — same as inference.py webcam mode.
        with engine_lock:
            engine.reset_state()
            engine.frame_id = 0
        explainer.reset()

        frame_results = []   # FramePrediction objects
        explain_results = []
        frame_data = []      # (frame_idx, face_crop, raw_frame) for Grad-CAM
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with engine_lock:
                result = engine.process_frame(frame)  # same as webcam

            # Explainability (per-frame)
            face_crop = None
            landmarks = None
            h, w = frame.shape[:2]

            if result.bbox:
                face_crop = crop_face(frame, result.bbox)

            with engine_lock:
                if engine.tracker.last_observation:
                    landmarks = engine.tracker.last_observation.landmarks

            explain = explainer.analyze_frame(
                face_bgr=face_crop,
                p_fake=result.smooth_prob,
                ear=result.ear,
                ear_var=result.ear_var,
                landmarks=landmarks,
                img_w=w, img_h=h,
            )
            explain["frame_index"] = frame_idx
            explain["timestamp"] = round(frame_idx / fps, 2) if fps > 0 else 0

            frame_results.append(result)
            explain_results.append(explain)
            frame_data.append((frame_idx, face_crop, frame))

            frame_idx += 1
            if frame_idx >= MAX_FRAMES_UPLOAD:
                break

        cap.release()

        if not frame_results:
            return jsonify({
                "prediction": "UNKNOWN", "confidence": 0.0,
                "frames_processed": 0, "real_frames": 0, "fake_frames": 0,
                "summary": "No readable frames found.",
                "reasons": [], "suspicious_frame_images": [],
                "real_frame_images": [],
            })

        # ── Classification: MAJORITY VOTING — same as analyze_video_file ──
        fake_count = sum(1 for r in frame_results if r.prediction_label == "FAKE")
        real_count = sum(1 for r in frame_results if r.prediction_label == "REAL")
        total_processed = len(frame_results)

        # Majority vote (same logic as inference.py line 600)
        final_label = "FAKE" if fake_count >= max(1, total_processed // 2) else "REAL"
        avg_prob = float(np.mean([r.smooth_prob for r in frame_results]))

        # ── Explainability aggregation ──
        from inference import FAKE_THRESHOLD as current_threshold
        video_analysis = explainer.analyze_video_results(
            explain_results, fake_threshold=current_threshold
        )

        # Override the explainability label with our majority-vote label
        # (inference.py uses majority voting, not average probability)
        video_analysis["overall_label"] = final_label
        video_analysis["overall_confidence"] = round(avg_prob, 4)

        # ── Grad-CAM for suspicious + real frames ──
        indexed_probs = [(i, frame_results[i].smooth_prob) for i in range(total_processed)]

        sorted_desc = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
        top_sus = [i for i, p in sorted_desc[:MAX_GRADCAM_FRAMES] if p > 0.35]

        sorted_asc = sorted(indexed_probs, key=lambda x: x[1])
        top_real = [i for i, p in sorted_asc[:MAX_GRADCAM_FRAMES] if p <= 0.4]

        suspicious_images = []
        real_images = []

        def _gradcam_batch(indices, out_list):
            for si in indices:
                fidx, fc, raw = frame_data[si]
                if fc is None or fc.size == 0:
                    continue
                try:
                    ft = preprocess_face(fc, engine.device)
                    seq = ft.squeeze(0).unsqueeze(0).repeat(1, N_FRAMES, 1, 1, 1)
                    cam = grad_cam.generate(ft, seq)
                    out_list.append({
                        "frame_index": fidx,
                        "timestamp": round(fidx / fps, 2) if fps > 0 else 0,
                        "p_fake": round(frame_results[si].smooth_prob, 4),
                        "label": frame_results[si].prediction_label,
                        "original": encode_image_base64(fc),
                        "heatmap": encode_image_base64(generate_heatmap_overlay(fc, cam)),
                    })
                except Exception as e:
                    print(f"[SusFace] Grad-CAM error frame {fidx}: {e}")

        if top_sus or top_real:
            with engine_lock:
                grad_cam = GradCAM(engine.model)
                _gradcam_batch(top_sus, suspicious_images)
                _gradcam_batch(top_real, real_images)
                grad_cam.remove_hooks()

        return jsonify({
            "prediction": final_label,
            "confidence": round(avg_prob, 4),
            "frames_processed": total_processed,
            "total_frames": total_frames_in_file,
            "duration": round(duration, 2),
            "fake_frames": fake_count,
            "real_frames": real_count,
            "fake_threshold": round(current_threshold, 4),
            "reasons": video_analysis["reasons"],
            "summary": video_analysis["summary"],
            "avg_spectral": video_analysis["avg_spectral"],
            "avg_texture": video_analysis["avg_texture"],
            "avg_spatial": video_analysis["avg_spatial"],
            "blink_anomaly_pct": video_analysis["blink_anomaly_pct"],
            "top_fake_frames": video_analysis["top_fake_frames"],
            "suspicious_frame_images": suspicious_images,
            "real_frame_images": real_images,
        })

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================================
if __name__ == "__main__":
    load_model()
    print("[SusFace] Starting Flask server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
