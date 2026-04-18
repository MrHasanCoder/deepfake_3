"""
explainability.py
-----------------
Rule-based explainability module for deepfake detection.
Analyzes frequency, texture, spatial, and blink features to generate
human-readable reasons for detection decisions.

No extra dependencies — uses numpy, scipy, and cv2 only.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Thresholds — tuned for the DeepShield pipeline
# ============================================================================

SPECTRAL_THRESHOLD = 0.35       # FFT high-freq energy ratio
TEXTURE_THRESHOLD = 80.0        # Laplacian variance deviation
SPATIAL_THRESHOLD = 0.15        # Landmark geometry ratio deviation
BLINK_VARIANCE_LOW = 0.0005     # Too little EAR variance → suspicious
BLINK_VARIANCE_HIGH = 0.015     # Too much EAR variance → suspicious


class ExplainabilityAnalyzer:
    """
    Rule-based explainability engine.
    Produces human-readable reasons alongside the model's prediction.
    """

    def __init__(self):
        self._baseline_texture = None
        self._texture_samples: list[float] = []

    # ── Per-frame analysis ──────────────────────────────────────────

    def analyze_frame(
        self,
        face_bgr: Optional[np.ndarray],
        p_fake: float,
        ear: float,
        ear_var: float,
        landmarks: Optional[list],
        img_w: int = 0,
        img_h: int = 0,
    ) -> Dict:
        """
        Run all sub-analyzers on a single frame.
        Returns a dict with individual scores + generated reasons.
        """
        scores = {
            "p_fake": p_fake,
            "spectral_score": 0.0,
            "texture_score": 0.0,
            "spatial_score": 0.0,
            "ear": ear,
            "ear_var": ear_var,
            "blink_anomaly": False,
        }

        if face_bgr is not None and face_bgr.size > 0:
            scores["spectral_score"] = self.compute_spectral_score(face_bgr)
            scores["texture_score"] = self.compute_texture_score(face_bgr)

        if landmarks and img_w > 0 and img_h > 0:
            scores["spatial_score"] = self.compute_spatial_score(
                landmarks, img_w, img_h
            )

        scores["blink_anomaly"] = self._check_blink_anomaly(ear_var)
        scores["reasons"] = self.generate_reasons(scores)

        return scores

    # ── Sub-analyzers ───────────────────────────────────────────────

    @staticmethod
    def compute_spectral_score(face_bgr: np.ndarray) -> float:
        """
        FFT-based spectral anomaly score.
        High-frequency energy ratio — GANs often introduce periodic
        artifacts that show up as excess high-frequency energy.
        Returns a score in [0, 1]; higher = more anomalous.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128)).astype(np.float32)

        # 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Radial mask: high-freq = outside a circle of radius r
        r = min(h, w) // 4
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        total_energy = magnitude.sum() + 1e-8
        high_freq_energy = magnitude[dist > r].sum()
        ratio = float(high_freq_energy / total_energy)

        return min(ratio, 1.0)

    def compute_texture_score(self, face_bgr: np.ndarray) -> float:
        """
        Laplacian variance as a texture inconsistency metric.
        Returns the deviation from a running baseline (higher = more anomalous).
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Build baseline from first 30 frames
        self._texture_samples.append(lap_var)
        if len(self._texture_samples) >= 10 and self._baseline_texture is None:
            self._baseline_texture = np.mean(self._texture_samples[:30])

        if self._baseline_texture is not None and self._baseline_texture > 0:
            deviation = abs(lap_var - self._baseline_texture) / self._baseline_texture
            return float(deviation) * 100.0  # scale for threshold comparison
        return 0.0

    @staticmethod
    def compute_spatial_score(
        landmarks: list, img_w: int, img_h: int
    ) -> float:
        """
        Facial geometry consistency check using landmark ratios.
        Measures deviation in key facial proportion ratios.
        Returns a score where higher = more anomalous.
        """
        try:
            def _pt(idx: int) -> Tuple[float, float]:
                lm = landmarks[idx]
                return (lm.x * img_w, lm.y * img_h)

            # Key landmarks (MediaPipe FaceMesh indices)
            # Left eye outer: 33, Right eye outer: 263
            # Nose tip: 1, Chin: 152
            # Left mouth: 61, Right mouth: 291

            left_eye = np.array(_pt(33))
            right_eye = np.array(_pt(263))
            nose = np.array(_pt(1))
            chin = np.array(_pt(152))
            mouth_l = np.array(_pt(61))
            mouth_r = np.array(_pt(291))

            # Inter-eye distance
            eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6

            # Nose-chin distance relative to eye distance
            nose_chin = np.linalg.norm(chin - nose)
            ratio1 = nose_chin / eye_dist

            # Mouth width relative to eye distance
            mouth_width = np.linalg.norm(mouth_r - mouth_l)
            ratio2 = mouth_width / eye_dist

            # Expected ratios for a typical face
            expected_r1 = 0.85  # nose-chin / eye distance
            expected_r2 = 0.55  # mouth width / eye distance

            dev1 = abs(ratio1 - expected_r1) / expected_r1
            dev2 = abs(ratio2 - expected_r2) / expected_r2

            return float((dev1 + dev2) / 2.0)

        except (IndexError, AttributeError):
            return 0.0

    @staticmethod
    def _check_blink_anomaly(ear_var: float) -> bool:
        """Check if EAR variance is abnormally low or high."""
        if ear_var < 1e-8:  # no data yet
            return False
        return ear_var < BLINK_VARIANCE_LOW or ear_var > BLINK_VARIANCE_HIGH

    # ── Reason generation ───────────────────────────────────────────

    @staticmethod
    def generate_reasons(scores: Dict) -> List[str]:
        """
        Rule-based reason generation from computed scores.
        Returns a list of human-readable explanation strings.
        """
        reasons = []

        if scores.get("spectral_score", 0) > SPECTRAL_THRESHOLD:
            reasons.append(
                "Frequency artifacts detected in facial region — "
                "unnatural periodic patterns found in the frequency spectrum"
            )

        if scores.get("texture_score", 0) > TEXTURE_THRESHOLD:
            reasons.append(
                "Facial texture inconsistencies found — "
                "the texture sharpness deviates significantly from baseline"
            )

        if scores.get("spatial_score", 0) > SPATIAL_THRESHOLD:
            reasons.append(
                "Facial structure geometry mismatch — "
                "key facial proportions deviate from expected ratios"
            )

        if scores.get("blink_anomaly", False):
            ear_var = scores.get("ear_var", 0)
            if ear_var < BLINK_VARIANCE_LOW:
                reasons.append(
                    "Unnatural eye behavior detected — "
                    "abnormally low blink activity suggests synthetic face"
                )
            else:
                reasons.append(
                    "Unnatural eye behavior detected — "
                    "erratic blink patterns inconsistent with natural behavior"
                )

        if scores.get("p_fake", 0) > 0.45 and not reasons:
            reasons.append(
                "Neural network detected subtle manipulation artifacts "
                "across multiple analysis branches"
            )

        return reasons

    @staticmethod
    def generate_summary(reasons: List[str], overall_label: str) -> str:
        """
        Generate a one-line summary sentence.
        """
        if overall_label == "REAL":
            return (
                "This content appears authentic — no significant manipulation "
                "indicators were detected across frequency, texture, spatial, "
                "and temporal analysis."
            )

        if not reasons:
            return f"This content is likely {overall_label.lower()} based on neural network analysis."

        # Take top 2 reasons, shortened
        short_reasons = []
        for r in reasons[:3]:
            # Use the part before the dash
            short = r.split("—")[0].strip().rstrip()
            short_reasons.append(short.lower())

        joined = " and ".join(short_reasons)
        return f"This content is likely {overall_label.lower()} due to {joined}."

    # ── Aggregate video analysis ────────────────────────────────────

    def analyze_video_results(
        self, frame_results: List[Dict], fake_threshold: float = 0.45
    ) -> Dict:
        """
        Aggregate per-frame explainability results into a video-level summary.
        fake_threshold: the engine's trained threshold for FAKE classification.
        """
        if not frame_results:
            return {
                "reasons": [],
                "summary": "No frames were analyzed.",
                "overall_label": "UNKNOWN",
                "overall_confidence": 0.0,
                "avg_spectral": 0.0,
                "avg_texture": 0.0,
                "avg_spatial": 0.0,
                "blink_anomaly_pct": 0.0,
                "top_fake_frames": [],
            }

        # Collect averages
        spectral_scores = [r.get("spectral_score", 0) for r in frame_results]
        texture_scores = [r.get("texture_score", 0) for r in frame_results]
        spatial_scores = [r.get("spatial_score", 0) for r in frame_results]
        blink_anomalies = [1 if r.get("blink_anomaly", False) else 0 for r in frame_results]
        p_fakes = [r.get("p_fake", 0) for r in frame_results]

        avg_spectral = float(np.mean(spectral_scores))
        avg_texture = float(np.mean(texture_scores))
        avg_spatial = float(np.mean(spatial_scores))
        blink_pct = float(np.mean(blink_anomalies)) * 100

        # Aggregate reasons from averages
        agg_scores = {
            "spectral_score": avg_spectral,
            "texture_score": avg_texture,
            "spatial_score": avg_spatial,
            "blink_anomaly": blink_pct > 40,
            "ear_var": BLINK_VARIANCE_LOW if blink_pct > 40 else BLINK_VARIANCE_LOW + 0.001,
            "p_fake": float(np.mean(p_fakes)),
        }
        reasons = ExplainabilityAnalyzer.generate_reasons(agg_scores)

        # Determine overall label using the model's trained threshold
        avg_p_fake = float(np.mean(p_fakes))
        if avg_p_fake > fake_threshold:
            overall_label = "FAKE"
        else:
            overall_label = "REAL"

        summary = ExplainabilityAnalyzer.generate_summary(reasons, overall_label)

        # Find top fake frames (by p_fake)
        indexed = [(i, r.get("p_fake", 0)) for i, r in enumerate(frame_results)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_fake = [
            {"frame_index": i, "p_fake": round(p, 4)}
            for i, p in indexed[:10]
            if p > fake_threshold * 0.7  # slightly below threshold
        ]

        return {
            "overall_label": overall_label,
            "overall_confidence": round(avg_p_fake, 4),
            "reasons": reasons,
            "summary": summary,
            "avg_spectral": round(avg_spectral, 4),
            "avg_texture": round(avg_texture, 4),
            "avg_spatial": round(avg_spatial, 4),
            "blink_anomaly_pct": round(blink_pct, 1),
            "top_fake_frames": top_fake,
        }

    def reset(self):
        """Reset baseline state for a new video."""
        self._baseline_texture = None
        self._texture_samples = []
