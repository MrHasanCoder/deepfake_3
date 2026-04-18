# SusFace — Real-Time Deepfake Detection Platform

A real-time deepfake detection system using a **4-branch hybrid deep learning model** with attention-weighted fusion. Supports webcam, video file upload, and a web application interface with explainability features (Grad-CAM++ heatmaps and rule-based analysis).

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Download Model Weights](#download-model-weights)
- [Usage](#usage)
  - [Web Application](#1-web-application-recommended)
  - [Webcam Mode (CLI)](#2-webcam-mode-cli)
  - [Video File Analysis (CLI)](#3-video-file-analysis-cli)
  - [Virtual Webcam (Google Meet)](#4-virtual-webcam-google-meet-integration)
- [Training Your Own Model](#training-your-own-model)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Architecture

The model uses **4 specialized branches** that analyze different aspects of a face:

| Branch | Backbone | What It Detects | Output Dim |
|--------|----------|-----------------|------------|
| **Textural** | MesoNet4 | Compression artifacts, blending boundaries | 128 |
| **Spatial** | MobileNetV3-Large | Facial structure anomalies (eyes, nose, mouth) | 256 |
| **Temporal** | TSM (Temporal Shift Module) | Flickering, blink irregularities, motion artifacts | 128 |
| **Spectral** | FFT CNN | GAN frequency fingerprints | 64 |

All branch embeddings are fused via a **learnable attention mechanism** that dynamically weights each branch per sample, followed by a fully connected classifier.

```
Input Face (224×224) ──┬──► MesoNet4 ──────────► 128-d ──┐
                       ├──► MobileNetV3 ────────► 256-d ──┤
                       │                                   ├──► Attention Fusion ──► FC ──► Sigmoid ──► FAKE/REAL
5-Frame Sequence  ─────┴──► TSM Branch ────────► 128-d ──┤
                       └──► FFT Spectral ──────►  64-d ──┘
```

---

## Features

- **Real-time detection** — ~30 FPS on RTX 3050 (4GB VRAM)
- **Multiple input modes** — Webcam, video file, screen share, web upload
- **Explainability** — Grad-CAM++ heatmaps showing where the model focuses
- **Rule-based analysis** — Human-readable explanations (texture, spectral, blink anomalies)
- **Prediction stabilization** — Hysteresis + EMA smoothing to prevent label flipping
- **Face quality gating** — Rejects blurry, too-small, or invalid face crops
- **MediaPipe face tracking** — Landmark-refined face crops with blink detection (EAR)

---

## Prerequisites

- **Python** 3.10 or higher
- **NVIDIA GPU** with CUDA support (recommended, CPU works but is slow)
- **CUDA Toolkit** 12.1+ and cuDNN (for GPU acceleration)
- **Git** installed

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MrHasanCoder/deepfake_3.git
cd deepfake_3
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

Install PyTorch separately **before** other dependencies. Choose your CUDA version:

```bash
# CUDA 12.1 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (slow, not recommended)
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output: `PyTorch 2.x.x, CUDA: True`

---

## Download Model Weights

The trained model weights (`best_model.pth`, ~108 MB) are hosted as a **GitHub Release asset** since they exceed GitHub's file size limit.

### Option A: Download from GitHub Releases (recommended)

1. Go to the [Releases page](https://github.com/MrHasanCoder/deepfake_3/releases)
2. Download `best_model.pth` from the latest release
3. Place it in the `checkpoints/` directory:

```bash
mkdir checkpoints
# Move the downloaded file
mv best_model.pth checkpoints/best_model.pth
```

### Option B: Download via command line

```bash
mkdir checkpoints
curl -L -o checkpoints/best_model.pth https://github.com/MrHasanCoder/deepfake_3/releases/download/v1.0.0/best_model.pth
```

### MediaPipe models (auto-download)

The face detection and landmarker models (`face_detection_short_range.tflite`, `face_landmarker.task`) are **downloaded automatically** on first run from Google's MediaPipe servers. No manual download needed.

---

## Usage

### 1. Web Application (Recommended)

The easiest way to use SusFace. Provides a browser-based UI with live webcam analysis, video upload, and visual explainability.

```bash
python webapp/app.py
```

Then open **http://localhost:5000** in your browser.

**Web App Features:**
- 📷 **Live Webcam** — Real-time detection with confidence scoring
- 🖥️ **Screen Capture** — Analyze screen-shared video calls
- 📁 **Video Upload** — Upload MP4/AVI/MOV/WebM/MKV files for analysis
- 🔥 **Grad-CAM Heatmaps** — Visual attention maps on suspicious frames
- 📊 **Explainability Report** — Detailed breakdown of detection reasons
- 📄 **PDF Export** — Download analysis reports

---

### 2. Webcam Mode (CLI)

Real-time detection with OpenCV display window:

```bash
python inference.py --model_path checkpoints/best_model.pth --source webcam
```

**With Grad-CAM overlay:**

```bash
python inference.py --model_path checkpoints/best_model.pth --source webcam --gradcam
```

Press **`q`** to quit.

---

### 3. Video File Analysis (CLI)

Analyze a pre-recorded video:

```bash
python inference.py --model_path checkpoints/best_model.pth --source path/to/video.mp4
```

**With Grad-CAM:**

```bash
python inference.py --model_path checkpoints/best_model.pth --source path/to/video.mp4 --gradcam
```

Results are logged to `inference_logs/frame_predictions.csv`.

---

### 4. Virtual Webcam (Google Meet Integration)

Feed the detection overlay to a virtual webcam (requires [OBS Virtual Camera](https://obsproject.com/) or similar):

```bash
pip install pyvirtualcam
python inference.py --model_path checkpoints/best_model.pth --source webcam --virtual_cam
```

This creates a virtual camera output that you can select in Google Meet, Zoom, etc.

---

## Training Your Own Model

### 1. Prepare dataset

Organize your dataset under `datasets/` with this structure:

```
datasets/
├── train/
│   ├── real/          # Real face video frames
│   └── fake/          # Fake/deepfake face video frames
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Each subdirectory should contain **video files** (MP4, AVI, etc.) or **image frames**.

### 2. Start training

```bash
python train.py --data_root . --epochs 50 --batch_size 8 --lr 1e-3
```

**Key training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `.` | Root directory containing `datasets/` |
| `--epochs` | `50` | Total training epochs |
| `--batch_size` | `8` | Batch size per step |
| `--accum_steps` | `2` | Gradient accumulation (effective batch = batch_size × accum_steps) |
| `--lr` | `1e-3` | Learning rate |
| `--unfreeze_epoch` | `10` | Epoch to unfreeze MobileNetV3 backbone |
| `--patience` | `8` | Early stopping patience |
| `--resume` | flag | Resume from last checkpoint |

### 3. Resume training

```bash
python train.py --data_root . --resume --epochs 100
```

**Training features:**
- Focal Loss for class imbalance handling
- Cosine annealing LR scheduler
- Staged backbone unfreezing (MobileNetV3)
- Mixed precision support (AMP)
- Early stopping
- Per-batch CSV logging for real-time monitoring

Checkpoints are saved to `checkpoints/` with the best model selected by validation AUC.

---

## Project Structure

```
deepfake_3/
├── models.py                  # 4-branch hybrid model architecture
├── train.py                   # Training pipeline with Focal Loss, AMP, early stopping
├── inference.py               # Inference engine (webcam, video, virtual cam)
├── dataset_loader.py          # Dataset loading and augmentation pipeline
├── optimize.py                # ONNX/TensorRT export utilities
├── evaluate_test_videos.py    # Automated evaluation on test video sets
├── test_evaluation.py         # Test evaluation utilities
├── scan_layers.py             # Model layer inspection tool
├── quick_gradcam_analysis.py  # Standalone Grad-CAM analysis script
├── requirements.txt           # Python dependencies
├── checkpoints/               # Saved model weights (gitignored)
│   ├── best_model.pth         # ← Download from GitHub Releases
│   └── latest.pth
├── webapp/
│   ├── app.py                 # Flask web backend
│   ├── explainability.py      # Rule-based explainability engine
│   ├── templates/
│   │   └── index.html         # Frontend UI
│   └── static/
│       └── jspdf.umd.min.js   # PDF export library
├── datasets/                  # Training data (gitignored)
├── inference_logs/            # Frame-level prediction logs
└── gradcam_outputs/           # Saved Grad-CAM heatmaps
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch 2.x |
| **Face Detection** | MediaPipe Face Detector + Face Landmarker |
| **Backbones** | MesoNet4, MobileNetV3-Large, TSM, FFT CNN |
| **Web Backend** | Flask |
| **Frontend** | HTML/CSS/JavaScript |
| **Explainability** | Grad-CAM++, Rule-based analysis |
| **Optimization** | ONNX, TensorRT (optional) |

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- [MesoNet](https://github.com/DariusAf/MesoNet) — Afchar et al., 2018
- [TSM](https://github.com/mit-han-lab/temporal-shift-module) — Lin et al., 2019
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Howard et al., 2019
- [MediaPipe](https://mediapipe.dev/) — Google
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., 2019
