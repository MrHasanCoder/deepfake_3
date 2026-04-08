"""
optimize.py
-----------
Export DeepfakeDetector to ONNX and (optionally) TensorRT.

Step 1: PyTorch → ONNX (cross-platform, works on CPU too)
Step 2: ONNX → TensorRT FP16 engine (RTX 3050 GPU required)
Step 3: Benchmark latency comparison

Usage:
  # ONNX only
  python optimize.py --model_path checkpoints/best_model.pth --onnx

  # ONNX + TensorRT
  python optimize.py --model_path checkpoints/best_model.pth --onnx --trt
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from models import DeepfakeDetector

# ════════════════════════════════════════════════════
# ONNX Export
# ════════════════════════════════════════════════════

class DetectorWrapper(nn.Module):
    """
    Wrapper to present a single-input interface for ONNX export.
    ONNX doesn't handle tuple inputs cleanly without a wrapper.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, single_frame, frame_seq):
        logit = self.model(single_frame, frame_seq)
        return torch.sigmoid(logit)


def export_onnx(model_path: str, onnx_path: str = "deepfake_detector.onnx",
                batch_size: int = 1, n_frames: int = 5, img_size: int = 256):
    print("\n[ONNX Export] Starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector(n_segment=n_frames).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = DetectorWrapper(model).to(device)

    # Dummy inputs
    sf  = torch.randn(batch_size, 3, img_size, img_size, device=device)
    fs  = torch.randn(batch_size, n_frames, 3, img_size, img_size, device=device)

    torch.onnx.export(
        wrapper,
        (sf, fs),
        onnx_path,
        opset_version=17,
        input_names=["single_frame", "frame_seq"],
        output_names=["fake_prob"],
        dynamic_axes={
            "single_frame": {0: "batch_size"},
            "frame_seq":    {0: "batch_size"},
            "fake_prob":    {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"[ONNX Export] Saved: {onnx_path}")

    # Validate
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("[ONNX Export] Model validated ✓")
    except ImportError:
        print("[ONNX Export] Install 'onnx' to validate: pip install onnx")

    return onnx_path


# ════════════════════════════════════════════════════
# TensorRT Engine Build
# ════════════════════════════════════════════════════

def build_tensorrt(onnx_path: str, engine_path: str = "deepfake_detector.engine",
                   fp16: bool = True, workspace_gb: int = 2):
    """
    Builds TensorRT FP16 engine from ONNX.
    Requires: tensorrt, cuda-python
    Install:  pip install tensorrt cuda-python
    """
    print("\n[TensorRT] Building engine...")
    try:
        import tensorrt as trt
    except ImportError:
        print("[TensorRT] tensorrt not installed.")
        print("  Install: pip install tensorrt")
        print("  Or use NVIDIA TensorRT container for RTX support.")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser     = trt.OnnxParser(network, TRT_LOGGER)
    config     = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TensorRT] FP16 mode enabled ✓")
    else:
        print("[TensorRT] FP16 not available — using FP32")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TensorRT] Parse error: {parser.get_error(i)}")
            return None

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[TensorRT] Engine build failed.")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized)

    print(f"[TensorRT] Engine saved: {engine_path}")
    return engine_path


# ════════════════════════════════════════════════════
# TensorRT Inference Helper
# ════════════════════════════════════════════════════

class TRTInference:
    """
    Loads a TensorRT engine and runs inference.
    Use this in inference.py instead of the PyTorch model for max speed.
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
            import cuda
        except ImportError:
            raise ImportError("Install: pip install tensorrt cuda-python")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded from {engine_path}")

    def infer(self, single_frame_np: np.ndarray,
              frame_seq_np: np.ndarray) -> float:
        """
        Inputs:
            single_frame_np : (1,3,256,256) float32 numpy
            frame_seq_np    : (1,5,3,256,256) float32 numpy
        Returns fake probability (float).
        """
        import tensorrt as trt
        import ctypes

        bindings = []
        outputs  = []

        for binding in self.engine:
            size  = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            mem   = torch.zeros(size, dtype=torch.float32).cuda()

            if self.engine.binding_is_input(binding):
                name = self.engine.get_binding_name(binding)
                if "single" in name:
                    arr = torch.from_numpy(single_frame_np).cuda()
                else:
                    arr = torch.from_numpy(frame_seq_np).cuda()
                mem.copy_(arr.flatten())
            else:
                outputs.append(mem)

            bindings.append(int(mem.data_ptr()))

        self.context.execute_v2(bindings=bindings)
        return torch.sigmoid(outputs[0]).item()


# ════════════════════════════════════════════════════
# Latency Benchmark
# ════════════════════════════════════════════════════

def benchmark(model_path: str, onnx_path: str, n_runs: int = 100):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector(n_segment=5).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sf = torch.randn(1, 3, 256, 256, device=device)
    fs = torch.randn(1, 5, 3, 256, 256, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad(), torch.cuda.amp.autocast():
            model(sf, fs)

    # PyTorch latency
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad(), torch.cuda.amp.autocast():
            model(sf, fs)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000 / n_runs

    print(f"\n{'─'*40}")
    print(f"  PyTorch (AMP FP16) : {pt_ms:.2f} ms/frame")

    # ONNX Runtime latency
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, providers=providers)

        sf_np = sf.cpu().numpy()
        fs_np = fs.cpu().numpy()

        # Warmup
        for _ in range(10):
            sess.run(None, {"single_frame": sf_np, "frame_seq": fs_np})

        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {"single_frame": sf_np, "frame_seq": fs_np})
        ort_ms = (time.perf_counter() - t0) * 1000 / n_runs
        print(f"  ONNX Runtime (CUDA): {ort_ms:.2f} ms/frame")
    except ImportError:
        print("  ONNX Runtime: not installed (pip install onnxruntime-gpu)")

    print(f"  Target             : <40 ms/frame")
    print(f"{'─'*40}\n")


# ════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--onnx_path",  default="deepfake_detector.onnx")
    p.add_argument("--engine_path",default="deepfake_detector.engine")
    p.add_argument("--onnx",       action="store_true", help="Export to ONNX")
    p.add_argument("--trt",        action="store_true", help="Build TensorRT engine")
    p.add_argument("--benchmark",  action="store_true", help="Latency benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.onnx:
        export_onnx(args.model_path, args.onnx_path)

    if args.trt:
        if not os.path.exists(args.onnx_path):
            export_onnx(args.model_path, args.onnx_path)
        build_tensorrt(args.onnx_path, args.engine_path)

    if args.benchmark:
        benchmark(args.model_path, args.onnx_path)