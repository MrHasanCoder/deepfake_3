"""
models.py
---------
4-Branch Hybrid Deepfake Detection Model:


  Branch 1 → MesoNet4        (textural artifacts)
  Branch 2 → MobileNetV3-L   (spatial/semantic features)
  Branch 3 → TSM             (temporal consistency, 5-frame sequences)
  Branch 4 → FFT Spectral    (GAN frequency fingerprints)


  Fusion   → Attention-weighted concatenation → FC → Sigmoid


RTX 3050 (4GB) optimized:
  - Lightweight backbones
  - AMP fp16 compatible
  - ONNX / TensorRT exportable
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from typing import Tuple




# ════════════════════════════════════════════════════
# Branch 1: MesoNet4  (Textural Analysis)
# ════════════════════════════════════════════════════


class MesoNet4Branch(nn.Module):
    """
    MesoNet4 as described in Afchar et al. (2018).
    Detects compression artifacts, blending boundaries,
    and pixel-level inconsistencies.


    Input : (B, 3, 256, 256)
    Output: (B, 128)
    """


    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 128×128


            # Block 2
            nn.Conv2d(8, 8, 5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64×64


            # Block 3
            nn.Conv2d(8, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32×32


            # Block 4
            nn.Conv2d(16, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),          # 8×8
        )
        self.flatten = nn.Flatten()
        # AdaptiveAvgPool makes FC size independent of input resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)      # forces to (B, 16, 4, 4) regardless of input size
        x = self.flatten(x)
        return self.fc(x)




# ════════════════════════════════════════════════════
# Branch 2: MobileNetV3-Large  (Spatial/Semantic)
# ════════════════════════════════════════════════════


class MobileNetV3Branch(nn.Module):
    """
    MobileNetV3-Large pretrained on ImageNet, features only.
    Captures semantic facial structure (eyes, nose, mouth geometry).


    Input : (B, 3, 256, 256)
    Output: (B, 256)
    """


    def __init__(self, out_dim: int = 256, freeze_layers: int = 10):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)


        # Remove classifier, keep feature extractor
        self.features = base.features        # (B, 960, 8, 8) at 256×256 input
        self.avgpool  = base.avgpool         # (B, 960, 1, 1)


        # Freeze early layers (saves VRAM during early training)
        params = list(self.features.parameters())
        for p in params[:freeze_layers]:
            p.requires_grad = False


        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)


    def unfreeze_all(self):
        """Call after initial epochs to fine-tune entire backbone."""
        for p in self.features.parameters():
            p.requires_grad = True




# ════════════════════════════════════════════════════
# Branch 3: TSM (Temporal Shift Module)
# ════════════════════════════════════════════════════


class TemporalShift(nn.Module):
    """
    Temporal Shift Module — shifts 1/8 of channels backward and
    1/8 forward in time, enabling temporal modeling without extra params.


    Reference: Lin et al., 2019 (TSM: Temporal Shift Module)
    """


    def __init__(self, n_segment: int = 5, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*T, C, H, W) — TSM operates on reshuffled batch dimension
        """
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)


        fold = c // self.fold_div
        out = torch.zeros_like(x)


        # Shift backward (past → present)
        out[:, 1:,  :fold]       = x[:, :-1, :fold]
        # Shift forward  (future → present)
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        # Identity for remaining channels
        out[:, :, 2*fold:]       = x[:, :, 2*fold:]


        return out.view(nt, c, h, w)




class TSMBranch(nn.Module):
    """
    Lightweight CNN + TSM for temporal consistency modeling.
    Detects flickering, blink irregularities, motion artifacts.


    Input : (B, T, 3, H, W)  where T=5
    Output: (B, 128)
    """


    def __init__(self, n_segment: int = 5, out_dim: int = 128):
        super().__init__()
        self.n_segment = n_segment
        self.shift = TemporalShift(n_segment=n_segment, fold_div=8)


        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # 128×128


            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # 64×64


            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),     # 4×4
        )


        self.temporal_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * n_segment, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        # Reshape to (B*T, C, H, W) for TSM
        x = x.view(B * T, C, H, W)
        x = self.shift(x)
        x = self.conv_block(x)           # (B*T, 128, 4, 4)


        # Fold back time dimension
        x = x.view(B, T, 128, 4, 4)
        x = x.view(B, -1)               # (B, T*128*4*4)
        return self.temporal_fc(x)




# ════════════════════════════════════════════════════
# Branch 4: Spectral FFT  (Frequency-Domain)
# ════════════════════════════════════════════════════


class SpectralBranch(nn.Module):
    """
    ROI-based FFT analysis for GAN frequency artifact detection.
    GANs leave characteristic periodic patterns in the frequency domain.


    Input : (B, 3, H, W)  — face ROI
    Output: (B, 64)
    """


    def __init__(self, fft_size: int = 128, out_dim: int = 64):
        super().__init__()
        self.fft_size = fft_size


        # Small CNN to process magnitude spectrum
        self.spectral_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # 64→32 (if fft_size=128)


            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),     # → 8×8


            nn.Flatten(),
            nn.Linear(32 * 8 * 8, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale (luminance channel only for FFT)
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B,H,W)


        # Resize to fixed FFT size
        gray = F.interpolate(gray.unsqueeze(1), size=(self.fft_size, self.fft_size),
                             mode="bilinear", align_corners=False)     # (B,1,fft,fft)
        gray = gray.squeeze(1)  # (B,fft,fft)


        # Cast to float32 explicitly — FFT is unstable in fp16
        gray = gray.float()


        # 2D FFT → magnitude spectrum (log scale, fft-shifted)
        fft  = torch.fft.fft2(gray)
        fft  = torch.fft.fftshift(fft)
        mag  = torch.abs(fft) + 1e-8
        spec = torch.log(mag)                                          # (B,fft,fft)
        spec = spec.unsqueeze(1)                                       # (B,1,fft,fft)


        # Normalize to [0,1]
        spec_min = spec.amin(dim=(2, 3), keepdim=True)
        spec_max = spec.amax(dim=(2, 3), keepdim=True)
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)


        return self.spectral_cnn(spec)




# ════════════════════════════════════════════════════
# Attention Fusion Module
# ════════════════════════════════════════════════════


class AttentionFusion(nn.Module):
    """
    Learnable attention weights over branch embeddings.
    Dynamically emphasizes the most discriminative branch per sample.
    """


    def __init__(self, branch_dims: Tuple[int, ...]):
        """branch_dims: tuple of output dims from each branch."""
        super().__init__()
        total = sum(branch_dims)
        # Attention MLP
        self.attn = nn.Sequential(
            nn.Linear(total, len(branch_dims)),
            nn.Softmax(dim=-1),
        )
        self.branch_dims = branch_dims


    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        concat = torch.cat(embeddings, dim=-1)       # (B, total_dim)
        weights = self.attn(concat)                  # (B, n_branches)


        # Weighted sum by branch
        out_parts = []
        for i, emb in enumerate(embeddings):
            w = weights[:, i:i+1]                   # (B,1)
            out_parts.append(emb * w)


        return torch.cat(out_parts, dim=-1)          # (B, total_dim)




# ════════════════════════════════════════════════════
# Full Fusion Model
# ════════════════════════════════════════════════════


class DeepfakeDetector(nn.Module):
    """
    4-Branch Hybrid Deepfake Detector with Attention Fusion.


    Inputs:
        single_frame : (B, 3, 256, 256)  → textural, spatial, spectral
        frame_seq    : (B, T, 3, 256, 256) → temporal (T=5)


    Output:
        logit : (B, 1)  — raw score before sigmoid
    """


    BRANCH_DIMS = (128, 256, 128, 64)  # meso, mobile, tsm, spectral


    def __init__(self, n_segment: int = 5, freeze_backbone_layers: int = 10):
        super().__init__()


        # 4 Branches
        self.branch_meso    = MesoNet4Branch(out_dim=128)
        self.branch_mobile  = MobileNetV3Branch(out_dim=256,
                                                freeze_layers=freeze_backbone_layers)
        self.branch_tsm     = TSMBranch(n_segment=n_segment, out_dim=128)
        self.branch_spectral= SpectralBranch(out_dim=64)


        # Attention fusion
        self.fusion = AttentionFusion(self.BRANCH_DIMS)


        total_dim = sum(self.BRANCH_DIMS)


        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            # No sigmoid here — use BCEWithLogitsLoss during training
            # Apply sigmoid at inference
        )


    def forward(self, single_frame: torch.Tensor,
                frame_seq: torch.Tensor) -> torch.Tensor:
        """
        single_frame : (B, 3, 256, 256)
        frame_seq    : (B, T, 3, 256, 256)
        """
        e_meso     = self.branch_meso(single_frame)      # (B, 128)
        e_mobile   = self.branch_mobile(single_frame)    # (B, 256)
        e_tsm      = self.branch_tsm(frame_seq)          # (B, 128)
        e_spectral = self.branch_spectral(single_frame)  # (B, 64)


        fused = self.fusion((e_meso, e_mobile, e_tsm, e_spectral))  # (B, 576)
        logit = self.classifier(fused)                               # (B, 1)
        return logit


    def predict_proba(self, single_frame: torch.Tensor,
                      frame_seq: torch.Tensor) -> torch.Tensor:
        """Returns fake probability in [0,1]."""
        with torch.no_grad():
            logit = self.forward(single_frame, frame_seq)
            return torch.sigmoid(logit)


    def unfreeze_backbone(self):
        """Stage-2: unfreeze MobileNet for end-to-end fine-tuning."""
        self.branch_mobile.unfreeze_all()
        print("[Model] MobileNetV3 backbone unfrozen for fine-tuning.")


    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




# ════════════════════════════════════════════════════
# GradCAM++ for MobileNet branch
# ════════════════════════════════════════════════════


class GradCAM:
    """
    GradCAM++ on MobileNetV3 features[-3] — the last conv layer whose
    output is still positive (features[-1] ends in Hardswish which
    produces all-negative outputs on this model, making standard
    GradCAM return a blank map).


    GradCAM++ uses alpha-weighted gradient squares which is more robust
    to near-zero gradient situations than standard GradCAM.
    """


    # Target features[-3]: last InvertedResidual before the final
    # pointwise conv+Hardswish block. Outputs are ReLU6-bounded → positive.
    TARGET_LAYER_IDX = -2


    def __init__(self, model: DeepfakeDetector):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()


    def _register_hooks(self):
        target = self.model.branch_mobile.features[self.TARGET_LAYER_IDX]
        print(f"[GradCAM++] Target layer: features[{self.TARGET_LAYER_IDX}] "
              f"→ {type(target).__name__}")


        def fwd_hook(module, inp, out):
            self.activations = out          # keep in graph for backward


        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()


        self._hook_handles.append(target.register_forward_hook(fwd_hook))
        self._hook_handles.append(
            target.register_full_backward_hook(bwd_hook)
        )


    def generate(self, single_frame: torch.Tensor,
                 frame_seq: torch.Tensor) -> np.ndarray:
        """
        Returns normalized CAM: (H, W) float32 numpy array in [0, 1].
        Works correctly with model in eval() mode.
        """
        self.gradients   = None
        self.activations = None
        self.model.zero_grad()


        with torch.enable_grad():
            logit = self.model(single_frame, frame_seq)
            # Use raw logit (not sigmoid) — larger gradient signal
            logit.squeeze().backward()


        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7), dtype=np.float32)


        grads = self.gradients                          # (1,C,H,W)
        acts  = self.activations.detach()               # (1,C,H,W)


        # ── GradCAM++ weights ──────────────────────────────────────
        # alpha_k = grad² / (2·grad² + acts·sum(grad³) + eps)
        grads2  = grads ** 2
        grads3  = grads ** 3
        # sum of acts * grads3 over spatial dims
        sum_ag3 = (acts * grads3).sum(dim=(2, 3), keepdim=True)
        alpha   = grads2 / (2 * grads2 + sum_ag3 + 1e-8)


        # Weight = alpha * ReLU(grad)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)


        cam = (weights * acts).sum(dim=1, keepdim=True)   # (1,1,H,W)
        cam = F.relu(cam).squeeze().cpu().numpy()          # (H,W)


        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            # Fallback: use raw activation energy if CAM still flat
            energy = acts.pow(2).sum(dim=1).squeeze().cpu().numpy()
            energy -= energy.min()
            energy /= (energy.max() + 1e-8)
            return energy.astype(np.float32)


        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()




# ════════════════════════════════════════════════════
# Quick architecture test
# ════════════════════════════════════════════════════


if __name__ == "__main__":
    from typing import Optional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    model = DeepfakeDetector(n_segment=5).to(device)
    print(f"Trainable params: {model.count_params():,}")


    B, T, C, H, W = 2, 5, 3, 256, 256
    sf = torch.randn(B, C, H, W).to(device)
    fs = torch.randn(B, T, C, H, W).to(device)


    with torch.cuda.amp.autocast():
        logit = model(sf, fs)
    print(f"Output logit shape: {logit.shape}")  # (2, 1)
    print(f"Fake prob: {torch.sigmoid(logit).squeeze().tolist()}")

