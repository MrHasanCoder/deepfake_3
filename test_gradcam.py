# paste this into a new file: test_gradcam.py
import torch
from models import DeepfakeDetector, GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepfakeDetector(n_segment=5).to(DEVICE)
ckpt  = torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

grad_cam = GradCAM(model)  # will print which layer it targets

sf  = torch.randn(1, 3, 256, 256, device=DEVICE)
fs  = torch.randn(1, 5, 3, 256, 256, device=DEVICE)

cam = grad_cam.generate(sf, fs)
print(f"CAM shape : {cam.shape}")
print(f"CAM max   : {cam.max():.4f}")
print(f"CAM mean  : {cam.mean():.4f}")

if cam.max() > 0.01:
    print("✓ GradCAM++ working correctly")
else:
    print("✗ Still zero — deeper issue")

grad_cam.remove_hooks()