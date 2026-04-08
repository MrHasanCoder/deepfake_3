"""
scan_layers.py — single pass, captures all layers at once.
Run: python scan_layers.py --model_path checkpoints/best_model.pth
"""
import torch, torch.nn.functional as F, argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="checkpoints/best_model.pth")
    return p.parse_args()
args = parse_args()

from models import DeepfakeDetector
model = DeepfakeDetector(n_segment=5).to(DEVICE)
ckpt  = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

features = model.branch_mobile.features
n = len(features)
print(f"Layers : {n}  — registering all hooks...", flush=True)

store   = {i: {"act": None, "grad": None} for i in range(n)}
handles = []
for idx in range(n):
    def make_fwd(i):
        def hook(m, inp, out): store[i]["act"] = out.detach().clone()
        return hook
    def make_bwd(i):
        def hook(m, gi, go):
            if go[0] is not None: store[i]["grad"] = go[0].detach().clone()
        return hook
    handles.append(features[idx].register_forward_hook(make_fwd(idx)))
    handles.append(features[idx].register_full_backward_hook(make_bwd(idx)))

print("Running forward + backward (single pass)...", flush=True)
sf = torch.randn(1, 3, 256, 256, device=DEVICE)
fs = torch.randn(1, 5, 3, 256, 256, device=DEVICE)
model.zero_grad()
with torch.enable_grad():
    logit = model(sf, fs)
    print(f"Logit  : {logit.item():.4f}", flush=True)
    logit.squeeze().backward()
print("Done. Removing hooks...\n", flush=True)
for h in handles: h.remove()

print(f"{'Idx':>4}  {'Type':28}  {'ActMax':>8}  {'GradMax':>10}  {'CAMMax':>8}  Status")
print("-"*80, flush=True)

best_idx, best_cam = None, 0.0
for idx in range(n):
    act  = store[idx]["act"]
    grad = store[idx]["grad"]
    lt   = type(features[idx]).__name__[:28]
    if act is None:
        print(f"{idx:>4}  {lt:28}  hook missed", flush=True); continue

    act_max  = float(act.max())
    grad_max = float(grad.abs().max()) if grad is not None else 0.0
    cam_max  = 0.0

    if grad is not None and act_max > 0 and grad_max > 1e-12:
        g2 = grad**2; g3 = grad**3
        den    = 2*g2 + (act*g3).sum(dim=(2,3),keepdim=True)
        alpha  = g2/(den+1e-8)
        w      = (alpha*F.relu(grad)).sum(dim=(2,3),keepdim=True)
        cam_max= float(F.relu((w*act).sum(dim=1)).max())

    if   act_max  <= 0:      status = "✗ negative acts"
    elif grad_max <= 1e-12:  status = "✗ zero grads"
    elif cam_max  <= 1e-6:   status = "✗ CAM=0"
    else:
        status = "✓ VALID"
        if cam_max > best_cam: best_cam, best_idx = cam_max, idx

    print(f"{idx:>4}  {lt:28}  {act_max:>8.3f}  {grad_max:>10.6f}  {cam_max:>8.4f}  {status}", flush=True)

print("\n"+"="*80)
if best_idx is not None:
    offset = best_idx - n
    print(f"✓ Best layer : features[{best_idx}]  ({type(features[best_idx]).__name__})")
    print(f"  CAM max    : {best_cam:.4f}")
    print(f"\n  → Change in models.py:  TARGET_LAYER_IDX = {offset}  (features[{best_idx}] of {n})")
else:
    print("✗ No valid layer found — gradients vanishing throughout MobileNet.")
    print("  This is expected at AUC=0.726 (underfit model).")
    print("  SKIP GradCAM and retrain first:")
    print("  python train.py --data_root . --batch_size 8 --accum_steps 2 --epochs 80 --num_workers 0 --lr 5e-4 --unfreeze_epoch 5 --patience 12 --focal_alpha 0.4 --focal_gamma 2.5")