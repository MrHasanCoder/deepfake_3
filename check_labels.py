# check_labels.py
import torch
import numpy as np
from dataset_loader import build_dataloaders
from models import DeepfakeDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# n_segment=5 to match the checkpoint that was trained with 5 frames
model = DeepfakeDetector(n_segment=5, freeze_backbone_layers=10).to(device)
ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Also use num_frames=5 to match what the model was trained on
_, val_loader, _ = build_dataloaders(".", batch_size=8, num_workers=0, num_frames=5)

real_scores, fake_scores = [], []

with torch.no_grad():
    for batch in val_loader:
        if batch is None:
            continue
        ft, sf, labels = batch
        sf = sf.to(device)
        ft = ft.to(device)
        logits = model(sf, ft)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = [float(probs)]
        else:
            probs = probs.tolist()
        for p, l in zip(probs, labels.tolist()):
            if l == 0:
                real_scores.append(p)
            else:
                fake_scores.append(p)

print(f"Real videos  → avg score: {np.mean(real_scores):.3f}  (should be LOW  < 0.5)")
print(f"Fake videos  → avg score: {np.mean(fake_scores):.3f}  (should be HIGH > 0.5)")
print()
if np.mean(real_scores) > np.mean(fake_scores):
    print("CONFIRMED: Labels are FLIPPED. Model predicts high=real, low=fake.")
    print("Fix: invert the score in inference.py → fake_prob = 1.0 - sigmoid(logit)")
else:
    print("Labels are CORRECT. Problem is domain gap (webcam vs dataset).")
    print("Fix: raise FAKE_THRESHOLD in inference.py to 0.65 or higher.")