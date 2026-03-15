"""
SurgiVision Segmentation Server
================================
Flask backend wrapping EndoViT + DPT head for real-time surgical segmentation.
Runs on localhost:5050 — called by the SurgiVision browser app.

CholecSeg8k 8-class mapping:
  0  Black Background
  1  Abdominal Wall
  2  Liver
  3  Gastrointestinal Tract
  4  Fat
  5  Grasper (instrument)
  6  Connective Tissue
  7  Blood / Bleeding
  8  Cystic Duct
  9  L-hook Electrocautery (instrument)
  10 Gallbladder
  11 Hepatic Vein
  12 Liver Ligament
"""

import io
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CHECKPOINT_PATH = Path("pretraining/pretrained_endovit_models/EndoViT_for_Segmentation/endovit_seg.pth")
NUM_CLASSES     = 13        # CholecSeg8k has 13 classes
IMAGE_SIZE      = 224       # EndoViT pretraining resolution
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"[SurgiVision] Using device: {DEVICE}")

# ─────────────────────────────────────────
#  CLASS DEFINITIONS
#  Maps CholecSeg8k class index → label, color, surgivision type
# ─────────────────────────────────────────
CLASS_INFO = {
    0:  {"label": "Background",            "color": "#1a1a1a", "type": "other",      "skip": True},
    1:  {"label": "Abdominal Wall",        "color": "#c08060", "type": "anatomy"},
    2:  {"label": "Liver",                 "color": "#c43020", "type": "anatomy"},
    3:  {"label": "Gastrointestinal",      "color": "#a06040", "type": "anatomy"},
    4:  {"label": "Fat",                   "color": "#d0b040", "type": "tissue"},
    5:  {"label": "Grasper",               "color": "#c0c8d0", "type": "instrument"},
    6:  {"label": "Connective Tissue",     "color": "#b08080", "type": "tissue"},
    7:  {"label": "Blood",                 "color": "#ff2040", "type": "fluid"},
    8:  {"label": "Cystic Duct",           "color": "#e03050", "type": "anatomy"},
    9:  {"label": "Electrocautery Hook",   "color": "#e0e0ff", "type": "instrument"},
    10: {"label": "Gallbladder",           "color": "#40a040", "type": "anatomy"},
    11: {"label": "Hepatic Vein",          "color": "#8040c0", "type": "anatomy"},
    12: {"label": "Liver Ligament",        "color": "#804020", "type": "tissue"},
}

# ─────────────────────────────────────────
#  DPT SEGMENTATION HEAD
#  Minimal implementation matching EndoViT's finetuning setup
# ─────────────────────────────────────────

class DPTHead(nn.Module):
    """Lightweight DPT segmentation head that sits on top of the ViT encoder."""
    def __init__(self, in_channels=768, num_classes=13, image_size=224):
        super().__init__()
        patch_size  = 16
        num_patches = (image_size // patch_size) ** 2  # 196 for 224x224
        self.patch_h = self.patch_w = image_size // patch_size  # 14x14

        # Reassemble 4 ViT layer outputs into feature maps
        self.reassemble = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1),
                nn.ConvTranspose2d(256, 256, kernel_size=k, stride=s, padding=p)
            )
            for (k, s, p) in [(4,4,0), (2,2,0), (1,1,0), (1,1,0)]  # scales: 4x, 2x, 1x, 0.5x
        ])

        # Fusion layers
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()),
        ])

        # Final segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
            nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True),
        )

    def forward(self, features):
        """features: list of 4 tensors [B, N+1, C] from ViT layers 3,6,9,12"""
        outs = []
        for i, feat in enumerate(features):
            # Remove CLS token, reshape to spatial
            x = feat[:, 1:, :]  # [B, N, C]
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, self.patch_h, self.patch_w)
            x = self.reassemble[i](x)
            x = self.fusion[i](x)
            outs.append(x)

        # Progressive fusion (bottom-up)
        x = outs[3]
        for i in range(2, -1, -1):
            x = nn.functional.interpolate(x, size=outs[i].shape[2:], mode='bilinear', align_corners=True)
            x = x + outs[i]

        return self.head(x)


class EndoViTSegmentation(nn.Module):
    """Full model: EndoViT backbone + DPT segmentation head."""
    def __init__(self, num_classes=13, image_size=224):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        # ViT-Base backbone (matches EndoViT architecture)
        self.backbone = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.head = DPTHead(in_channels=768, num_classes=num_classes, image_size=image_size)
        self._hook_layers = []
        self._features    = {}
        self._register_hooks()

    def _register_hooks(self):
        """Hook into ViT layers 3, 6, 9, 11 to extract intermediate features."""
        target_layers = [2, 5, 8, 11]  # 0-indexed
        for idx in target_layers:
            layer = self.backbone.blocks[idx]
            layer.register_forward_hook(
                lambda m, inp, out, i=idx: self._features.__setitem__(i, out)
            )

    def forward(self, x):
        self._features.clear()
        _ = self.backbone(x)
        feats = [self._features[i] for i in [2, 5, 8, 11]]
        return self.head(feats)


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────

def load_model():
    print("[SurgiVision] Loading EndoViT + DPT segmentation model...")
    model = EndoViTSegmentation(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)

    # Check for fine-tuned checkpoint first, fall back to backbone-only
    finetuned_path = Path("pretraining/pretrained_endovit_models/EndoViT_for_Segmentation/surgivision_seg_best.pth")
    backbone_path  = CHECKPOINT_PATH

    if finetuned_path.exists():
        print(f"[SurgiVision] ✓ Loading FINE-TUNED checkpoint: {finetuned_path}")
        ckpt = torch.load(finetuned_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        val_miou = ckpt.get("val_miou", 0)
        print(f"[SurgiVision] ✓ Full model loaded — val mIoU: {val_miou*100:.1f}%")
        print(f"[SurgiVision]   Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif backbone_path.exists():
        print(f"[SurgiVision] Loading backbone-only checkpoint: {backbone_path}")
        print(f"[SurgiVision] ⚠ DPT head is randomly initialized — run Colab training first!")
        ckpt = torch.load(backbone_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        backbone_state = {
            k.replace("encoder.", "").replace("module.", ""): v
            for k, v in state.items()
            if not any(skip in k for skip in ["decoder", "mask_token"])
        }
        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"[SurgiVision] Backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"[SurgiVision] ⚠ No checkpoint found — using random weights (for testing only)")

    model = model.to(DEVICE)
    model.eval()
    print(f"[SurgiVision] Model ready on {DEVICE}")
    return model


# ─────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────

# EndoViT normalization stats (from Endo700k dataset)
TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.3464, 0.2280, 0.2228], std=[0.2520, 0.2128, 0.2093])
])


def preprocess(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image.convert("RGB")).unsqueeze(0)


# ─────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────

def run_segmentation(model, image: Image.Image, orig_w: int, orig_h: int):
    """Returns list of detected structures with mask pixel lists."""
    tensor = preprocess(image).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)                        # [1, 13, 224, 224]
        pred   = logits.argmax(dim=1).squeeze(0)      # [224, 224]
        pred_np = pred.cpu().numpy().astype(np.uint8) # [224, 224]

    # Scale mask back to original image size
    mask_img = Image.fromarray(pred_np).resize((orig_w, orig_h), Image.NEAREST)
    mask_np  = np.array(mask_img)

    structures = []
    for class_idx, info in CLASS_INFO.items():
        if info.get("skip"):
            continue

        pixel_mask = (mask_np == class_idx)
        pixel_count = int(pixel_mask.sum())

        # Skip classes covering < 0.5% of image
        total_pixels = orig_w * orig_h
        if pixel_count < total_pixels * 0.005:
            continue

        # Compute bounding box
        rows = np.any(pixel_mask, axis=1)
        cols = np.any(pixel_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Normalize bbox to 0-1
        bbox = [
            float(cmin / orig_w),
            float(rmin / orig_h),
            float((cmax - cmin) / orig_w),
            float((rmax - rmin) / orig_h),
        ]

        # Build sparse mask: list of [x,y] pixel coords (downsampled for JSON size)
        ys, xs = np.where(pixel_mask)
        # Downsample to max 2000 points for performance
        if len(xs) > 2000:
            idx = np.random.choice(len(xs), 2000, replace=False)
            xs, ys = xs[idx], ys[idx]
        mask_points = [[int(x), int(y)] for x, y in zip(xs, ys)]

        structures.append({
            "label":       info["label"],
            "type":        info["type"],
            "color":       info["color"],
            "confidence":  round(pixel_count / total_pixels, 3),
            "bbox":        bbox,
            "mask_points": mask_points,
            "pixel_count": pixel_count,
        })

    # Sort by pixel count descending
    structures.sort(key=lambda s: s["pixel_count"], reverse=True)
    return structures


# ─────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────

app   = Flask(__name__)
CORS(app)  # Allow requests from localhost:8080 (SurgiVision)
model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE, "model": "EndoViT-DPT-Seg"})


@app.route("/segment", methods=["POST"])
def segment():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field (base64 JPEG)"}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(data["image"])
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        # Run segmentation
        structures = run_segmentation(model, image, orig_w, orig_h)

        return jsonify({
            "structures": structures,
            "image_size": [orig_w, orig_h],
            "model":      "EndoViT-DPT",
            "device":     DEVICE,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("[SurgiVision] Server starting on http://localhost:5050")
    print("[SurgiVision] Health check: http://localhost:5050/health")
    app.run(host="0.0.0.0", port=5050, debug=False)
