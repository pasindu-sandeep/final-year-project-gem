import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image

# ================================
# Config
# ================================
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
ENTROPY_THRESHOLD = 1.00
DEFAULT_CLASS_NAMES = [
    "Yellow Sapphires-Cut",
    "Yellow Sapphires-Rough",
    "Zircon Spectrum (Cut)",
    "Zircon Spectrum(Rough)",
]

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ================================
# Model
# ================================
class GemClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ================================
# Model loading
# ================================
def load_model(model_path: str, device: torch.device):
    """Load GemClassifier from a bundle file. Returns (model, class_names, error)."""
    if not os.path.exists(model_path):
        return None, None, f"Model not found at {model_path}"
    try:
        bundle = torch.load(model_path, map_location=device, weights_only=False)
        class_names = bundle.get("class_names", DEFAULT_CLASS_NAMES)
        m = GemClassifier(num_classes=len(class_names)).to(device)
        m.load_state_dict(bundle["model_state_dict"])
        m.eval()
        return m, class_names, None
    except Exception as e:
        return None, None, str(e)


# ================================
# Image helpers
# ================================
def decode_image(img_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPG/PNG) to an RGB numpy array."""
    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


def _preprocess(img_array: np.ndarray, device: torch.device) -> torch.Tensor:
    return val_transform(Image.fromarray(img_array)).unsqueeze(0).to(device)


# ================================
# Spectrum validation
# ================================
def _is_likely_spectrum(img_array: np.ndarray):
    """Return (True, None) if image looks like a spectrum, else (False, reason)."""
    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if h < 10 or w < 10:
        return False, "Image is too small"
    if (w / max(h, 1)) < 0.35:
        return False, "Image is extremely portrait-oriented"
    global_std = float(gray.std())
    if global_std < 6.0:
        return False, "Image has too little contrast"
    max_brightness = float(gray.max())
    if max_brightness < 30:
        return False, "Image is too dark"
    if float(gray.mean(axis=1).std()) < 4.0 and float(gray.mean(axis=0).std()) < 4.0:
        return False, "No band structure detected"
    bright_frac = np.sum(gray.mean(axis=1) > max_brightness * 0.55) / max(h, 1)
    if bright_frac > 0.90 and global_std < 15.0:
        return False, "Uniformly lit photo"
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    bright_mask = gray > (max_brightness * 0.3)
    if bright_mask.sum() > 50:
        hue_range = float(hsv[:, :, 0][bright_mask].max() - hsv[:, :, 0][bright_mask].min())
        mean_sat = float(hsv[:, :, 1][bright_mask].mean())
        if hue_range < 15 and mean_sat > 100 and global_std < 20.0:
            return False, "Single-colour gem photo"
    return True, None


def _entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ================================
# Inference
# ================================
def run_inference(model: GemClassifier, class_names: list, img_array: np.ndarray, device: torch.device):
    """Run spectrum classification. Returns (result_dict, None) or (None, error_str)."""
    try:
        ok, reason = _is_likely_spectrum(img_array)
        if not ok:
            return None, f"Not a spectrum image: {reason}"
        tensor = _preprocess(img_array, device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        if confidence < CONFIDENCE_THRESHOLD:
            return None, f"Low confidence ({confidence * 100:.1f}%)"
        if _entropy(probs) > ENTROPY_THRESHOLD:
            return None, f"Too uncertain (entropy={_entropy(probs):.2f})"
        return {
            "predicted_class": class_names[pred_idx],
            "confidence": round(confidence * 100, 2),
            "all_scores": {class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))},
        }, None
    except Exception as e:
        return None, str(e)
