import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict

@st.cache_resource(show_spinner=False)
def load_model_cached(wpath: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(wpath)
    return model.to(device)

def get_name_map(model: YOLO):
    if isinstance(model.names, dict):
        return model.names
    return {i: n for i, n in enumerate(model.names)}

def predict(model: YOLO, pil_img: Image.Image, conf: float):
    results = model.predict(pil_img, conf=conf, verbose=False)
    return results[0]

def _ensure_mask_size(mask_bool: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    if mask_bool.shape[:2] == (Ht, Wt):
        return mask_bool
    try:
        resized = cv2.resize(mask_bool.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(bool)
        return resized
    except Exception:
        return (np.array(Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((Wt, Ht), resample=Image.NEAREST)) > 0)

def combine_masks_for_ids(r, wanted_ids: set[int], target_hw: tuple[int, int]):
    if r.masks is None or r.boxes is None or len(r.boxes) == 0 or not wanted_ids:
        return None
    cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
    masks = r.masks.data.detach().cpu().numpy()
    selected = []
    for j, cid in enumerate(cls_ids):
        if cid in wanted_ids:
            m = masks[j] > 0.5
            m = _ensure_mask_size(m, target_hw)
            selected.append(m)
    if not selected:
        return None
    return np.any(np.stack(selected, axis=0), axis=0)

def mask_edge(mask_bool: np.ndarray, thickness: int = 2) -> np.ndarray:
    if thickness < 1: thickness = 1
    try:
        m = mask_bool.astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        er = cv2.erode(m, k, iterations=thickness)
        edge = (m > 0) & (er == 0)
        return edge
    except Exception:
        m = mask_bool
        edge = np.zeros_like(m, dtype=bool)
        edge[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
        return edge

def overlay_masks(base_rgb: np.ndarray, inclusion_mask: np.ndarray | None, gem_mask: np.ndarray | None, alpha: float = 0.35, show_inclusion_fill: bool = True, show_inclusion_outline: bool = True, show_gem_outline: bool = True):
    out = base_rgb.copy().astype(np.float32)
    if inclusion_mask is not None and show_inclusion_fill:
        color = np.array([255, 0, 0], dtype=np.float32)
        out[inclusion_mask] = (1 - alpha) * out[inclusion_mask] + alpha * color
    if inclusion_mask is not None and show_inclusion_outline:
        e = mask_edge(inclusion_mask, thickness=2)
        out[e] = np.array([255, 255, 0], dtype=np.float32)
    if gem_mask is not None and show_gem_outline:
        e = mask_edge(gem_mask, thickness=2)
        out[e] = np.array([0, 255, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def axis_aligned_bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1, (x1 - x0 + 1), (y1 - y0 + 1)

def pca_sizes_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) < 10: return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    v1 = eigvecs[:, order[0]]
    v2 = eigvecs[:, order[1]]
    p1 = X @ v1
    p2 = X @ v2
    return float(p1.max() - p1.min()), float(p2.max() - p2.min())

def mm_per_pixel_from_distance(distance_mm: float, img_w: int, img_h: int, hfov_deg: float):
    hfov = np.deg2rad(hfov_deg)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * (img_h / img_w))
    physical_w_mm = 2.0 * distance_mm * np.tan(hfov / 2.0)
    physical_h_mm = 2.0 * distance_mm * np.tan(vfov / 2.0)
    return physical_w_mm / max(1, img_w), physical_h_mm / max(1, img_h)

def estimate_gem_dimensions_from_yolo(pil_img, model, conf, name_map, inclusion_labels, distance_cm, hfov_deg):
    base = np.array(pil_img)
    H, W = base.shape[:2]
    r = predict(model, pil_img, conf=conf)
    class_area_sum = defaultdict(int)

    if r.masks is not None and r.boxes is not None and len(r.boxes.cls) > 0:
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        masks = r.masks.data.detach().cpu().numpy()
        for j, cid in enumerate(cls_ids):
            m = _ensure_mask_size((masks[j] > 0.5), (H, W))
            class_area_sum[cid] += int(m.sum())

    inclusion_id_set = {i for i, n in name_map.items() if n in inclusion_labels}
    candidates = [(area, cid) for cid, area in class_area_sum.items() if cid not in inclusion_id_set]
    if not candidates: return None

    gem_class_id = max(candidates, key=lambda x: x[0])[1]
    gem_mask = combine_masks_for_ids(r, {gem_class_id}, (H, W))
    if gem_mask is None: return None
    bbox = axis_aligned_bbox_from_mask(gem_mask)
    if bbox is None: return None

    x0, y0, x1, y1, w_px, h_px = bbox
    pca = pca_sizes_from_mask(gem_mask)
    major_px, minor_px = pca if pca else (float(w_px), float(h_px))

    distance_mm = float(distance_cm) * 10.0
    mm_per_px_x, mm_per_px_y = mm_per_pixel_from_distance(distance_mm, W, H, float(hfov_deg))
    mm_per_px_avg = 0.5 * (mm_per_px_x + mm_per_px_y)

    return {
        "width_px": w_px, "height_px": h_px,
        "width_mm": w_px * mm_per_px_x, "height_mm": h_px * mm_per_px_y,
        "major_px": major_px, "minor_px": minor_px,
        "major_mm": major_px * mm_per_px_avg, "minor_mm": minor_px * mm_per_px_avg,
        "gem_class": name_map.get(gem_class_id, "Unknown"),
    }