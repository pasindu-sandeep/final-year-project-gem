import os
import time
import io
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# Import the new utils
from gem_ml_utils import load_model_cached, get_name_map, predict, _ensure_mask_size, combine_masks_for_ids, overlay_masks, axis_aligned_bbox_from_mask, pca_sizes_from_mask, mm_per_pixel_from_distance, estimate_gem_dimensions_from_yolo
from gem_3d_utils import render_glb_bytes, load_trimesh_from_glb_bytes, plot_meshes
from cut_optimizer import optimize_cut_shape
from hardware_serial import send_command # Uses existing hardware connection!

BACKEND_URL = "http://127.0.0.1:5000"

def render_advanced_analysis_page():
    if "show_3d_ui" not in st.session_state:
        st.session_state.show_3d_ui = False

    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        SCRIPT_DIR = Path(os.getcwd())
    WEIGHTS_PATH = str((SCRIPT_DIR / "best.pt").resolve())

    st.markdown("""
        <div class="main-header" style="background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);">
            <h2 style="margin:0; color:white;">💎 Gem Dimensions, Inclusions & 3D</h2>
        </div><br>
    """, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown("### Upload & Settings")
        st.caption(f"Using weights at: `{WEIGHTS_PATH}`")

        if not Path(WEIGHTS_PATH).exists():
            st.error(f"Couldn't find `best.pt` at `{WEIGHTS_PATH}`. Place it next to this script.")
            st.stop()

        model = load_model_cached(WEIGHTS_PATH)
        name_map = get_name_map(model)
        all_class_names = [name_map[i] for i in sorted(name_map.keys())]

        kws = ["inclusion", "fracture", "crack", "feather", "cavity", "cloud", "needle"]
        suggested_inclusions = [n for n in all_class_names if any(k in n.lower() for k in kws)]

        uploaded = st.file_uploader("Upload 1 gem image (JPG/PNG).", type=["jpg", "jpeg", "png"])
        conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.01)

        inclusion_labels = suggested_inclusions
        gem_pick = "Auto (largest non-inclusion)"

        st.markdown("### Camera distance")
        distance_cm = st.number_input("Distance camera → gem (cm)", min_value=0.1, value=30.0, step=0.5)

        with st.expander("Advanced: size calibration (optional)"):
            hfov_deg = st.number_input("Camera Horizontal FOV (deg)", min_value=20.0, max_value=120.0, value=60.0, step=1.0)

        st.markdown("### Overlay display")
        show_inclusion_fill = st.checkbox("Show inclusion fill", value=True)
        show_inclusion_outline = st.checkbox("Show inclusion outline", value=True)
        show_gem_outline = st.checkbox("Show gem outline", value=True)
        alpha = st.slider("Inclusion fill opacity", 0.05, 0.85, 0.35, 0.01)

        run = st.button("▶️ Run Segmentation", type="primary", use_container_width=True)

        st.markdown("### Webcam Capture")
        capture_webcam = st.button("📸 Capture 36 Images from Webcam", use_container_width=True)

        st.divider()
        if st.button("📦 3D Model Generation Tools", use_container_width=True):
            st.session_state.show_3d_ui = True

    with left:
        st.markdown("### Segmentation Result")

        if run and uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            base = np.array(pil_img)
            H, W = base.shape[:2]

            with st.spinner("Running YOLO segmentation..."):
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
            if not candidates and class_area_sum:
                candidates = [(area, cid) for cid, area in class_area_sum.items()]
            gem_class_id = max(candidates, key=lambda x: x[0])[1] if candidates else None
            gem_name = name_map.get(gem_class_id, "Unknown")

            gem_mask = combine_masks_for_ids(r, {gem_class_id} if gem_class_id is not None else set(), (H, W))
            inclusion_mask = combine_masks_for_ids(r, inclusion_id_set, (H, W))

            if gem_mask is None:
                st.error("No gem mask found. Lower confidence or choose the correct gem class.")
            else:
                overlay = overlay_masks(base, inclusion_mask, gem_mask, alpha, show_inclusion_fill, show_inclusion_outline, show_gem_outline)
                st.image(overlay, use_container_width=True)

                gem_area_px = int(gem_mask.sum())
                inc_area_px = int(inclusion_mask.sum()) if inclusion_mask is not None else 0
                inc_frac = (inc_area_px / gem_area_px) if gem_area_px > 0 else 0.0
                bbox = axis_aligned_bbox_from_mask(gem_mask)
                x0, y0, x1, y1, w_px, h_px = bbox if bbox else (0, 0, 0, 0, 0, 0)
                pca = pca_sizes_from_mask(gem_mask)
                major_px, minor_px = pca if pca else (float(w_px), float(h_px))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Gem class", gem_name)
                m2.metric("Inclusion %", f"{inc_frac * 100:.2f}%")
                m3.metric("Width (px)", f"{w_px}px")
                m4.metric("Height (px)", f"{h_px}px")

        elif capture_webcam:
            st.markdown("## 📸 Webcam Capture Running")
            
            # Using main project's serial connection!
            if st.session_state.get("ser"):
                send_command("START")
                st.toast("Motor Started via Main Controller", icon="⚙️")
            else:
                st.warning("Hardware not connected! Motor will not spin. Connect in 'Hardware Controls' tab.")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam not available")
            else:
                frame_placeholder = st.empty()
                progress_placeholder = st.empty()
                status_text = st.empty()
                captured_frames = []
                total_images = 36
                interval = 2.05
                progress_bar = progress_placeholder.progress(0)

                try:
                    for i in range(total_images):
                        start_time = time.time()
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            if time.time() - start_time >= interval: break
                        
                        if not ret: break
                        captured_frames.append(frame_rgb)
                        progress_bar.progress((i + 1) / total_images)
                        status_text.write(f"Captured {i + 1}/{total_images}")
                finally:
                    if st.session_state.get("ser"): send_command("STOP")
                    cap.release()

                st.success("Capture Complete!")
                st.session_state["captured_frames"] = captured_frames

    if st.session_state.show_3d_ui:
        st.divider()
        st.subheader("📦 3D Model Generation")

        uploaded_file = st.file_uploader("Upload Gem Image for 3D Reconstruction", type=["png", "jpg", "jpeg"], key="3d_upload")

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image_bytes = uploaded_file.getvalue()
            pil_3d_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            dimensions = estimate_gem_dimensions_from_yolo(pil_3d_img, model, conf, name_map, inclusion_labels, distance_cm, hfov_deg)

            if st.button("Generate 3D Model", use_container_width=True):
                st.info("Sending image to backend...")
                files = {"image": (uploaded_file.name, io.BytesIO(image_bytes), uploaded_file.type)}
                try:
                    res = requests.post(f"{BACKEND_URL}/generate_3d", files=files, proxies={"http": None, "https": None}, timeout=120)
                except Exception:
                    st.error("Cannot connect to backend")
                    st.stop()

                data = res.json()
                task_id = data["task_id"]
                st.success(f"Task Created: {task_id}")
                progress = st.progress(0)
                glb_data = None

                for _ in range(120):
                    r = requests.get(f"{BACKEND_URL}/get_glb/{task_id}", proxies={"http": None, "https": None})
                    content_type = r.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        progress.progress(min(progress.progress + 5, 95))
                        time.sleep(5)
                    elif "model/gltf-binary" in content_type:
                        glb_data = r.content
                        progress.progress(100)
                        break

                if glb_data:
                    st.success("✅ 3D Model Ready!")
                    render_glb_bytes(glb_data, "Generated 3D Gem")
                    
                    target_shape_path = SCRIPT_DIR / "shapes" / "oval.glb"
                    if target_shape_path.exists():
                        with st.spinner("Computing optimized cut..."):
                            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                                tmp.write(glb_data)
                                rough_glb_path = tmp.name
                            try:
                                result = optimize_cut_shape(rough_gem=rough_glb_path, target_shape=target_shape_path, rough_sample_n=8000, target_sample_n=1500, n_slices=60, spin_step=15, axis_positions=11, transverse_steps=3)
                                rough_mesh = load_trimesh_from_glb_bytes(glb_data)
                                if result.fitted_mesh:
                                    st.success(f"Optimized cut found. Score: {result.score:.5f}")
                                    fig = plot_meshes(rough_mesh, result.fitted_mesh)
                                    st.plotly_chart(fig, use_container_width=True)
                            finally:
                                if os.path.exists(rough_glb_path): os.remove(rough_glb_path)