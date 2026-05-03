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

from gem_ml_utils import (
    load_model_cached,
    get_name_map,
    predict,
    _ensure_mask_size,
    combine_masks_for_ids,
    overlay_masks,
    axis_aligned_bbox_from_mask,
    pca_sizes_from_mask,
    mm_per_pixel_from_distance,
    estimate_gem_dimensions_from_yolo,
)
from gem_3d_utils import render_glb_bytes, load_trimesh_from_glb_bytes, plot_meshes
from cut_optimizer import optimize_cut_shape
from hardware_serial import send_command

BACKEND_URL = "http://13.51.195.52:8000"


def _read_json_or_stop(response: requests.Response, context: str) -> dict:
    """Safely read JSON from a backend response and stop Streamlit with a useful error if invalid."""
    try:
        return response.json()
    except ValueError:
        st.error(f"Backend did not return valid JSON while {context}.")
        st.write("Status code:", response.status_code)
        st.write("Content-Type:", response.headers.get("Content-Type", ""))
        st.write("Response text:", response.text)
        st.stop()


def render_advanced_analysis_page():
    if "show_3d_ui" not in st.session_state:
        st.session_state.show_3d_ui = False

    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        SCRIPT_DIR = Path(os.getcwd())

    WEIGHTS_PATH = str((SCRIPT_DIR / "best.pt").resolve())

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown("### Inclusion & Dimension Analysis")

        if not Path(WEIGHTS_PATH).exists():
            st.error(f"Couldn't find `best.pt` at `{WEIGHTS_PATH}`. Place it next to this script.")
            st.stop()

        model = load_model_cached(WEIGHTS_PATH)
        name_map = get_name_map(model)
        all_class_names = [name_map[i] for i in sorted(name_map.keys())]

        kws = ["inclusion", "fracture", "crack", "feather", "cavity", "cloud", "needle"]
        suggested_inclusions = [n for n in all_class_names if any(k in n.lower() for k in kws)]

        uploaded = st.file_uploader(
            "Upload gem images (JPG/PNG).",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.01)

        inclusion_labels = suggested_inclusions

        distance_cm = st.number_input(
            "Distance camera → gem (cm)",
            min_value=0.1,
            value=30.0,
            step=0.5,
        )

        st.markdown("### Overlay display")
        show_inclusion_fill = st.checkbox("Show inclusion fill", value=True)
        show_inclusion_outline = st.checkbox("Show inclusion outline", value=True)
        show_gem_outline = st.checkbox("Show gem outline", value=True)
        alpha = st.slider("Inclusion fill opacity", 0.05, 0.85, 0.35, 0.01)

        run = st.button("▶️ Run", type="primary", width="stretch")

        st.markdown("### Webcam Capture")
        capture_webcam = st.button("📸 Capture 36 Images from Webcam", width="stretch")

        st.divider()
        if st.button("📦 3D Model Generation Tools", width="stretch"):
            st.session_state.show_3d_ui = True

    with left:
        st.markdown("### Segmentation Result")

        if run and uploaded:
            summary_results = []

            for row_start in range(0, len(uploaded), 2):
                cols = st.columns(2, gap="large")

                for col_idx in range(2):
                    file_idx = row_start + col_idx
                    if file_idx >= len(uploaded):
                        continue

                    uploaded_file = uploaded[file_idx]

                    with cols[col_idx]:
                        st.markdown(f"#### {uploaded_file.name}")

                        pil_img = Image.open(uploaded_file).convert("RGB")
                        base = np.array(pil_img)
                        H, W = base.shape[:2]

                        with st.spinner(f"Running YOLO segmentation on {uploaded_file.name}..."):
                            r = predict(model, pil_img, conf=conf)

                        class_area_sum = defaultdict(int)
                        if r.masks is not None and r.boxes is not None and len(r.boxes.cls) > 0:
                            cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
                            masks = r.masks.data.detach().cpu().numpy()
                            for j, cid in enumerate(cls_ids):
                                m = _ensure_mask_size((masks[j] > 0.5), (H, W))
                                class_area_sum[cid] += int(m.sum())

                        inclusion_id_set = {i for i, n in name_map.items() if n in inclusion_labels}
                        candidates = [
                            (area, cid)
                            for cid, area in class_area_sum.items()
                            if cid not in inclusion_id_set
                        ]
                        if not candidates and class_area_sum:
                            candidates = [(area, cid) for cid, area in class_area_sum.items()]

                        gem_class_id = max(candidates, key=lambda x: x[0])[1] if candidates else None
                        gem_name = name_map.get(gem_class_id, "Unknown")

                        gem_mask = combine_masks_for_ids(
                            r,
                            {gem_class_id} if gem_class_id is not None else set(),
                            (H, W),
                        )
                        inclusion_mask = combine_masks_for_ids(r, inclusion_id_set, (H, W))

                        if gem_mask is None:
                            st.error("No gem mask found. Lower confidence or choose the correct gem class.")
                        else:
                            overlay = overlay_masks(
                                base,
                                inclusion_mask,
                                gem_mask,
                                alpha,
                                show_inclusion_fill,
                                show_inclusion_outline,
                                show_gem_outline,
                            )
                            st.image(overlay, width="stretch")

                            gem_area_px = int(gem_mask.sum())
                            inc_area_px = int(inclusion_mask.sum()) if inclusion_mask is not None else 0
                            inc_frac = (inc_area_px / gem_area_px) if gem_area_px > 0 else 0.0
                            bbox = axis_aligned_bbox_from_mask(gem_mask)
                            x0, y0, x1, y1, w_px, h_px = bbox if bbox else (0, 0, 0, 0, 0, 0)
                            pca = pca_sizes_from_mask(gem_mask)
                            major_px, minor_px = pca if pca else (float(w_px), float(h_px))

                            summary_results.append(
                                {
                                    "gem_name": gem_name,
                                    "inc_percent": inc_frac * 100,
                                    "width_px": w_px,
                                    "height_px": h_px,
                                }
                            )

            if summary_results:
                gem_class_counts = defaultdict(int)
                for item in summary_results:
                    gem_class_counts[item["gem_name"]] += 1

                most_common_gem_class = max(gem_class_counts.items(), key=lambda x: x[1])[0]
                avg_inclusion_percent = sum(item["inc_percent"] for item in summary_results) / len(summary_results)
                avg_width_px = sum(item["width_px"] for item in summary_results) / len(summary_results)
                avg_height_px = sum(item["height_px"] for item in summary_results) / len(summary_results)

                st.divider()
                st.markdown("### Average Summary")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Gem class", most_common_gem_class)
                a2.metric("Avg Inclusion %", f"{avg_inclusion_percent:.2f}%")
                a3.metric("Avg Width (px)", f"{avg_width_px:.0f}px")
                a4.metric("Avg Height (px)", f"{avg_height_px:.0f}px")

        elif capture_webcam:
            st.markdown("## 📸 Webcam Capture Running")

            if st.session_state.get("ser"):
                send_command("START")
                st.toast("Motor Started via Main Controller", icon="⚙️")
            else:
                st.warning(
                    "Hardware not connected! Motor will not spin. Connect in 'Hardware Controls' tab."
                )

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
                        ret = False

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                            if time.time() - start_time >= interval:
                                break

                        if not ret:
                            break

                        captured_frames.append(frame_rgb)
                        progress_bar.progress((i + 1) / total_images)
                        status_text.write(f"Captured {i + 1}/{total_images}")
                finally:
                    if st.session_state.get("ser"):
                        send_command("STOP")
                    cap.release()

                st.success("Capture Complete!")
                st.session_state["captured_frames"] = captured_frames

    if st.session_state.show_3d_ui:
        st.divider()
        st.subheader("📦 3D Model Generation")

        uploaded_file = st.file_uploader(
            "Upload Gem Image for 3D Reconstruction",
            type=["png", "jpg", "jpeg"],
            key="3d_upload",
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width="stretch")
            image_bytes = uploaded_file.getvalue()
            pil_3d_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            dimensions = estimate_gem_dimensions_from_yolo(
                pil_3d_img,
                model,
                conf,
                name_map,
                inclusion_labels,
                distance_cm,
                hfov_deg,
            )

            if dimensions:
                with st.expander("Estimated dimensions from YOLO", expanded=False):
                    st.write(dimensions)

            if st.button("Generate 3D Model", width="stretch"):
                st.info("Sending image to backend...")

                files = {
                    "image": (
                        uploaded_file.name,
                        io.BytesIO(image_bytes),
                        uploaded_file.type,
                    )
                }

                try:
                    res = requests.post(
                        f"{BACKEND_URL}/generate_3d",
                        files=files,
                        proxies={"http": None, "https": None},
                        timeout=12000,
                    )
                except Exception as e:
                    st.error(f"Cannot connect to backend: {e}")
                    st.stop()

                if res.status_code != 200:
                    st.error(f"Backend error while creating task: {res.status_code}")
                    st.write(res.text)
                    st.stop()

                data = _read_json_or_stop(res, "creating the 3D generation task")

                task_id = data.get("task_id")
                if not task_id:
                    st.error("Backend response did not contain `task_id`.")
                    st.write(data)
                    st.stop()

                st.success(f"Task Created: {task_id}")

                progress = st.progress(0)
                progress_value = 0
                glb_data = None
                status_placeholder = st.empty()

                status_placeholder.info("Generating 3D model...")

                for _ in range(120):
                    try:
                        r = requests.get(
                            f"{BACKEND_URL}/get_glb/{task_id}",
                            proxies={"http": None, "https": None},
                            timeout=120,
                        )
                    except Exception as e:
                        st.error(f"Cannot check backend task status: {e}")
                        st.stop()

                    if r.status_code != 200:
                        st.error(f"Backend error while checking task: {r.status_code}")
                        st.write(r.text)
                        st.stop()

                    content_type = r.headers.get("Content-Type", "")

                    if "application/json" in content_type:
                        status_data = _read_json_or_stop(r, "checking the 3D generation status")
                        status = status_data.get("status", "UNKNOWN")

                        status_placeholder.info(f"Status: {status}")

                        if status == "INPROGRESS":
                            progress_value = min(progress_value + 5, 95)
                            progress.progress(progress_value)
                            time.sleep(5)
                            continue

                        st.error(f"Unexpected backend status: {status}")
                        st.write(status_data)
                        st.stop()

                    elif "model/gltf-binary" in content_type:
                        glb_data = r.content
                        progress.progress(100)
                        status_placeholder.success("3D model file received from backend.")
                        break

                    else:
                        st.error(f"Unexpected response type from backend: {content_type}")
                        st.write(r.text)
                        st.stop()

                if glb_data is None:
                    st.error("Model generation timeout. Backend kept returning INPROGRESS.")
                    st.stop()

                st.success("✅ 3D Model Ready!")
                render_glb_bytes(glb_data, "Generated 3D Gem")

                target_shape_path = SCRIPT_DIR / "shapes" / "oval.glb"
                if target_shape_path.exists():
                    with st.spinner("Computing optimized cut..."):
                        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                            tmp.write(glb_data)
                            rough_glb_path = tmp.name

                        try:
                            result = optimize_cut_shape(
                                rough_gem=rough_glb_path,
                                target_shape=target_shape_path,
                                rough_sample_n=8000,
                                target_sample_n=1500,
                                n_slices=60,
                                spin_step=15,
                                axis_positions=11,
                                transverse_steps=3,
                            )

                            rough_mesh = load_trimesh_from_glb_bytes(glb_data)

                            if result.fitted_mesh:
                                st.success(f"Optimized cut found. Score: {result.score:.5f}")
                                fig = plot_meshes(rough_mesh, result.fitted_mesh)
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("No valid cut placement found.")

                        except Exception as e:
                            st.error(f"Cut optimization failed: {e}")

                        finally:
                            if os.path.exists(rough_glb_path):
                                os.remove(rough_glb_path)
                else:
                    st.warning(f"Skipping cut optimization. Missing target shape file: {target_shape_path}")
