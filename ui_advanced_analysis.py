import os
import time
import io
import base64
import tempfile
from collections import defaultdict
from pathlib import Path
import serial
import serial.tools.list_ports

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageFilter

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

@st.cache_resource(show_spinner=False)
def get_arduino():
    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        device = (p.device or "").lower()
        description = (p.description or "").lower()
        hwid = (p.hwid or "").lower()

        is_arduino_like = any(
            key in f"{device} {description} {hwid}"
            for key in ["usbmodem", "usbserial", "ttyacm", "ttyusb", "arduino", "ch340"]
        )

        if not is_arduino_like:
            continue

        try:
            ser = serial.Serial(p.device, 9600, timeout=1)
            time.sleep(2)
            return ser, p.device, None
        except Exception as e:
            return None, p.device, str(e)

    return None, None, None


def render_hardware_status():
    ser, port, error = get_arduino()
    st.session_state["ser"] = ser

    with st.sidebar:
        st.markdown("### Device Status")

        if ser is not None and getattr(ser, "is_open", False):
            st.success(f"✅ Connected: {port}")
        elif port and error:
            st.error(f"❌ Found {port} but failed: {error}")
        else:
            st.warning("⚠️ Device not connected")

    return ser


def send_motor_signal(action: str) -> bool:
    ser = st.session_state.get("ser")

    if ser is None or not getattr(ser, "is_open", False):
        st.warning("Hardware not connected. Motor command was not sent.")
        return False

    try:
        if action.upper() == "START":
            ser.write(b"1")
        elif action.upper() == "STOP":
            ser.write(b"0")
        else:
            raise ValueError(f"Unknown motor action: {action}")

        ser.flush()
        return True

    except Exception as e:
        st.error(f"Motor {action.lower()} failed: {e}")
        return False


def _read_json_or_stop(response: requests.Response, context: str) -> dict:
    try:
        return response.json()
    except ValueError:
        st.error(f"Backend did not return valid JSON while {context}.")
        st.write("Status code:", response.status_code)
        st.write("Content-Type:", response.headers.get("Content-Type", ""))
        st.write("Response text:", response.text)
        st.stop()


def _inject_page_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 2rem;
            }

            .gem-hero {
                padding: 1.15rem 1.35rem;
                border-radius: 1.1rem;
                background: linear-gradient(135deg, #102033 0%, #183b56 48%, #0f766e 100%);
                color: white;
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.20);
                margin-bottom: 1.2rem;
            }

            .gem-hero h2 {
                margin: 0 0 0.25rem 0;
                font-size: 1.55rem;
                font-weight: 750;
            }

            .gem-hero p {
                margin: 0;
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.94rem;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 750;
                margin: 0.2rem 0 0.55rem 0;
            }

            .section-subtitle {
                color: #64748b;
                font-size: 0.88rem;
                margin-top: -0.35rem;
                margin-bottom: 0.75rem;
            }

            .empty-state {
                padding: 2.2rem 1.2rem;
                border-radius: 1rem;
                border: 1px dashed rgba(100, 116, 139, 0.45);
                background: rgba(248, 250, 252, 0.72);
                text-align: center;
                color: #64748b;
            }

           div[data-testid="stMetric"] {
                background: #f3f4f6;
                border: 1px solid rgba(226, 232, 240, 0.95);
                padding: 0.75rem;
                border-radius: 0.85rem;
            }

            div[data-testid="stMetric"] * {
                color: #0f172a !important;
            }

            div[data-testid="stMetricLabel"] p {
                color: #334155 !important;
                font-weight: 600 !important;
            }

            div[data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 700 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state():
    defaults = {
        "three_d_glb_data": None,
        "three_d_task_id": None,
        "three_d_cut_fig": None,
        "three_d_cut_score": None,
        "three_d_cut_message": None,
        "three_d_error": None,
        "segmentation_results": None,
        "segmentation_summary": None,
        "captured_frames": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_3d_results():
    st.session_state["three_d_glb_data"] = None
    st.session_state["three_d_task_id"] = None
    st.session_state["three_d_cut_fig"] = None
    st.session_state["three_d_cut_score"] = None
    st.session_state["three_d_cut_message"] = None
    st.session_state["three_d_error"] = None


def _clear_segmentation_results():
    st.session_state["segmentation_results"] = None
    st.session_state["segmentation_summary"] = None
    st.session_state["captured_frames"] = None


def _get_blurred_background_data_uri(image_path: str | None):
    if not image_path:
        return None

    try:
        img_path = Path(image_path)
        if not img_path.exists():
            return None

        img = Image.open(img_path).convert("RGB")
        img = img.resize((1400, 900))
        img = img.filter(ImageFilter.GaussianBlur(radius=12))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _style_cut_figure(fig, background_image_path=None):
    bg_uri = _get_blurred_background_data_uri(background_image_path)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(240,240,240,0.95)" if bg_uri is None else "rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        ),
        showlegend=True,
    )

    if bg_uri is not None:
        fig.add_layout_image(
            dict(
                source=bg_uri,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=0.35,
                layer="below",
            )
        )

    return fig


def _run_3d_generation(
    uploaded_file,
    model,
    conf,
    name_map,
    inclusion_labels,
    distance_cm,
    hfov_deg,
    script_dir,
    cut_bg_image_path=None,
):
    if uploaded_file is None:
        st.warning("Please upload an image for 3D model generation.")
        return

    image_bytes = uploaded_file.getvalue()
    pil_3d_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    _ = estimate_gem_dimensions_from_yolo(
        pil_3d_img,
        model,
        conf,
        name_map,
        inclusion_labels,
        distance_cm,
        hfov_deg,
    )

    files = {
        "image": (
            uploaded_file.name,
            io.BytesIO(image_bytes),
            uploaded_file.type,
        )
    }

    with st.spinner("Sending image to backend..."):
        try:
            res = requests.post(
                f"{BACKEND_URL}/generate_3d",
                files=files,
                proxies={"http": None, "https": None},
                timeout=12000,
            )
        except Exception as e:
            st.error(f"Cannot connect to backend: {e}")
            return

    if res.status_code != 200:
        st.error(f"Backend error while creating task: {res.status_code}")
        st.write(res.text)
        return

    data = _read_json_or_stop(res, "creating the 3D generation task")

    task_id = data.get("task_id")
    if not task_id:
        st.error("Backend response did not contain `task_id`.")
        st.write(data)
        return

    st.session_state["three_d_task_id"] = task_id

    progress = st.progress(0)
    progress_value = 0
    glb_data = None
    status_placeholder = st.empty()

    status_placeholder.info(f"Task Created: {task_id}")

    for _ in range(120):
        try:
            r = requests.get(
                f"{BACKEND_URL}/get_glb/{task_id}",
                proxies={"http": None, "https": None},
                timeout=120,
            )
        except Exception as e:
            st.error(f"Cannot check backend task status: {e}")
            return

        if r.status_code != 200:
            st.error(f"Backend error while checking task: {r.status_code}")
            st.write(r.text)
            return

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
            return

        elif "model/gltf-binary" in content_type:
            glb_data = r.content
            progress.progress(100)
            status_placeholder.success("3D model generation completed.")
            break

        else:
            st.error(f"Unexpected response type from backend: {content_type}")
            st.write(r.text)
            return

    if glb_data is None:
        st.error("Model generation timeout. Backend kept returning INPROGRESS.")
        return

    st.session_state["three_d_glb_data"] = glb_data
    st.session_state["three_d_error"] = None
    st.session_state["three_d_cut_fig"] = None
    st.session_state["three_d_cut_score"] = None
    st.session_state["three_d_cut_message"] = None

    target_shape_path = script_dir / "shapes" / "oval.glb"
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
                    fig = plot_meshes(rough_mesh, result.fitted_mesh)
                    fig = _style_cut_figure(fig, background_image_path=cut_bg_image_path)
                    st.session_state["three_d_cut_fig"] = fig
                    st.session_state["three_d_cut_score"] = result.score
                    st.session_state["three_d_cut_message"] = "Optimized cut found."
                else:
                    st.session_state["three_d_cut_message"] = "No valid cut placement found."
            except Exception as e:
                st.session_state["three_d_cut_message"] = f"Cut optimization failed: {e}"
            finally:
                if os.path.exists(rough_glb_path):
                    os.remove(rough_glb_path)
    else:
        st.session_state["three_d_cut_message"] = f"Skipping cut optimization. Missing target shape file: {target_shape_path}"


def _run_segmentation(
    uploaded_files,
    model,
    name_map,
    inclusion_labels,
    conf,
    alpha,
    show_inclusion_fill,
    show_inclusion_outline,
    show_gem_outline,
):
    if not uploaded_files:
        st.warning("Please upload at least one gem image before running segmentation.")
        return

    results = []
    summary_results = []

    with st.spinner("Running segmentation on uploaded images..."):
        for uploaded_file in uploaded_files:
            pil_img = Image.open(uploaded_file).convert("RGB")
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
                results.append(
                    {
                        "filename": uploaded_file.name,
                        "error": "No gem mask found. Lower confidence or choose the correct gem class.",
                    }
                )
                continue

            overlay = overlay_masks(
                base,
                inclusion_mask,
                gem_mask,
                alpha,
                show_inclusion_fill,
                show_inclusion_outline,
                show_gem_outline,
            )

            gem_area_px = int(gem_mask.sum())
            inc_area_px = int(inclusion_mask.sum()) if inclusion_mask is not None else 0
            inc_frac = (inc_area_px / gem_area_px) if gem_area_px > 0 else 0.0
            bbox = axis_aligned_bbox_from_mask(gem_mask)
            x0, y0, x1, y1, w_px, h_px = bbox if bbox else (0, 0, 0, 0, 0, 0)
            pca = pca_sizes_from_mask(gem_mask)
            major_px, minor_px = pca if pca else (float(w_px), float(h_px))

            results.append(
                {
                    "filename": uploaded_file.name,
                    "overlay": overlay,
                    "gem_name": gem_name,
                    "inc_percent": inc_frac * 100,
                    "width_px": w_px,
                    "height_px": h_px,
                    "major_px": major_px,
                    "minor_px": minor_px,
                    "error": None,
                }
            )

            summary_results.append(
                {
                    "gem_name": gem_name,
                    "inc_percent": inc_frac * 100,
                    "width_px": w_px,
                    "height_px": h_px,
                }
            )

    st.session_state["segmentation_results"] = results

    if summary_results:
        gem_class_counts = defaultdict(int)
        for item in summary_results:
            gem_class_counts[item["gem_name"]] += 1

        most_common_gem_class = max(gem_class_counts.items(), key=lambda x: x[1])[0]
        avg_inclusion_percent = sum(item["inc_percent"] for item in summary_results) / len(summary_results)
        avg_width_px = sum(item["width_px"] for item in summary_results) / len(summary_results)
        avg_height_px = sum(item["height_px"] for item in summary_results) / len(summary_results)

        st.session_state["segmentation_summary"] = {
            "gem_class": most_common_gem_class,
            "avg_inclusion_percent": avg_inclusion_percent,
            "avg_width_px": avg_width_px,
            "avg_height_px": avg_height_px,
        }
    else:
        st.session_state["segmentation_summary"] = None

def _run_segmentation_on_captured_frames(
    captured_frames,
    model,
    name_map,
    inclusion_labels,
    conf,
    alpha,
    show_inclusion_fill,
    show_inclusion_outline,
    show_gem_outline,
):
    if not captured_frames:
        st.warning("No webcam frames were captured.")
        return

    results = []
    summary_results = []

    with st.spinner("Running YOLO predictions on captured webcam images..."):
        for idx, frame in enumerate(captured_frames):
            pil_img = Image.fromarray(frame).convert("RGB")
            H, W = frame.shape[:2]

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

            inclusion_mask = combine_masks_for_ids(
                r,
                inclusion_id_set,
                (H, W),
            )

            if gem_mask is None:
                results.append(
                    {
                        "filename": f"Captured Image {idx + 1}",
                        "error": "No gem found.",
                    }
                )
                continue

            overlay = overlay_masks(
                frame,
                inclusion_mask,
                gem_mask,
                alpha,
                show_inclusion_fill,
                show_inclusion_outline,
                show_gem_outline,
            )

            gem_area_px = int(gem_mask.sum())
            inc_area_px = int(inclusion_mask.sum()) if inclusion_mask is not None else 0
            inc_frac = (inc_area_px / gem_area_px) if gem_area_px > 0 else 0.0

            bbox = axis_aligned_bbox_from_mask(gem_mask)
            x0, y0, x1, y1, w_px, h_px = bbox if bbox else (0, 0, 0, 0, 0, 0)

            pca = pca_sizes_from_mask(gem_mask)
            major_px, minor_px = pca if pca else (float(w_px), float(h_px))

            results.append(
                {
                    "filename": f"Captured Image {idx + 1}",
                    "overlay": overlay,
                    "gem_name": gem_name,
                    "inc_percent": inc_frac * 100,
                    "width_px": w_px,
                    "height_px": h_px,
                    "major_px": major_px,
                    "minor_px": minor_px,
                    "error": None,
                }
            )

            summary_results.append(
                {
                    "gem_name": gem_name,
                    "inc_percent": inc_frac * 100,
                    "width_px": w_px,
                    "height_px": h_px,
                }
            )

    st.session_state["segmentation_results"] = results

    if summary_results:
        gem_class_counts = defaultdict(int)

        for item in summary_results:
            gem_class_counts[item["gem_name"]] += 1

        most_common_gem_class = max(gem_class_counts.items(), key=lambda x: x[1])[0]
        avg_inclusion_percent = sum(item["inc_percent"] for item in summary_results) / len(summary_results)
        avg_width_px = sum(item["width_px"] for item in summary_results) / len(summary_results)
        avg_height_px = sum(item["height_px"] for item in summary_results) / len(summary_results)

        st.session_state["segmentation_summary"] = {
            "gem_class": most_common_gem_class,
            "avg_inclusion_percent": avg_inclusion_percent,
            "avg_width_px": avg_width_px,
            "avg_height_px": avg_height_px,
        }
    else:
        st.session_state["segmentation_summary"] = None

def _render_3d_section(
    model,
    conf,
    name_map,
    inclusion_labels,
    distance_cm,
    hfov_deg,
    script_dir,
    cut_bg_image_path=None,
):
    header_col, clear_col = st.columns([8, 1])

    with header_col:
        st.markdown('<div class="section-title">📦 3D Model Generation</div>', unsafe_allow_html=True)

    with clear_col:
        if st.session_state["three_d_glb_data"] is not None or st.session_state["three_d_cut_fig"] is not None:
            if st.button("✖", key="clear_3d_result", help="Clear 3D results"):
                _clear_3d_results()
                st.rerun()

    with st.container(border=True):
        upload_col, run_col = st.columns([1.4, 0.6], gap="large")

        with upload_col:
            uploaded_3d_file = st.file_uploader(
                "Upload Gem Image for 3D Reconstruction",
                type=["png", "jpg", "jpeg"],
                key="3d_upload",
            )

        with run_col:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_clicked = st.button(
                "Generate 3D Model",
                type="primary",
                width="stretch",
                disabled=uploaded_3d_file is None,
                key="generate_3d_btn",
            )

        if uploaded_3d_file is not None:
            st.caption(f"Selected for 3D generation: {uploaded_3d_file.name}")

            image_bytes = uploaded_3d_file.getvalue()
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

        if generate_clicked:
            motor_started = send_motor_signal("START")

            if motor_started:
                st.toast("Motor Started", icon="⚙️")

            try:
                _run_3d_generation(
                    uploaded_file=uploaded_3d_file,
                    model=model,
                    conf=conf,
                    name_map=name_map,
                    inclusion_labels=inclusion_labels,
                    distance_cm=distance_cm,
                    hfov_deg=hfov_deg,
                    script_dir=script_dir,
                    cut_bg_image_path=cut_bg_image_path,
                )
            finally:
                if motor_started:
                    send_motor_signal("STOP")
                    st.toast("Motor Stopped", icon="🛑")

    if st.session_state["three_d_glb_data"] is not None:
        st.success("✅ 3D Model Ready!")
        render_glb_bytes(st.session_state["three_d_glb_data"], "Generated 3D Gem")

    if st.session_state["three_d_cut_fig"] is not None or st.session_state["three_d_cut_message"] is not None:
        st.markdown("#### Optimized Cut Shape")

        if st.session_state["three_d_cut_fig"] is not None:
            if st.session_state["three_d_cut_score"] is not None:
                st.success(f"Optimized cut found. Score: {st.session_state['three_d_cut_score']:.5f}")
            st.plotly_chart(st.session_state["three_d_cut_fig"], width="stretch")

        if st.session_state["three_d_cut_message"] is not None and st.session_state["three_d_cut_fig"] is None:
            if "failed" in st.session_state["three_d_cut_message"].lower():
                st.error(st.session_state["three_d_cut_message"])
            elif "skipping" in st.session_state["three_d_cut_message"].lower():
                st.warning(st.session_state["three_d_cut_message"])
            else:
                st.info(st.session_state["three_d_cut_message"])
def truncate_text(text, max_chars=10):
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def _render_segmentation_section(
    uploaded,
    run_clicked,
    capture_webcam_clicked,
    model,
    name_map,
    inclusion_labels,
    conf,
    alpha,
    show_inclusion_fill,
    show_inclusion_outline,
    show_gem_outline,
):
    header_col, clear_col = st.columns([8, 1])

    with header_col:
        st.markdown('<div class="section-title">Segmentation Results</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Processed images, inclusion overlays, and average measurements appear here after running segmentation.</div>',
            unsafe_allow_html=True,
        )

    with clear_col:
        if st.session_state["segmentation_results"] is not None:
            if st.button("✖", key="clear_seg_result", help="Clear segmentation results"):
                _clear_segmentation_results()
                st.rerun()

    if run_clicked:
        _run_segmentation(
            uploaded_files=uploaded,
            model=model,
            name_map=name_map,
            inclusion_labels=inclusion_labels,
            conf=conf,
            alpha=alpha,
            show_inclusion_fill=show_inclusion_fill,
            show_inclusion_outline=show_inclusion_outline,
            show_gem_outline=show_gem_outline,
        )

    if capture_webcam_clicked:
        st.markdown("### Webcam Capture")

        motor_started = send_motor_signal("START")

        if motor_started:
            st.toast("Motor Started", icon="⚙️")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Webcam not available")

            if motor_started:
                send_motor_signal("STOP")
                st.toast("Motor Stopped", icon="🛑")

            cap.release()
        else:
            with st.container(border=True):
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
                    if motor_started:
                        send_motor_signal("STOP")
                        st.toast("Motor Stopped", icon="🛑")
                    cap.release()

                st.success("Capture Complete!")
                st.session_state["captured_frames"] = captured_frames

                _run_segmentation_on_captured_frames(
                    captured_frames=captured_frames,
                    model=model,
                    name_map=name_map,
                    inclusion_labels=inclusion_labels,
                    conf=conf,
                    alpha=alpha,
                    show_inclusion_fill=show_inclusion_fill,
                    show_inclusion_outline=show_inclusion_outline,
                    show_gem_outline=show_gem_outline,
                )

    results = st.session_state["segmentation_results"]
    summary = st.session_state["segmentation_summary"]

    if results is None:
        st.markdown(
            """
            <div class="empty-state">
                <h4 style="margin:0 0 0.4rem 0; color:#334155;">No segmentation run yet</h4>
                <p style="margin:0;">Upload gem images on the right, adjust settings, then click <b>Run Segmentation</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for row_start in range(0, len(results), 2):
        cols = st.columns(2, gap="large")

        for col_idx in range(2):
            idx = row_start + col_idx
            if idx >= len(results):
                continue

            item = results[idx]

            with cols[col_idx]:
                with st.container(border=True):
                    short_filename = truncate_text(item['filename'], 10)
                    st.markdown(f"#### {short_filename}")

                    if item["error"]:
                        st.error(item["error"])
                    else:
                        st.image(item["overlay"], width="stretch")

    if summary:
        st.divider()
        st.markdown('<div class="section-title">Average Summary</div>', unsafe_allow_html=True)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Gem class", summary["gem_class"])
        a2.metric("Avg Inclusion", f"{summary['avg_inclusion_percent']:.2f}%")
        a3.metric("Avg Width", f"{summary['avg_width_px']:.0f}px")
        a4.metric("Avg Height", f"{summary['avg_height_px']:.0f}px")


def render_advanced_analysis_page():
    _inject_page_styles()
    _init_state()
    render_hardware_status()

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(os.getcwd())

    weights_path = str((script_dir / "best.pt").resolve())

    # Put your background image here later if you want.
    # If the file is missing, the cut section will automatically use light gray.
    cut_bg_image_path = str((script_dir / "assets" / "cut_bg.png").resolve())

    workspace_col, controls_col = st.columns([2.15, 1], gap="large")

    with controls_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">1. Upload & Detection Settings</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Choose images and tune the YOLO confidence threshold.</div>',
                unsafe_allow_html=True,
            )

            if not Path(weights_path).exists():
                st.error(f"Couldn't find `best.pt` at `{weights_path}`. Place it next to this script.")
                st.stop()

            model = load_model_cached(weights_path)
            name_map = get_name_map(model)
            all_class_names = [name_map[i] for i in sorted(name_map.keys())]

            kws = ["inclusion", "fracture", "crack", "feather", "cavity", "cloud", "needle"]
            suggested_inclusions = [n for n in all_class_names if any(k in n.lower() for k in kws)]
            inclusion_labels = suggested_inclusions

            uploaded = st.file_uploader(
                "Gem images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Upload one or more gem images for inclusion and dimension analysis.",
            )

            if uploaded:
                st.caption(f"{len(uploaded)} image(s) selected")
            else:
                st.caption("No images selected yet")

            conf = st.slider(
                "Confidence threshold",
                0.05,
                0.95,
                0.25,
                0.01,
                help="Lower values detect more objects but may add false positives.",
            )

        with st.container(border=True):
            st.markdown('<div class="section-title">2. Camera Calibration</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Used for approximate physical dimension estimation.</div>',
                unsafe_allow_html=True,
            )

            distance_cm = st.number_input(
                "Distance camera → gem (cm)",
                min_value=0.1,
                value=30.0,
                step=0.5,
            )

            with st.expander("Advanced calibration", expanded=False):
                hfov_deg = st.number_input(
                    "Camera Horizontal FOV (deg)",
                    min_value=20.0,
                    max_value=120.0,
                    value=60.0,
                    step=1.0,
                    help="Default 60° is used when the exact camera FOV is unknown.",
                )

        with st.container(border=True):
            st.markdown('<div class="section-title">3. Overlay Display</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Control how segmentation masks are shown on the image.</div>',
                unsafe_allow_html=True,
            )

            show_inclusion_fill = st.checkbox("Show inclusion fill", value=True)
            show_inclusion_outline = st.checkbox("Show inclusion outline", value=True)
            show_gem_outline = st.checkbox("Show gem outline", value=True)
            alpha = st.slider("Inclusion fill opacity", 0.05, 0.85, 0.35, 0.01)

        with st.container(border=True):
            st.markdown('<div class="section-title">4. Actions</div>', unsafe_allow_html=True)
            run_clicked = st.button("▶️ Run Segmentation", type="primary", width="stretch")
            capture_webcam_clicked = st.button("📸 Capture 36 Images from Webcam", width="stretch")

    with workspace_col:
        _render_3d_section(
            model=model,
            conf=conf,
            name_map=name_map,
            inclusion_labels=inclusion_labels,
            distance_cm=distance_cm,
            hfov_deg=hfov_deg,
            script_dir=script_dir,
            cut_bg_image_path=cut_bg_image_path,
        )

        st.divider()

        _render_segmentation_section(
            uploaded=uploaded,
            run_clicked=run_clicked,
            capture_webcam_clicked=capture_webcam_clicked,
            model=model,
            name_map=name_map,
            inclusion_labels=inclusion_labels,
            conf=conf,
            alpha=alpha,
            show_inclusion_fill=show_inclusion_fill,
            show_inclusion_outline=show_inclusion_outline,
            show_gem_outline=show_gem_outline,
        )