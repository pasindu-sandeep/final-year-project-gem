import os
import torch
from pathlib import Path
import streamlit as st

# --- INITIALIZE PAGE CONFIG FIRST ---
st.set_page_config(
    page_title="Gem Authenticator System",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IMPORT MODULES ---
from design import inject_css
from hardware_serial import read_serial_messages
from ui_hardware_controls import render_hardware_page
from ui_ai_authentication import render_ai_auth_page
from ui_advanced_analysis import render_advanced_analysis_page

# --- IMPORT SPECTRUM MODULES ---
from spectrum import load_model, run_inference, decode_image

# =========================================================
# SPECTRUM MODEL SETUP
# =========================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

MODEL_PATH = str(SCRIPT_DIR / "gem_classifier_bundle.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading spectrum model...")
def _load_model():
    return load_model(MODEL_PATH, DEVICE)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
default_states = {
    "ser": None,
    "connected_port": None,
    "hardware_status": "Ready",
    "page": "Hardware Controls",
    "theme_mode": "Dark",
    "last_ai_result": None,
    "auto_captured_paths": [],
    "auto_capture_log": [],
    "last_auto_run_dir": None,
    "show_3d_ui": False,
    "last_spec_result": None  # Added for the spectrum component
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Inject the UI based on the selected theme
inject_css(st.session_state.theme_mode)

# =========================================================
# HEADER & NAVIGATION
# =========================================================
st.markdown(
    """
    <div class="main-header">
        <h1>💎 Gem Authenticator System</h1>
        <p>Hardware control + AI-based gem authentication + 3D Analysis + Spectrum</p>
    </div>
    """, unsafe_allow_html=True
)

# Adjusted column widths slightly to accommodate the 4th navigation item
top_left, top_mid, top_right = st.columns([2, 1, 1])

with top_left:
    st.session_state.page = st.radio(
        "Navigation", 
        ["Hardware Controls", "AI Authentication", "Advanced 3D & Inclusions", "Spectrum Analysis"], 
        horizontal=True, 
        label_visibility="collapsed"
    )

with top_mid:
    theme = st.radio(
        "Theme", 
        ["Dark", "Light"], 
        horizontal=True, 
        index=0 if st.session_state.theme_mode == "Dark" else 1, 
        label_visibility="collapsed"
    )
    if theme != st.session_state.theme_mode:
        st.session_state.theme_mode = theme
        st.rerun()

with top_right:
    st.markdown(
        f'<div class="status-pill">Status: {st.session_state.hardware_status}</div>', 
        unsafe_allow_html=True
    )

# Always try to read hardware messages globally
read_serial_messages()

# =========================================================
# PAGE ROUTING
# =========================================================
if st.session_state.page == "Hardware Controls":
    render_hardware_page()
elif st.session_state.page == "AI Authentication":
    render_ai_auth_page()
elif st.session_state.page == "Advanced 3D & Inclusions":
    render_advanced_analysis_page()
elif st.session_state.page == "Spectrum Analysis":
    # --- NEW SPECTRUM UI INTEGRATED DIRECTLY ---
    st.markdown(
        """
        <style>
          .gradient-header {
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
            color: white; padding: 18px 24px; border-radius: 18px;
            box-shadow: 0 10px 24px rgba(99,102,241,0.3);
            margin-bottom: 20px;
          }
          .subtle { color: #e2e8f0; font-size: 0.9rem; margin-top: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gradient-header">
          <h2 style="margin:0; color:white;">📊 Gem Spectrum Analysis</h2>
          <div class="subtle">Deep learning powered identification for gem absorption/emission spectra.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, class_names, model_err = _load_model()
    if model_err:
        st.error(f"Model error: {model_err}")
        st.stop()

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.subheader("Input Source")
        input_type = st.radio("Choose input method:", ["📤 File Upload", "📸 Real-time Capture"])

        if input_type == "📤 File Upload":
            uploaded_spectrum = st.file_uploader(
                "Upload Spectrum Image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="spec_up",
            )
            if uploaded_spectrum:
                st.image(uploaded_spectrum, caption="Uploaded Image", use_container_width=True)
                if st.button("🚀 Analyze Uploaded Spectrum", use_container_width=True):
                    with st.spinner("Classifying..."):
                        try:
                            img = decode_image(uploaded_spectrum.getvalue())
                            result, err = run_inference(model, class_names, img, DEVICE)
                            if err:
                                st.error(f"Error: {err}")
                            else:
                                st.session_state.last_spec_result = result
                        except Exception as e:
                            st.error(f"Image decode error: {e}")
        else:
            cam_image = st.camera_input("Capture Spectrum Image")
            if cam_image:
                if st.button("🚀 Analyze Captured Image", use_container_width=True):
                    with st.spinner("Classifying..."):
                        try:
                            img = decode_image(cam_image.getvalue())
                            result, err = run_inference(model, class_names, img, DEVICE)
                            if err:
                                st.error(f"Error: {err}")
                            else:
                                st.session_state.last_spec_result = result
                        except Exception as e:
                            st.error(f"Image decode error: {e}")

    with col_res:
        st.subheader("Classification Results")
        if st.session_state.last_spec_result:
            data = st.session_state.last_spec_result
            st.success(f"### Predicted Class: {data['predicted_class']}")
            st.metric("Confidence", f"{data['confidence']}%")
            st.markdown("#### Probability Distribution")
            for cls, score in data["all_scores"].items():
                st.progress(score / 100, text=f"{cls}: {score}%")
        else:
            st.info("Upload or capture an image to see analysis results.")