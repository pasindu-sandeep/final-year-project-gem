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
    "show_3d_ui": False
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
        <p>Hardware control + AI-based gem authentication + 3D Analysis</p>
    </div>
    """, unsafe_allow_html=True
)

top_left, top_mid, top_right = st.columns([1.5, 1, 1])

with top_left:
    st.session_state.page = st.radio(
        "Navigation", 
        ["Hardware Controls", "AI Authentication", "Advanced 3D & Inclusions"], 
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
else:
    render_advanced_analysis_page()