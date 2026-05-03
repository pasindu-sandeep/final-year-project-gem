import streamlit as st

def inject_css(theme="Dark"):
    # Define colors based on theme
    if theme == "Dark":
        bg_color = "#0a0e17"
        bg_image = "radial-gradient(circle at 15% 50%, rgba(0, 243, 255, 0.05), transparent 25%), radial-gradient(circle at 85% 30%, rgba(188, 19, 254, 0.05), transparent 25%)"
        text_color = "#e2e8f0"
        card_bg = "rgba(15, 23, 42, 0.7)"
        card_border = "rgba(0, 243, 255, 0.2)"
        input_bg = "rgba(0, 0, 0, 0.5)"
        btn_bg = "rgba(0, 243, 255, 0.05)"
        btn_border = "#00f3ff"
        btn_text = "#00f3ff"
        btn_hover_bg = "#00f3ff"
        btn_hover_text = "#000000"
        btn_glow = "rgba(0, 243, 255, 0.4)"
        accent_purple = "#bc13fe"
    else:
        bg_color = "#f0f4f8"
        bg_image = "radial-gradient(circle at 15% 50%, rgba(37, 99, 235, 0.05), transparent 25%), radial-gradient(circle at 85% 30%, rgba(124, 58, 237, 0.05), transparent 25%)"
        text_color = "#0f172a"
        card_bg = "rgba(255, 255, 255, 0.8)"
        card_border = "rgba(37, 99, 235, 0.2)"
        input_bg = "rgba(255, 255, 255, 0.9)"
        btn_bg = "rgba(37, 99, 235, 0.05)"
        btn_border = "#2563eb"
        btn_text = "#2563eb"
        btn_hover_bg = "#2563eb"
        btn_hover_text = "#ffffff"
        btn_glow = "rgba(37, 99, 235, 0.3)"
        accent_purple = "#7c3aed"

    css = f"""
    <style>
    /* Base App */
    .stApp {{
        background-color: {bg_color} !important;
        background-image: {bg_image} !important;
        color: {text_color} !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* Header */
    .main-header {{
        background: {"linear-gradient(90deg, #0f172a 0%, #1e293b 100%)" if theme == "Dark" else "linear-gradient(90deg, #ffffff 0%, #e2e8f0 100%)"};
        border-left: 4px solid {btn_border};
        border-right: 4px solid {accent_purple};
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,{"0.4" if theme=="Dark" else "0.1"});
        margin-bottom: 24px;
    }}
    .main-header h1 {{ margin: 0; font-size: 2.2rem; color: {"#ffffff" if theme=="Dark" else "#0f172a"}; text-shadow: 0 0 10px {btn_glow}; }}
    .main-header p {{ margin-top: 5px; color: {"#94a3b8" if theme=="Dark" else "#475569"}; font-size: 1rem; }}

    /* Glass Cards */
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,{"0.3" if theme=="Dark" else "0.05"});
        margin-bottom: 20px;
        color: {text_color};
    }}

    /* Fix Streamlit Overrides */
    .stMarkdown, .stText, p, label, span, div, h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}

    /* Buttons */
    div.stButton > button {{
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        border: 1px solid {btn_border} !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        box-shadow: 0 0 10px {btn_glow} !important;
        transition: all 0.3s ease-in-out !important;
    }}
    div.stButton > button:hover {{
        background-color: {btn_hover_bg} !important;
        color: {btn_hover_text} !important;
        box-shadow: 0 0 20px {btn_glow} !important;
    }}

    /* Warning/Danger Buttons */
    div.stButton > button:contains("Disconnect"), 
    div.stButton > button:contains("STOP"),
    div.stButton > button:contains("Clear") {{
        color: #ff4d4d !important;
        border-color: #ff4d4d !important;
        box-shadow: 0 0 10px rgba(255, 77, 77, 0.2) !important;
    }}
    div.stButton > button:contains("Disconnect"):hover, 
    div.stButton > button:contains("STOP"):hover,
    div.stButton > button:contains("Clear"):hover {{
        background-color: #ff4d4d !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(255, 77, 77, 0.6) !important;
    }}

    /* FIX THE WHITE BOXES - Inputs & Uploaders */
    .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {input_bg} !important;
        border: 1px dashed {btn_border} !important;
        border-radius: 8px !important;
    }}
    /* Force text inside uploader to match theme */
    [data-testid="stFileUploaderDropzone"] *, 
    .stSelectbox div[data-baseweb="select"] * {{
        color: {text_color} !important;
    }}

    /* Status Pills & Tags */
    .status-pill {{
        display: inline-block; padding: 8px 16px; border-radius: 20px;
        background: {"rgba(188, 19, 254, 0.1)" if theme=="Dark" else "rgba(124, 58, 237, 0.1)"};
        color: {accent_purple};
        border: 1px solid {accent_purple};
        font-weight: bold; font-size: 0.9rem;
    }}
    .mini-tag {{
        display: inline-block; padding: 4px 10px; border-radius: 12px;
        background: {"rgba(255,255,255,0.05)" if theme=="Dark" else "rgba(0,0,0,0.05)"};
        border: 1px solid {"rgba(255,255,255,0.1)" if theme=="Dark" else "rgba(0,0,0,0.1)"};
        font-size: 0.8rem; margin-right: 6px; margin-bottom: 6px;
    }}

    /* Results */
    .result-good {{
        background: rgba(20, 184, 106, 0.1); color: #14b86a;
        border-left: 4px solid #14b86a; padding: 16px; border-radius: 8px; font-weight: bold; font-size: 1.1rem;
    }}
    .result-bad {{
        background: rgba(255, 77, 79, 0.1); color: #ff4d4f;
        border-left: 4px solid #ff4d4f; padding: 16px; border-radius: 8px; font-weight: bold; font-size: 1.1rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)