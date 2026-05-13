import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

from model import (
    ALL_LABELS,
    confidence_tier,
    load_model,
    run_single,
)


# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Zero-Shot News Classifier"}
)


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:         #0a0a0a;
    --bg2:        #111111;
    --surface:    #171717;
    --border:     #333333;
    --border2:    #444444;
    --accent:     #00ff9d;
}

html, body, .stApp { background: var(--bg) !important; color: #f0f0f0 !important; font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }

h1 { font-family: 'Instrument Serif', serif; font-size: 2.8rem !important; letter-spacing: -0.04em; color: white; }

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rc-label { color: white; font-size: 1.05rem; flex: 1; word-break: break-word; }

/* Sidebar toggle fix */
#MainMenu, footer { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }
button[aria-label="Collapse sidebar"] { visibility: visible !important; opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [("history", []), ("total_scans", 0), ("total_time", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='color:#00ff9d; margin:0;'>News Classifier</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:0.9rem 1.1rem;margin:1.2rem 0;'>
        Scans: <span style='float:right;color:#00ff9d;'>{st.session_state.total_scans}</span><br>
        Total Time: <span style='float:right;color:#aaa;'>{st.session_state.total_time:.2f}s</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Parameters**")
    threshold = st.slider("Confidence threshold", 0.30, 0.95, 0.40, 0.05)
    top_n = st.slider("Max results", 3, 15, 8)
    multi_label = st.toggle("Multi-label mode", value=True)

    st.markdown("---")
    st.markdown("**Active Labels**")
    selected_labels = st.multiselect("Labels", ALL_LABELS, default=ALL_LABELS[:30], label_visibility="collapsed")

    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


# ── MAIN APP ─────────────────────────────────────────────────────────────────
st.title("News Classifier")
st.markdown("**Zero-shot text classification**")

col_in, col_out = st.columns([1.05, 0.95], gap="large")

with col_in:
    text_input = st.text_area(
        "Input",
        height=340,
        placeholder="Paste your text here...",
        label_visibility="collapsed"
    )

    b1, b2 = st.columns([2, 1])
    with b1:
        analyze_btn = st.button("Run Classification", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

with col_out:
    st.markdown("**Results**")

    if not analyze_btn:
        st.info("👈 Run a classification to see results")

    if analyze_btn and text_input.strip() and selected_labels:
        with st.spinner("Classifying..."):
            pairs, elapsed = run_single(text_input, selected_labels, threshold, top_n, multi_label)

        st.session_state.total_scans += 1
        st.session_state.total_time += elapsed

        if pairs:
            top_label, top_score = pairs[0]

            # Fixed Metrics
            c1, c2, c3 = st.columns([1.15, 0.95, 0.95])
            c1.markdown(f"""<div style="background:#171717;border:1px solid #333;border-radius:12px;padding:1.3rem;min-height:125px;">
                <div style="color:#888;font-size:0.78rem;">TOP CATEGORY</div>
                <div style="color:white;font-size:1.25rem;margin-top:8px;line-height:1.3;">{top_label}</div>
            </div>""", unsafe_allow_html=True)

            c2.markdown(f"""<div style="background:#171717;border:1px solid #333;border-radius:12px;padding:1.3rem;min-height:125px;">
                <div style="color:#888;font-size:0.78rem;">CONFIDENCE</div>
                <div style="color:#00ff9d;font-size:1.9rem;font-weight:600;margin-top:6px;">{top_score:.1%}</div>
            </div>""", unsafe_allow_html=True)

            c3.markdown(f"""<div style="background:#171717;border:1px solid #333;border-radius:12px;padding:1.3rem;min-height:125px;">
                <div style="color:#888;font-size:0.78rem;">INFERENCE TIME</div>
                <div style="color:white;font-size:1.9rem;font-weight:600;margin-top:6px;">{elapsed}s</div>
            </div>""", unsafe_allow_html=True)

            # Bar Chart + Cards + Export (same as before)
            # ... (I kept it short here, but you can keep your previous chart + cards code)

            st.session_state.history.append({"label": top_label, "score": top_score, "text": text_input, "time": elapsed})
