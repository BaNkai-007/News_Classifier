import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

from model import ALL_LABELS, confidence_tier, load_model, run_single


st.set_page_config(page_title="News Classifier", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=DM+Mono&display=swap');
    html, body, .stApp { background: #0a0a0a; color: #f0f0f0; font-family: 'Inter', sans-serif; }
    h1 { font-family: 'DM Mono', monospace; letter-spacing: -0.02em; }
    .result-card {
        background: #171717; border: 1px solid #333; border-radius: 10px;
        padding: 1rem 1.3rem; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 1rem;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session State
for key in ["history", "total_scans", "total_time"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else 0

# Sidebar
with st.sidebar:
    st.title("News Classifier")
    st.markdown(f"**Scans:** {st.session_state.total_scans}  \n**Total Time:** {st.session_state.total_time:.2f}s")

    st.markdown("---")
    threshold = st.slider("Confidence threshold", 0.3, 0.95, 0.4, 0.05)
    top_n = st.slider("Max results", 3, 15, 8)
    multi_label = st.toggle("Multi-label", True)

    st.markdown("---")
    selected_labels = st.multiselect("Active Labels", ALL_LABELS, default=ALL_LABELS[:15])

# Main
st.title("News Classifier")
col_in, col_out = st.columns([1, 1])

with col_in:
    text_input = st.text_area("Input Text", height=300, placeholder="Paste news text here...")

    b1, b2 = st.columns(2)
    with b1:
        run_btn = st.button("Run Classification", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

with col_out:
    st.subheader("Results")

    if run_btn and text_input.strip():
        with st.spinner("Classifying..."):
            pairs, elapsed = run_single(text_input, selected_labels, threshold, top_n, multi_label)

        st.session_state.total_scans += 1
        st.session_state.total_time += elapsed

        if pairs:
            top_label, top_score = pairs[0]

            # Metrics
            c1, c2, c3 = st.columns([1.1, 0.95, 0.95])
            c1.markdown(f"**Top Category**  \n{top_label}")
            c2.markdown(f"**Confidence**  \n**{top_score:.1%}**")
            c3.markdown(f"**Time**  \n{elapsed}s")

            # You can add chart and cards later if needed

            st.session_state.history.append({"label": top_label, "score": top_score, "text": text_input})

    else:
        st.info("Run classification to see results")
