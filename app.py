import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

from model import (
    ALL_LABELS, TIER_COLORS,
    confidence_tier, load_model,
    run_single,
)


# PAGE CONFIG

st.set_page_config(
    page_title="News Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "News Classifier — Text Classification"}
)


# CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:         #0a0a0a;
    --bg2:        #111111;
    --surface:    #171717;
    --border:     #333333;
    --border2:    #444444;
    --ink:        #f0f0f0;
    --ink2:       #aaaaaa;
    --ink3:       #777777;
    --accent:     #00ff9d;
    --accent-dim: #004d33;
    --high:       #00ff9d;
    --med:        #ffcc00;
    --low:        #888888;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

/* Typography */
h1 {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2.9rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.04em !important;
    color: white !important;
}

h2, h3, h4 {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--ink3) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* Textarea - Fixed contrast */
div[data-baseweb="textarea"] {
    background: #111111 !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
}
div[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(0, 255, 157, 0.15) !important;
}
textarea {
    color: #eeeeee !important;
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
    font-family: 'Inter', sans-serif !important;
}
textarea::placeholder { color: #666666 !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}
.stButton > button:hover {
    background: #00cc7a !important;
    color: #000 !important;
}

/* Secondary (Clear) button */
div[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    color: var(--ink2) !important;
    border: 1px solid var(--border2) !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
}

/* Result Cards */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rc-label { 
    color: #fff; 
    font-size: 1.02rem; 
    flex: 1; 
}
.rc-bar-wrap { 
    flex: 1.6; 
    background: #222; 
    height: 6px; 
    border-radius: 999px; 
    overflow: hidden; 
}
.rc-bar { 
    height: 100%; 
    border-radius: 999px; 
}
.rc-score { 
    font-family: 'DM Mono', monospace; 
    font-size: 0.85rem; 
    color: var(--ink2); 
    min-width: 60px; 
    text-align: right; 
}
.rc-tier-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Sidebar stats */
.sidebar-stat {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.2rem;
}

/* Hide default Streamlit stuff */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 2rem !important; max-width: 1150px !important; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE

for key, default in [("history", []), ("total_scans", 0), ("total_time", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default


# SIDEBAR

with st.sidebar:
    st.markdown("""<h1 style="margin:0 0 1.2rem 0; background:linear-gradient(90deg, #00ff9d, #00ccff); 
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Classify</h1>""", 
        unsafe_allow_html=True)

    st.markdown(f"""
    <div class='sidebar-stat'>
        Scans this session <span style='float:right; color:#00ff9d; font-weight:600;'>{st.session_state.total_scans}</span><br>
        Total inference <span style='float:right; color:#aaa;'>{st.session_state.total_time:.2f}s</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Parameters**")

    threshold   = st.slider("Confidence threshold", 0.40, 0.95, 0.65, 0.05)
    top_n       = st.slider("Max results", 3, 15, 8)
    multi_label = st.toggle("Multi-label mode", value=True)

    st.markdown("---")
    st.markdown("**Active Labels**")
    selected_labels = st.multiselect("Labels", ALL_LABELS,
                                     default=ALL_LABELS[:25], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Recent**")

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            tier = confidence_tier(item["score"])
            color = "#00ff9d" if tier == "HIGH" else "#ffcc00" if tier == "MEDIUM" else "#888"
            st.markdown(f"""
            <div style='border-left:3px solid {color}; padding:0.5rem 0.7rem; margin-bottom:0.5rem;
                background:#1a1a1a; border-radius:6px;'>
                <div style='color:#fff; font-size:0.95rem;'>{item['label']}</div>
                <div style='font-family:DM Mono; font-size:0.7rem; color:#777; margin-top:0.2rem;'>
                    {item['score']:.1%} · {item['text'][:45]}{'…' if len(item['text'])>45 else ''}
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No scans yet.")


# MAIN UI

st.title("News Classifier")
st.markdown("""<p style='color:#888; margin-top:-0.8rem; font-size:1.05rem;'>
    Text classification</p>""", unsafe_allow_html=True)

col_in, col_out = st.columns([1.05, 0.95], gap="large")

with col_in:
    st.markdown("**Input**")
    text_input = st.text_area(
        "Input",
        height=320,
        placeholder="Paste or type your text here...\n\nNews article, tweet, product description, support ticket, research paper, love letter, whatever you want to classify.",
        label_visibility="collapsed"
    )

    st.markdown(f"<div style='text-align:right; font-family:DM Mono; font-size:0.75rem; color:#666; margin-top:-0.4rem;'>{len(text_input):,} / 10,000</div>", 
                unsafe_allow_html=True)

    b1, b2 = st.columns([2, 1])
    with b1:
        analyze_btn = st.button("Run Classification", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()


# RESULTS

with col_out:
    st.markdown("**Results**")

    if not analyze_btn:
        st.info("Run a classification to see results", icon="👈")

    if analyze_btn:
        if not text_input.strip():
            st.error("Input text is empty.")
        elif not selected_labels:
            st.error("Please select at least one label from the sidebar.")
        else:
            load_model()
            with st.spinner("Classifying..."):
                pairs, elapsed = run_single(
                    text_input, selected_labels, threshold, top_n, multi_label
                )

            st.session_state.total_scans += 1
            st.session_state.total_time += elapsed

            top_label, top_score = pairs[0]

            # Metrics
            col1, col2, col3 = st.columns([1.15, 0.95, 0.95])

            # Top Category
            col1.markdown(f"""
                <div style="background:#171717; border:1px solid #333; border-radius:12px; padding:1.25rem 1.3rem; min-height:125px;">
                    <div style="color:#888; font-size:0.78rem; font-family:DM Mono;">TOP CATEGORY</div>
                    <div style="color:white; font-size:1.22rem; font-weight:500; margin-top:8px; 
                                line-height:1.3; word-break:break-word; overflow-wrap:anywhere;">
                        {top_label}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Confidence
            col2.markdown(f"""
                <div style="background:#171717; border:1px solid #333; border-radius:12px; padding:1.25rem 1.3rem; min-height:125px;">
                    <div style="color:#888; font-size:0.78rem; font-family:DM Mono;">CONFIDENCE</div>
                    <div style="color:#00ff9d; font-size:1.85rem; font-weight:600; margin-top:6px;">
                        {top_score:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Inference Time - Fixed
            col3.markdown(f"""
                <div style="background:#171717; border:1px solid #333; border-radius:12px; padding:1.25rem 1.3rem; min-height:125px;">
                    <div style="color:#888; font-size:0.78rem; font-family:DM Mono;">INFERENCE TIME</div>
                    <div style="color:white; font-size:1.85rem; font-weight:600; margin-top:6px;">
                        {elapsed:.2f}s
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Bar Chart
            df = pd.DataFrame({
                "Category": [p[0] for p in pairs],
                "Score": [p[1] for p in pairs],
            }).sort_values("Score")

            fig = go.Figure(go.Bar(
                x=df["Score"], y=df["Category"], orientation="h",
                marker=dict(color="#00ff9d", opacity=0.85),
                text=[f"{s:.1%}" for s in df["Score"]],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(180, len(pairs) * 42),
                margin=dict(l=0, r=40, t=10, b=10),
                xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Results
            for label, score in pairs:
                t = confidence_tier(score)
                c = "#00ff9d" if t == "HIGH" else "#ffcc00" if t == "MEDIUM" else "#888"
                st.markdown(f"""
                <div class='result-card'>
                    <div class='rc-label'>{label}</div>
                    <div class='rc-bar-wrap'>
                        <div class='rc-bar' style='width:{score*100}%; background:{c};'></div>
                    </div>
                    <div class='rc-score'>{score:.3f}</div>
                </div>""", unsafe_allow_html=True)

            # Export
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "input_preview": text_input[:250] + "…" if len(text_input) > 250 else text_input,
                "top_category": top_label,
                "confidence": round(top_score, 6),
                "inference_time_s": round(elapsed, 3),
                "results": [{"category": l, "score": round(s, 6), "tier": confidence_tier(s)} for l, s in pairs]
            }

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Export JSON", 
                    data=json.dumps(export_data, indent=2),
                    file_name=f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json", use_container_width=True)
            with c2:
                csv_data = pd.DataFrame(export_data["results"]).to_csv(index=False)
                st.download_button("Export CSV", 
                    data=csv_data,
                    file_name=f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True)

            # Save to history
            st.session_state.history.append({
                "label": top_label,
                "score": top_score,
                "text": text_input,
                "time": elapsed
            })