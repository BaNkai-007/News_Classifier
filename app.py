import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

from model import ALL_LABELS, confidence_tier, load_model, run_single


# PAGE CONFIG
st.set_page_config(
    page_title="News Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #050505;
        --surface: #0f0f0f;
        --glass: rgba(20, 20, 30, 0.75);
        --accent: #00f0ff;
        --text: #f0f0f0;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Hero Title */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -0.06em;
        background: linear-gradient(90deg, #ffffff, var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }

    .tagline {
        color: #888;
        font-size: 1.15rem;
        margin-top: -0.8rem;
        margin-bottom: 2.2rem;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .glass {
        background: var(--glass);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(12px);
    }

    /* Input Area */
    div[data-baseweb="textarea"] {
        background: #111 !important;
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        border-radius: 16px !important;
    }
    textarea {
        font-size: 1.15rem !important;
        line-height: 1.75 !important;
    }

    /* Metrics */
    .metric-card {
        background: var(--glass);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--accent);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.78rem;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 600;
    }

    /* Result Cards */
    .result-card {
        background: var(--glass);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.2s ease;
    }
    .result-card:hover {
        border-color: var(--accent);
    }
    .rc-label {
        flex: 1;
        font-size: 1.08rem;
        color: white;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: #0a0a0a !important;
    }

    #MainMenu, footer { visibility: hidden !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE
for key in ["history", "total_scans", "total_time"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else 0


# SIDEBAR
with st.sidebar:
    st.markdown("# News Classifier")
    st.caption("AI-Powered Zero-Shot Analysis")

    st.markdown("---")
    st.markdown("**Session Stats**")
    st.markdown(f"**Scans**  {st.session_state.total_scans}")
    st.markdown(f"**Total Time**  {st.session_state.total_time:.2f}s")

    st.markdown("---")
    st.markdown("**Parameters**")
    threshold = st.slider("Confidence threshold", 0.3, 0.95, 0.4, 0.05)
    top_n = st.slider("Max results", 3, 15, 8)
    multi_label = st.toggle("Multi-label mode", value=True)

    st.markdown("---")
    st.markdown("**Active Labels**")
    selected_labels = st.multiselect(
        "Select labels", 
        ALL_LABELS, 
        default=ALL_LABELS[:18],
        label_visibility="collapsed"
    )


# MAIN LAYOUT
st.title("News Classifier")
st.markdown('<p class="tagline">Instant AI understanding of any news article</p>', unsafe_allow_html=True)

col_in, col_out = st.columns([1.05, 0.95], gap="large")

with col_in:
    st.markdown("**Paste your article**")
    text_input = st.text_area(
        "Input",
        height=420,
        placeholder="Breaking: Apple has acquired a major stake in OpenAI...",
        label_visibility="collapsed"
    )

    b1, b2 = st.columns([2, 1])
    with b1:
        analyze_btn = st.button("🚀 Analyze Article", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

with col_out:
    st.markdown("**Analysis Results**")

    if not analyze_btn:
        st.markdown("""
        <div style="background:rgba(15,15,25,0.6); border:1px solid rgba(0,240,255,0.15); 
                    border-radius:16px; padding:3rem 2rem; text-align:center; color:#666;">
            <h3 style="margin:0 0 1rem 0; opacity:0.7;">Your analysis will appear here</h3>
            <p style="margin:0; font-size:1.1rem;">Run the classifier on any news text</p>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and text_input.strip() and selected_labels:
        with st.spinner("Analyzing with AI..."):
            pairs, elapsed = run_single(text_input, selected_labels, threshold, top_n, multi_label)

        st.session_state.total_scans += 1
        st.session_state.total_time += elapsed

        if pairs:
            top_label, top_score = pairs[0]

            # Creative Metrics
            c1, c2, c3 = st.columns([1.15, 0.95, 0.95])
            c1.markdown(f"""
                <div class="metric-card glass">
                    <div class="metric-label">TOP CATEGORY</div>
                    <div style="font-size:1.45rem; font-weight:600; line-height:1.3;">{top_label}</div>
                </div>
            """, unsafe_allow_html=True)

            c2.markdown(f"""
                <div class="metric-card glass">
                    <div class="metric-label">CONFIDENCE</div>
                    <div style="color:#00f0ff; font-size:2rem; font-weight:700;">{top_score:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

            c3.markdown(f"""
                <div class="metric-card glass">
                    <div class="metric-label">INFERENCE TIME</div>
                    <div style="font-size:1.9rem; font-weight:600;">{elapsed}s</div>
                </div>
            """, unsafe_allow_html=True)

            # Bar Chart
            df = pd.DataFrame({"Category": [p[0] for p in pairs], "Score": [p[1] for p in pairs]}).sort_values("Score")
            fig = go.Figure(go.Bar(
                x=df["Score"], y=df["Category"], orientation="h",
                marker=dict(color="#00f0ff", opacity=0.9),
                text=[f"{s:.1%}" for s in df["Score"]],
                textposition="outside"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=max(220, len(pairs) * 45),
                margin=dict(l=20, r=60, t=20, b=10),
                xaxis=dict(range=[0, 1.15], tickformat=".0%")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Results
            st.markdown("**Detailed Breakdown**")
            for label, score in pairs:
                st.markdown(f"""
                <div class="result-card glass">
                    <div class="rc-label">{label}</div>
                    <div style="flex:1.6; background:#222; height:7px; border-radius:999px; overflow:hidden;">
                        <div style="height:100%; width:{score*100}%; background:#00f0ff;"></div>
                    </div>
                    <div style="font-family:monospace; min-width:70px; text-align:right; color:#00f0ff;">
                        {score:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Export
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "input_preview": text_input[:300] + "…" if len(text_input) > 300 else text_input,
                "top_category": top_label,
                "confidence": round(top_score, 4),
                "inference_time_s": elapsed,
                "results": [{"category": l, "score": round(s, 4)} for l, s in pairs]
            }

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button("Export JSON", 
                    data=json.dumps(export_data, indent=2),
                    file_name=f"news_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json", use_container_width=True)
            with col_exp2:
                csv_data = pd.DataFrame(export_data["results"]).to_csv(index=False)
                st.download_button("Export CSV", 
                    data=csv_data,
                    file_name=f"news_classification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True)

            st.session_state.history.append({"label": top_label, "score": top_score, "text": text_input})
