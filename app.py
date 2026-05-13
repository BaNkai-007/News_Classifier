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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');

    :root {
        --bg: #0a0a0a;
        --surface: #111111;
        --accent: #00d4ff;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif;
    }

    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        color: white;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #888;
        font-size: 1.1rem;
        margin-top: -0.8rem;
        margin-bottom: 2rem;
    }

    /* Input Area - News Draft Style */
    div[data-baseweb="textarea"] {
        background: #111111 !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    textarea {
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        color: #eee !important;
    }

    /* Metrics - Clean News Cards */
    .metric-card {
        background: #111111;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.4rem 1.3rem;
        text-align: center;
        min-height: 118px;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        color: #888;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 600;
        color: white;
    }

    /* Result Cards */
    .result-card {
        background: #111111;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .rc-label {
        flex: 1;
        font-size: 1.05rem;
        color: white;
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
        background: #00d4ff;
        border-radius: 999px;
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
    st.title("News Classifier")
    
    st.markdown(f"""
    **Scans this session**  **{st.session_state.total_scans}**  
    **Total inference**  **{st.session_state.total_time:.2f}s**
    """)

    st.markdown("---")
    st.markdown("**Parameters**")
    threshold = st.slider("Confidence threshold", 0.3, 0.95, 0.4, 0.05)
    top_n = st.slider("Max results", 3, 15, 8)
    multi_label = st.toggle("Multi-label mode", value=True)

    st.markdown("---")
    st.markdown("**Active Labels**")
    selected_labels = st.multiselect(
        "Labels", 
        ALL_LABELS, 
        default=ALL_LABELS[:20],
        label_visibility="collapsed"
    )


# MAIN UI
st.title("News Classifier")
st.markdown('<p class="subtitle">Zero-shot text classification for news &amp; articles</p>', unsafe_allow_html=True)

col_in, col_out = st.columns([1.05, 0.95], gap="large")

with col_in:
    st.markdown("**Input**")
    text_input = st.text_area(
        "Input",
        height=380,
        placeholder="Paste your news article, headline, or paragraph here...",
        label_visibility="collapsed"
    )

    b1, b2 = st.columns([2, 1])
    with b1:
        analyze_btn = st.button("Classify Article", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()


# RESULTS
with col_out:
    st.markdown("**Results**")

        if not analyze_btn:
        st.info("👉 Run classification to see results")

    if analyze_btn and text_input.strip() and selected_labels:
        with st.spinner("Analyzing article..."):
            pairs, elapsed = run_single(
                text_input, selected_labels, threshold, top_n, multi_label
            )

        st.session_state.total_scans += 1
        st.session_state.total_time += elapsed

        if pairs:
            top_label, top_score = pairs[0]

            # Professional Metrics
            c1, c2, c3 = st.columns([1.15, 0.95, 0.95])

            c1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">TOP CATEGORY</div>
                    <div style="font-size:1.4rem; font-weight:600; line-height:1.3;">{top_label}</div>
                </div>
            """, unsafe_allow_html=True)

            c2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CONFIDENCE</div>
                    <div style="color:#00d4ff; font-size:1.85rem; font-weight:600;">{top_score:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

            c3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">INFERENCE TIME</div>
                    <div style="font-size:1.85rem; font-weight:600;">{elapsed}s</div>
                </div>
            """, unsafe_allow_html=True)

            # Bar Chart
            df = pd.DataFrame({
                "Category": [p[0] for p in pairs],
                "Score": [p[1] for p in pairs]
            }).sort_values("Score")

            fig = go.Figure(go.Bar(
                x=df["Score"], 
                y=df["Category"], 
                orientation="h",
                marker=dict(color="#00d4ff", opacity=0.85),
                text=[f"{s:.1%}" for s in df["Score"]],
                textposition="outside"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(200, len(pairs) * 42),
                margin=dict(l=20, r=60, t=10, b=10),
                xaxis=dict(range=[0, 1.1], tickformat=".0%"),
                yaxis=dict(tickfont=dict(size=13))
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Results
            st.markdown("**All Matches**")
            for label, score in pairs:
                st.markdown(f"""
                <div class="result-card">
                    <div class="rc-label">{label}</div>
                    <div class="rc-bar-wrap">
                        <div class="rc-bar" style="width: {score*100}%"></div>
                    </div>
                    <div style="font-family: monospace; min-width: 65px; text-align: right;">{score:.3f}</div>
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

            # Save to history
            st.session_state.history.append({
                "label": top_label,
                "score": top_score,
                "text": text_input[:80] + "…" if len(text_input) > 80 else text_input
            })
