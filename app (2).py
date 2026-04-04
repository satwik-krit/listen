import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import datetime
import time
import random

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SonicSense AI",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --surface2:  #1c1c27;
    --border:    #2a2a3d;
    --accent:    #7c6aff;
    --accent2:   #00d4aa;
    --accent3:   #ff6b6b;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}
.card:hover { border-color: var(--accent); }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0d0d18 0%, #151528 50%, #0d0d18 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,106,255,.18) 0%, transparent 70%);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,170,.12) 0%, transparent 70%);
}
.hero-title {
    font-family: var(--mono);
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0 0 .5rem;
    background: linear-gradient(90deg, #fff 30%, var(--accent) 70%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    letter-spacing: .5px;
    font-weight: 300;
}

/* Metric chips */
.chip-row { display: flex; gap: .75rem; flex-wrap: wrap; margin: 1.5rem 0 0; }
.chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: .35rem 1rem;
    font-family: var(--mono);
    font-size: .75rem;
    color: var(--muted);
    letter-spacing: .5px;
}
.chip span { color: var(--accent2); font-weight: 700; }

/* Result badge */
.result-badge {
    display: inline-block;
    padding: .5rem 1.4rem;
    border-radius: 50px;
    font-family: var(--mono);
    font-size: .9rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-normal { background: rgba(0,212,170,.15); color: var(--accent2); border: 1px solid rgba(0,212,170,.4); }
.badge-anomaly { background: rgba(255,107,107,.15); color: var(--accent3); border: 1px solid rgba(255,107,107,.4); }

/* Feature value cards */
.feat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: 1rem 0; }
.feat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: all .2s;
}
.feat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feat-name { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: .3rem; font-family: var(--mono); }
.feat-val { font-family: var(--mono); font-size: 1.3rem; font-weight: 700; color: var(--text); }
.feat-bar { height: 3px; background: var(--border); border-radius: 2px; margin-top: .6rem; overflow: hidden; }
.feat-fill { height: 100%; border-radius: 2px; }

/* History table */
.hist-row {
    display: flex; align-items: center; gap: 1rem;
    padding: .75rem 1rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-bottom: .5rem;
    background: var(--surface2);
    transition: border-color .2s;
    font-size: .85rem;
}
.hist-row:hover { border-color: var(--accent); }
.hist-name { flex: 1; font-family: var(--mono); color: var(--text); font-size: .8rem; }
.hist-time { color: var(--muted); font-size: .75rem; white-space: nowrap; }
.hist-comp { background: rgba(124,106,255,.15); color: var(--accent); border-radius: 50px; padding: .2rem .7rem; font-size: .72rem; font-family: var(--mono); white-space: nowrap; }

/* Section label */
.sec-label {
    font-family: var(--mono);
    font-size: .7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 .75rem;
    display: flex; align-items: center; gap: .75rem;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Upload zone override */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] * { color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5a4fcc) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    letter-spacing: .5px !important;
    padding: .6rem 1.5rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* Plotly chart background */
.js-plotly-plot .plotly { border-radius: 12px; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: .75rem !important;
    color: var(--muted) !important;
    letter-spacing: .5px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── History persistence ─────────────────────────────────────────────────────────
HISTORY_FILE = "upload_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def add_to_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]   # keep last 50
    save_history(history)

# ── Mock feature extraction (replace with your actual processor) ────────────────
def extract_features(audio_bytes):
    """Replace this with your actual feature extraction code."""
    np.random.seed(int.from_bytes(audio_bytes[:4], 'little') % (2**32))
    return {
        "RMS Mean":           round(np.random.uniform(0.01, 0.9), 4),
        "ZCR":                round(np.random.uniform(0.05, 0.6), 4),
        "RMS Variance":       round(np.random.uniform(0.001, 0.05), 5),
        "Median Pitch":       round(np.random.uniform(80, 800), 2),
        "Spectral Centroid":  round(np.random.uniform(500, 5000), 2),
        "Spectral Rolloff":   round(np.random.uniform(1000, 8000), 2),
        "MFCC-1":             round(np.random.uniform(-200, 200), 3),
        "MFCC-2":             round(np.random.uniform(-80, 80), 3),
    }

# ── Mock models (replace with your actual loaded models) ───────────────────────
COMPONENTS = ["Valve", "Fan", "Pump", "Slider"]

def predict_component(features: dict) -> tuple[str, dict]:
    """Replace with your RandomForest model prediction."""
    vals = list(features.values())
    seed = int(abs(sum(vals)) * 1000) % (2**32)
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(len(COMPONENTS)) * 2)
    idx = int(np.argmax(probs))
    return COMPONENTS[idx], dict(zip(COMPONENTS, probs.round(3)))

def predict_anomaly(features: dict) -> tuple[bool, float]:
    """Replace with your Autoencoder anomaly detection."""
    vals = list(features.values())
    score = abs(np.sin(sum(vals))) * 0.8 + np.random.uniform(0, 0.2)
    threshold = 0.5
    return score > threshold, round(score, 4)

# ── Feature metadata ────────────────────────────────────────────────────────────
FEATURE_META = {
    "RMS Mean":          {"unit": "",     "color": "#7c6aff", "range": (0, 1)},
    "ZCR":               {"unit": "",     "color": "#00d4aa", "range": (0, 1)},
    "RMS Variance":      {"unit": "",     "color": "#ff6b6b", "range": (0, 0.05)},
    "Median Pitch":      {"unit": "Hz",   "color": "#ffa94d", "range": (0, 1000)},
    "Spectral Centroid": {"unit": "Hz",   "color": "#da77f2", "range": (0, 6000)},
    "Spectral Rolloff":  {"unit": "Hz",   "color": "#4ecdc4", "range": (0, 9000)},
    "MFCC-1":            {"unit": "",     "color": "#f8b500", "range": (-200, 200)},
    "MFCC-2":            {"unit": "",     "color": "#ff7eb3", "range": (-80, 80)},
}

# ── Chart helpers ───────────────────────────────────────────────────────────────
CHART_BG = "rgba(0,0,0,0)"
GRID_COL  = "rgba(255,255,255,0.05)"
TEXT_COL  = "#6b6b85"

def base_layout(**kwargs):
    return dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Space Mono, monospace", color=TEXT_COL, size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        **kwargs,
    )

def radar_chart(features):
    cats = list(features.keys())
    # Normalise 0-1 for radar
    norm = []
    for k, v in features.items():
        lo, hi = FEATURE_META[k]["range"]
        norm.append(np.clip((v - lo) / (hi - lo), 0, 1))
    norm_loop = norm + [norm[0]]
    cats_loop  = cats  + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_loop, theta=cats_loop,
        fill='toself',
        fillcolor='rgba(124,106,255,0.15)',
        line=dict(color='#7c6aff', width=2),
        marker=dict(color='#7c6aff', size=6),
        name="Features",
    ))
    fig.update_layout(**base_layout(
        polar=dict(
            bgcolor='rgba(255,255,255,0.02)',
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_COL, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID_COL, tickfont=dict(size=10, color="#9999bb")),
        ),
        showlegend=False,
        height=360,
    ))
    return fig

def bar_chart(features):
    names  = list(features.keys())
    values = list(features.values())
    colors = [FEATURE_META[n]["color"] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{v}" for v in values],
        textposition='outside',
        textfont=dict(size=10, family="Space Mono"),
    ))
    fig.update_layout(**base_layout(
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COL, zeroline=False),
        height=320,
    ))
    return fig

def gauge_chart(score, threshold=0.5):
    color = "#ff6b6b" if score > threshold else "#00d4aa"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(family="Space Mono", size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor=TEXT_COL, tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=0,
            steps=[
                dict(range=[0, threshold], color="rgba(0,212,170,0.08)"),
                dict(range=[threshold, 1], color="rgba(255,107,107,0.08)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.8,
                value=threshold,
            ),
        ),
    ))
    fig.update_layout(**base_layout(height=260))
    return fig

def donut_chart(probs):
    labels = list(probs.keys())
    values = list(probs.values())
    colors = ["#7c6aff","#00d4aa","#ffa94d","#ff6b6b"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(width=0)),
        textinfo='label+percent',
        textfont=dict(family="Space Mono", size=11),
        insidetextorientation='radial',
    ))
    fig.update_layout(**base_layout(
        showlegend=False,
        height=280,
        annotations=[dict(
            text=f"<b>{labels[np.argmax(values)]}</b>",
            x=0.5, y=0.5, font=dict(size=16, color="#e8e8f0", family="Space Mono"),
            showarrow=False,
        )],
    ))
    return fig

def heatmap_chart(features):
    names  = list(features.keys())
    values = list(features.values())
    norm = []
    for k, v in features.items():
        lo, hi = FEATURE_META[k]["range"]
        norm.append(np.clip((v - lo) / (hi - lo), 0, 1))
    fig = go.Figure(go.Heatmap(
        z=[norm],
        x=names,
        colorscale=[[0,"#1c1c27"],[0.5,"#7c6aff"],[1,"#00d4aa"]],
        showscale=False,
        text=[[f"{v}" for v in values]],
        texttemplate="%{text}",
        textfont=dict(size=10, family="Space Mono"),
    ))
    fig.update_layout(**base_layout(
        xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(visible=False),
        height=160,
    ))
    return fig

def history_trend_chart(history):
    if len(history) < 2:
        return None
    scores = [h.get("anomaly_score", 0) for h in history[:20]][::-1]
    names  = [h.get("filename","?")[:12] for h in history[:20]][::-1]
    colors = ["#ff6b6b" if h.get("is_anomaly") else "#00d4aa" for h in history[:20]][::-1]
    fig = go.Figure(go.Scatter(
        x=list(range(len(scores))), y=scores,
        mode='lines+markers',
        line=dict(color="#7c6aff", width=2),
        marker=dict(color=colors, size=8, line=dict(width=0)),
        fill='tozeroy',
        fillcolor='rgba(124,106,255,0.07)',
    ))
    fig.add_hline(y=0.5, line=dict(color="#ff6b6b", width=1, dash="dot"))
    fig.update_layout(**base_layout(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(gridcolor=GRID_COL, range=[0,1]),
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.5rem'>
        <div style='font-family:Space Mono;font-size:1.1rem;font-weight:700;color:#e8e8f0;letter-spacing:-0.5px'>
            🔊 SonicSense
        </div>
        <div style='font-size:.75rem;color:#6b6b85;margin-top:.25rem'>Acoustic Anomaly Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Page", ["⚡ Analyze", "📂 History", "🔩 RUL Monitor"], label_visibility="collapsed")

    st.markdown('<div class="sec-label">Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Anomaly threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Score above this = anomaly")
    show_raw = st.toggle("Show raw feature values", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#6b6b85;line-height:1.8'>
        <b style='color:#9999bb'>Features detected</b><br>
        RMS Mean · ZCR · RMS Var<br>
        Median Pitch · Spec Centroid<br>
        Spec Rolloff · MFCC-1 · MFCC-2
    </div>
    """, unsafe_allow_html=True)

    if "🔩" in page:
        st.markdown("---")
        st.markdown("""
        <div style='font-size:.72rem;color:#6b6b85;line-height:1.8'>
            <b style='color:#9999bb'>RUL Monitor</b><br>
            Model: CNN-BiLSTM-Attention<br>
            Dataset: NASA IMS<br>
            Bearing: Rexnord ZA-2115<br>
            Speed: 2000 RPM · 6000 lb
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page: ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
if "⚡" in page:

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">SonicSense AI</div>
        <div class="hero-sub">Drop an audio file · extract features · detect anomalies in real-time</div>
        <div class="chip-row">
            <div class="chip">model <span>RandomForest</span></div>
            <div class="chip">anomaly <span>Autoencoder</span></div>
            <div class="chip">features <span>8</span></div>
            <div class="chip">classes <span>Valve · Fan · Pump · Slider</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sec-label">Upload audio</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","flac"], label_visibility="collapsed")

    if uploaded:
        audio_bytes = uploaded.read()

        with st.spinner("Extracting features…"):
            time.sleep(0.6)
            features = extract_features(audio_bytes)

        with st.spinner("Running models…"):
            time.sleep(0.4)
            component, probs = predict_component(features)
            is_anomaly, score = predict_anomaly(features)

        # Save history
        add_to_history({
            "filename":     uploaded.name,
            "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "component":    component,
            "is_anomaly":   is_anomaly,
            "anomaly_score": score,
            "features":     features,
            "probs":        probs,
        })

        # ── Results banner ──────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-family:Space Mono;font-size:.7rem;color:#6b6b85;letter-spacing:1px;text-transform:uppercase;margin-bottom:.8rem">Detected Component</div>
                <div style="font-family:Space Mono;font-size:2.2rem;font-weight:700;color:#e8e8f0">{component}</div>
                <div style="font-size:.8rem;color:#6b6b85;margin-top:.4rem">Machine classification</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            badge_cls  = "badge-anomaly" if is_anomaly else "badge-normal"
            badge_txt  = "⚠ Anomaly" if is_anomaly else "✓ Normal"
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-family:Space Mono;font-size:.7rem;color:#6b6b85;letter-spacing:1px;text-transform:uppercase;margin-bottom:.8rem">Health Status</div>
                <div class="result-badge {badge_cls}">{badge_txt}</div>
                <div style="font-size:.8rem;color:#6b6b85;margin-top:.8rem">Reconstruction score: <b style="color:#e8e8f0">{score}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-family:Space Mono;font-size:.7rem;color:#6b6b85;letter-spacing:1px;text-transform:uppercase;margin-bottom:.8rem">File Info</div>
                <div style="font-family:Space Mono;font-size:.95rem;color:#e8e8f0;word-break:break-all">{uploaded.name}</div>
                <div style="font-size:.8rem;color:#6b6b85;margin-top:.4rem">{len(audio_bytes)/1024:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature values ──────────────────────────────────────────────────
        if show_raw:
            st.markdown('<div class="sec-label">Extracted features</div>', unsafe_allow_html=True)
            feat_html = '<div class="feat-grid">'
            for name, val in features.items():
                meta = FEATURE_META[name]
                lo, hi = meta["range"]
                pct = int(np.clip((val - lo) / (hi - lo), 0, 1) * 100)
                unit = meta["unit"]
                col  = meta["color"]
                feat_html += f"""
                <div class="feat-card">
                    <div class="feat-name">{name}</div>
                    <div class="feat-val">{val} <span style="font-size:.65rem;color:#6b6b85">{unit}</span></div>
                    <div class="feat-bar"><div class="feat-fill" style="width:{pct}%;background:{col}"></div></div>
                </div>"""
            feat_html += '</div>'
            st.markdown(feat_html, unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Visualisations</div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🕸 Radar", "📊 Bar", "🌡 Heatmap", "🎯 Anomaly"])

        with tab1:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(radar_chart(features), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div style="font-family:Space Mono;font-size:.7rem;color:#6b6b85;line-height:2;margin-top:.5rem">
                All 8 features normalised<br>to [0, 1] for comparison.<br><br>
                Outer edge = max of range.<br>
                Purple fill shows the<br>current file's signature.
                </div>""", unsafe_allow_html=True)

        with tab2:
            st.plotly_chart(bar_chart(features), use_container_width=True, config={"displayModeBar": False})

        with tab3:
            st.plotly_chart(heatmap_chart(features), use_container_width=True, config={"displayModeBar": False})

        with tab4:
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("<div style='text-align:center;font-family:Space Mono;font-size:.7rem;color:#6b6b85;margin-bottom:.5rem'>ANOMALY SCORE</div>", unsafe_allow_html=True)
                st.plotly_chart(gauge_chart(score, threshold), use_container_width=True, config={"displayModeBar": False})
            with g2:
                st.markdown("<div style='text-align:center;font-family:Space Mono;font-size:.7rem;color:#6b6b85;margin-bottom:.5rem'>COMPONENT PROBABILITY</div>", unsafe_allow_html=True)
                st.plotly_chart(donut_chart(probs), use_container_width=True, config={"displayModeBar": False})

        # ── Audio player ────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Audio playback</div>', unsafe_allow_html=True)
        st.audio(audio_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;border:1px dashed #2a2a3d;border-radius:16px;margin-top:1rem">
            <div style="font-size:3rem;margin-bottom:1rem">🎵</div>
            <div style="font-family:Space Mono;font-size:1rem;color:#9999bb">Drop an audio file above to begin</div>
            <div style="font-size:.85rem;color:#6b6b85;margin-top:.5rem">Supports WAV · MP3 · OGG · FLAC</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "📂" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#e8e8f0">File History</div>
        <div style="color:#6b6b85;font-size:.9rem;margin-top:.25rem">All previously analysed audio files</div>
    </div>
    """, unsafe_allow_html=True)

    history = load_history()

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;border:1px dashed #2a2a3d;border-radius:16px">
            <div style="font-size:2.5rem;margin-bottom:1rem">📂</div>
            <div style="font-family:Space Mono;color:#9999bb">No files analysed yet</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary row
        total     = len(history)
        anomalies = sum(1 for h in history if h.get("is_anomaly"))
        comp_counts = {}
        for h in history:
            c = h.get("component", "?")
            comp_counts[c] = comp_counts.get(c, 0) + 1
        top_comp = max(comp_counts, key=comp_counts.get) if comp_counts else "—"

        s1, s2, s3 = st.columns(3)
        s1.metric("Total files", total)
        s2.metric("Anomalies detected", anomalies, delta=f"{anomalies/total*100:.0f}%", delta_color="inverse")
        s3.metric("Most common component", top_comp)

        # Trend chart
        trend = history_trend_chart(history)
        if trend:
            st.markdown('<div class="sec-label">Anomaly score trend</div>', unsafe_allow_html=True)
            st.plotly_chart(trend, use_container_width=True, config={"displayModeBar": False})

        # Filter
        st.markdown('<div class="sec-label">File log</div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search = st.text_input("Search filename", placeholder="Type to filter…", label_visibility="collapsed")
        with col_f2:
            filt_comp = st.selectbox("Component", ["All"] + COMPONENTS, label_visibility="collapsed")

        filtered = [
            h for h in history
            if (search.lower() in h.get("filename","").lower() or not search)
            and (filt_comp == "All" or h.get("component") == filt_comp)
        ]

        for h in filtered:
            badge = "⚠" if h.get("is_anomaly") else "✓"
            badge_col = "#ff6b6b" if h.get("is_anomaly") else "#00d4aa"
            st.markdown(f"""
            <div class="hist-row">
                <div style="font-size:1rem">{badge}</div>
                <div class="hist-name">{h.get('filename','unknown')}</div>
                <div class="hist-comp">{h.get('component','?')}</div>
                <div style="font-family:Space Mono;font-size:.75rem;color:{badge_col}">
                    score {h.get('anomaly_score',0):.3f}
                </div>
                <div class="hist-time">{h.get('timestamp','')}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View features", expanded=False):
                feats = h.get("features", {})
                if feats:
                    st.plotly_chart(bar_chart(feats), use_container_width=True,
                                    config={"displayModeBar": False})
                    st.plotly_chart(radar_chart(feats), use_container_width=True,
                                    config={"displayModeBar": False})

        if not filtered:
            st.markdown("<div style='color:#6b6b85;text-align:center;padding:2rem'>No results match your filter.</div>", unsafe_allow_html=True)

        # Clear history
        st.markdown("---")
        if st.button("🗑 Clear all history"):
            save_history([])
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Page: RUL MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "🔩" in page:

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">RUL Monitor</div>
        <div class="hero-sub">Remaining Useful Life · Bearing Prognostics · NASA IMS Dataset</div>
        <div class="chip-row">
            <div class="chip">model <span>CNN-BiLSTM-Attention</span></div>
            <div class="chip">hardware <span>Rexnord ZA-2115</span></div>
            <div class="chip">speed <span>2000 RPM</span></div>
            <div class="chip">load <span>6000 lb radial</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Input mode</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns([1, 1, 4])
    with col_m1:
        live_mode = st.button("▶ Live Stream", use_container_width=True)
    with col_m2:
        manual_mode = st.button("⬆ Manual Upload", use_container_width=True)

    if "rul_mode" not in st.session_state:
        st.session_state.rul_mode = "live"
    if live_mode:
        st.session_state.rul_mode = "live"
    if manual_mode:
        st.session_state.rul_mode = "manual"

    if st.session_state.rul_mode == "manual":
        st.markdown("""
        <div style="border:2px dashed #2a2a3d;border-radius:14px;padding:2rem;text-align:center;
                    color:#6b6b85;margin-bottom:1rem;">
            <div style="font-size:2rem;margin-bottom:.5rem">⬇</div>
            <div style="font-family:Space Mono;font-size:.85rem;color:#9999bb">
                Drop vibration snapshot (.txt)
            </div>
            <div style="font-size:.75rem;margin-top:.4rem">
                1-second burst · 20,480 data points @ 20 kHz
            </div>
        </div>
        """, unsafe_allow_html=True)
        snap_file = st.file_uploader("Vibration snapshot", type=["txt","csv"],
                                     label_visibility="collapsed")
    else:
        snap_file = None

    # ── Generate mock RUL timeline ──────────────────────────────────────────
    import math

    N_POINTS   = 120
    TOTAL_HRS  = 984.5
    HEALTHY_PT = int(N_POINTS * 0.75)

    rng_rul = np.random.default_rng(42)
    hours_axis, raw_rul, smooth_rul = [], [], []
    for i in range(N_POINTS):
        hours_axis.append(round(i * TOTAL_HRS / N_POINTS, 1))
        if i < HEALTHY_PT:
            base = 1.0
        else:
            base = max(0.0, 1.0 - (i - HEALTHY_PT) / (N_POINTS - HEALTHY_PT))
        noise = float(rng_rul.normal(0, 0.035))
        raw_rul.append(round(np.clip(base + noise, 0, 1.05), 4))
        smooth_rul.append(round(base, 4))

    # Current reading = last point (or random step in live mode)
    if "rul_step" not in st.session_state:
        st.session_state.rul_step = 0

    cur_idx  = min(st.session_state.rul_step, N_POINTS - 1)
    cur_rul  = smooth_rul[cur_idx]
    health   = int(cur_rul * 100)
    ttf_hrs  = round(cur_rul * TOTAL_HRS, 1)

    def rul_status(r):
        if r > 0.80: return "Normal Operation",  "#00d4aa", "rgba(0,212,170,0.12)"
        if r > 0.60: return "Early Warning",      "#ffa94d", "rgba(255,169,77,0.12)"
        if r > 0.40: return "Outer Race Wear",    "#ffa94d", "rgba(255,169,77,0.12)"
        if r > 0.20: return "Inner Race Defect",  "#ff6b6b", "rgba(255,107,107,0.12)"
        return         "Imminent Failure",         "#ff6b6b", "rgba(255,107,107,0.12)"

    status_txt, status_col, status_bg = rul_status(cur_rul)

    # ── KPI row ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">System status</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    gauge_color = "#00d4aa" if health > 80 else "#ffa94d" if health > 40 else "#ff6b6b"

    with k1:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="font-family:Space Mono;font-size:.65rem;color:#6b6b85;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:.6rem">Health Index</div>
            <div style="font-family:Space Mono;font-size:2.8rem;font-weight:700;color:{gauge_color};
                        line-height:1">{health}%</div>
            <div style="height:5px;background:#2a2a3d;border-radius:3px;margin-top:.8rem;overflow:hidden">
                <div style="width:{health}%;height:100%;background:{gauge_color};border-radius:3px;
                            transition:width .5s"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="font-family:Space Mono;font-size:.65rem;color:#6b6b85;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:.6rem">Time to Failure</div>
            <div style="font-family:Space Mono;font-size:2.2rem;font-weight:700;color:#7c6aff;
                        line-height:1">{ttf_hrs}</div>
            <div style="font-size:.75rem;color:#6b6b85;margin-top:.4rem">hours remaining</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="font-family:Space Mono;font-size:.65rem;color:#6b6b85;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:.6rem">Bearing Status</div>
            <div style="display:inline-block;padding:.4rem 1.2rem;border-radius:50px;
                        background:{status_bg};color:{status_col};
                        border:1px solid {status_col}44;
                        font-family:Space Mono;font-size:.8rem;font-weight:700;
                        letter-spacing:.5px;margin-top:.2rem">{status_txt}</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="font-family:Space Mono;font-size:.65rem;color:#6b6b85;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:.6rem">RUL (normalised)</div>
            <div style="font-family:Space Mono;font-size:2.2rem;font-weight:700;color:#e8e8f0;
                        line-height:1">{cur_rul:.4f}</div>
            <div style="font-size:.75rem;color:#6b6b85;margin-top:.4rem">model output [0 – 1]</div>
        </div>""", unsafe_allow_html=True)

    # ── Live metrics row ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Sensor readings</div>', unsafe_allow_html=True)

    deg = 1.0 - cur_rul
    rng2 = np.random.default_rng(cur_idx * 7)
    vib_rms   = round(0.012 + deg * 0.38 + float(rng2.uniform(0, 0.01)), 4)
    kurtosis  = round(3.1   + deg * 12   + float(rng2.uniform(0, 0.3)),  2)
    crest_fac = round(3.2   + deg * 6    + float(rng2.uniform(0, 0.2)),  2)
    temperature = round(38  + deg * 22   + float(rng2.uniform(0, 0.5)),  1)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, unit in [
        (m1, "Vibration RMS", vib_rms,     "g"),
        (m2, "Kurtosis",      kurtosis,     "dimensionless"),
        (m3, "Crest Factor",  crest_fac,    "peak / RMS"),
        (m4, "Temperature",   temperature,  "°C"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
                <div style="font-family:Space Mono;font-size:.65rem;color:#6b6b85;
                            letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.5rem">{label}</div>
                <div style="font-family:Space Mono;font-size:1.6rem;font-weight:700;
                            color:#e8e8f0">{val}</div>
                <div style="font-size:.72rem;color:#6b6b85;margin-top:.2rem">{unit}</div>
            </div>""", unsafe_allow_html=True)

    # ── RUL Trend Chart ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">RUL degradation trend</div>', unsafe_allow_html=True)

    fig_rul = go.Figure()

    fig_rul = go.Figure()

    # Healthy zone shading
    fig_rul.add_hrect(y0=0.8, y1=1.05,
                      fillcolor="rgba(0,212,170,0.05)",
                      line_width=0, annotation_text="Healthy zone",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color="#00d4aa44"))

    # Threshold line
    fig_rul.add_hline(y=0.4, line=dict(color="#ff6b6b", width=1, dash="dot"),
                      annotation_text="Failure threshold",
                      annotation_font=dict(size=9, color="#ff6b6b"))

    # Raw prediction (jittery)
    fig_rul.add_trace(go.Scatter(
        x=hours_axis, y=raw_rul,
        mode='lines',
        name='Raw prediction',
        line=dict(color="#3a4060", width=1),
        hovertemplate="Hour %{x}h<br>Raw: %{y:.3f}<extra></extra>",
    ))

    # Smoothed RUL (bold red)
    fig_rul.add_trace(go.Scatter(
        x=hours_axis, y=smooth_rul,
        mode='lines',
        name='Smoothed RUL',
        line=dict(color="#ff6b6b", width=2.5),
        hovertemplate="Hour %{x}h<br>Smoothed: %{y:.3f}<extra></extra>",
    ))

    # Current position marker
    fig_rul.add_vline(x=hours_axis[cur_idx],
                      line=dict(color="#7c6aff", width=1.5, dash="dash"),
                      annotation_text="Now",
                      annotation_font=dict(size=9, color="#7c6aff"))

    fig_rul.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#6b6b85", size=11),
        margin=dict(l=10, r=10, t=20, b=20),
        height=340,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title="Operating Hours",
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10),
            zeroline=False,
        ),
        yaxis=dict(
            title="RUL",
            gridcolor="rgba(255,255,255,0.04)",
            range=[-0.05, 1.1],
            tickfont=dict(size=10),
            zeroline=False,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig_rul, use_container_width=True, config={"displayModeBar": False})

    # ── Advance simulation ───────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Simulation control</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([1, 1, 4])
    with sc1:
        if st.button("⏩ Advance 5 steps"):
            st.session_state.rul_step = min(st.session_state.rul_step + 5, N_POINTS - 1)
            st.rerun()
    with sc2:
        if st.button("↺ Reset simulation"):
            st.session_state.rul_step = 0
            st.rerun()

    st.markdown(f"""
    <div style="font-family:Space Mono;font-size:.7rem;color:#4a5568;margin-top:.5rem">
        Step {cur_idx + 1} / {N_POINTS} · Hour {hours_axis[cur_idx]} / {TOTAL_HRS}
    </div>
    """, unsafe_allow_html=True)

    # ── Footer info ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-family:Space Mono;font-size:.68rem;color:#4a5568;line-height:2;letter-spacing:.5px">
        MODEL: CNN-BiLSTM-Attention &nbsp;·&nbsp;
        DATASET: NASA IMS Bearing &nbsp;·&nbsp;
        BEARING: Rexnord ZA-2115 double-row &nbsp;·&nbsp;
        SAMPLING: 20 kHz / 20,480 pts &nbsp;·&nbsp;
        RUL RANGE: [0.0 → 1.0]
    </div>
    """, unsafe_allow_html=True)
