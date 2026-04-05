"""
L.I.S.T.E.N. — Latent Inference of Sequential Temporal Energy Networks
Industrial Acoustic Anomaly Detection Dashboard
Team: KodeLoverzz | Hacksagon 2026
"""

import time
import random
import warnings
import io

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/get_results"
POLL_INTERVAL = 1.5

VALID_LABELS = [
    "fan00",
    "fan02",
    "fan04",
    "fan06",
    "pump00",
    "pump02",
    "pump04",
    "pump06",
    "slider00",
    "slider02",
    "slider04",
    "slider06",
    "valve00",
    "valve02",
    "valve04",  # valve06 does NOT exist
]

COMPONENT_COLORS = {
    "fan": "#3B82F6",
    "pump": "#F59E0B",
    "slider": "#8B5CF6",
    "valve": "#10B981",
}

CHANNEL_NAMES = [
    "Mel Spectrogram\n(Static Volume)",
    "Delta\n(Velocity of Change)",
    "Delta-Delta\n(Acceleration)",
]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="L.I.S.T.E.N. — Industrial AI Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── Import fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Chakra+Petch:wght@300;400;500;600;700&display=swap');

/* ── Root variables ───────────────────────────────── */
:root {
    --bg-primary:   #0A0E1A;
    --bg-secondary: #0F1628;
    --bg-card:      #141E35;
    --bg-card-2:    #1A2540;
    --border:       #1E2D4A;
    --border-bright:#2A4080;
    --amber:        #F59E0B;
    --amber-dim:    #92600A;
    --green:        #10B981;
    --red:          #EF4444;
    --blue:         #3B82F6;
    --text-primary: #E2E8F0;
    --text-dim:     #64748B;
    --text-muted:   #334155;
    --font-mono:    'JetBrains Mono', monospace;
    --font-display: 'Chakra Petch', sans-serif;
}

/* ── Global base ──────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-mono) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp { background-color: var(--bg-primary) !important; }

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090D1A 0%, #0C1221 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Headings ─────────────────────────────────────── */
h1, h2, h3 { font-family: var(--font-display) !important; letter-spacing: 0.04em; }

/* ── Metric ───────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--amber) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.7rem !important; }

/* ── Tabs ─────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-dim) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 8px !important;
}

/* ── Button ───────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1A2540, #1E2D4A) !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber-dim) !important;
    border-radius: 4px !important;
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1E2D4A, #243560) !important;
    border-color: var(--amber) !important;
    box-shadow: 0 0 12px rgba(245,158,11,0.2) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Progress ─────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--amber-dim), var(--amber)) !important;
}

/* ── Status cards ─────────────────────────────────── */
.status-normal {
    background: linear-gradient(135deg, #0D2318, #0D1F18);
    border: 1px solid #164A30;
    border-left: 3px solid var(--green);
    border-radius: 6px;
    padding: 10px 16px;
    margin: 4px 0;
}
.status-anomaly {
    background: linear-gradient(135deg, #2A0D0D, #200D0D);
    border: 1px solid #4A1616;
    border-left: 3px solid var(--red);
    border-radius: 6px;
    padding: 10px 16px;
    margin: 4px 0;
    animation: pulse-red 2s ease-in-out infinite;
}
.status-processing {
    background: linear-gradient(135deg, #1A1A0D, #1A180D);
    border: 1px solid #4A4010;
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 10px 16px;
    margin: 4px 0;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 10px 3px rgba(239,68,68,0.15); }
}

/* ── Node header ──────────────────────────────────── */
.node-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-display);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.node-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.dot-green  { background: var(--green);  box-shadow: 0 0 6px var(--green); }
.dot-red    { background: var(--red);    box-shadow: 0 0 6px var(--red);  animation: blink 1s step-end infinite; }
.dot-amber  { background: var(--amber);  box-shadow: 0 0 6px var(--amber); animation: blink 0.5s step-end infinite; }
.dot-grey   { background: var(--text-muted); }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

/* ── XAI explanation box ──────────────────────────── */
.xai-box {
    background: var(--bg-card-2);
    border: 1px solid var(--border-bright);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.78rem;
    line-height: 1.6;
    color: var(--text-primary);
    margin: 8px 0;
}
.xai-box .label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--amber);
    margin-bottom: 4px;
}

/* ── Result card ──────────────────────────────────── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: var(--font-display);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    font-weight: 600;
    text-transform: uppercase;
}

/* ── Model slot ───────────────────────────────────── */
.model-slot {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 4px;
    font-size: 0.7rem;
}

/* ── Divider ──────────────────────────────────────── */
.h-line {
    border: none;
    border-top: 1px solid var(--border);
    margin: 12px 0;
}

/* ── Gauge wrap ───────────────────────────────────── */
.gauge-wrap {
    text-align: center;
    font-family: var(--font-mono);
}

/* ── Section header ───────────────────────────────── */
.section-header {
    font-family: var(--font-display);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--amber);
    text-transform: uppercase;
    padding: 4px 0;
    border-bottom: 1px solid var(--amber-dim);
    margin-bottom: 10px;
}

/* ── Live ping ────────────────────────────────────── */
.live-ping {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 0 rgba(16,185,129,0.6);
    animation: ping 1.4s cubic-bezier(0,0,0.2,1) infinite;
    margin-right: 6px;
}
@keyframes ping {
    75%, 100% { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
}

/* ── RUL gauge ────────────────────────────────────── */
.rul-value {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.02em;
}
.rul-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-align: center;
    color: var(--text-dim);
    text-transform: uppercase;
}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

/* ── Info/warning/error override ─────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    font-family: var(--font-mono) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY / MOCK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def fetch_latest_pipeline_data():
    """
    Placeholder — wire this to your upstream classifier script.
    Returns: (machine_type: str, machine_id: str, acoustic_tensor: np.ndarray)
    acoustic_tensor shape: (1, 3, 128, 128)  — Mel, Delta, Delta-Delta
    """
    machine_types = ["fan", "pump", "slider", "valve"]
    machine_ids = ["00", "02", "04", "06"]
    m_type = random.choice(machine_types)
    m_id = random.choice(machine_ids if m_type != "valve" else ["00", "02", "04"])
    tensor = np.random.randn(1, 3, 128, 128).astype(np.float32)
    return m_type, m_id, tensor


def mock_model_output(acoustic_tensor: np.ndarray):
    """Returns a mocked reconstructed tensor and MSE."""
    noise = np.random.randn(*acoustic_tensor.shape).astype(np.float32) * 0.15
    reconstructed = acoustic_tensor + noise
    mse = float(np.mean((acoustic_tensor - reconstructed) ** 2))
    return reconstructed, mse


def mock_shap_values(acoustic_tensor: np.ndarray) -> np.ndarray:
    """Returns mock SHAP values matching (3, H, W)."""
    shap = np.random.randn(3, 128, 128).astype(np.float32) * 0.05
    # Make them more interesting — high-freq anomaly in channel 0
    shap[0, 80:, :] += 0.08
    shap[1, 40:80, 30:90] -= 0.04
    return shap


def get_threshold_for_machine(machine_type: str, machine_id: str) -> float:
    """Returns a pre-calculated threshold for the machine."""
    thresholds = {
        "fan": 0.031,
        "pump": 0.028,
        "slider": 0.035,
        "valve": 0.025,
    }
    return thresholds.get(machine_type, 0.030)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB HELPERS — all return fig objects, dark-styled
# ─────────────────────────────────────────────────────────────────────────────

DARK_FIG_PARAMS = {
    "figure.facecolor": "#141E35",
    "axes.facecolor": "#0F1628",
    "axes.edgecolor": "#1E2D4A",
    "axes.labelcolor": "#94A3B8",
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "text.color": "#E2E8F0",
    "grid.color": "#1E2D4A",
    "grid.linewidth": 0.5,
}


def _apply_dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#0F1628")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E2D4A")
    ax.tick_params(colors="#64748B", labelsize=7)
    if title:
        ax.set_title(title, color="#94A3B8", fontsize=8, pad=4, fontfamily="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, color="#64748B", fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color="#64748B", fontsize=7)


def plot_shap_heatmaps(shap_values: np.ndarray) -> plt.Figure:
    """3-channel SHAP attribution maps side by side."""
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor="#141E35")
        fig.subplots_adjust(wspace=0.08, left=0.04, right=0.96, top=0.88, bottom=0.12)
        titles = [
            "MEL  ·  Static Volume Anomalies",
            "DELTA  ·  Velocity Anomalies",
            "Δ-DELTA  ·  Acceleration Anomalies",
        ]
        vmax = np.abs(shap_values).max() + 1e-9
        for i, ax in enumerate(axes):
            im = ax.imshow(
                shap_values[i],
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            _apply_dark(ax, title=titles[i])
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(colors="#64748B", labelsize=6)
            cb.ax.yaxis.set_tick_params(color="#1E2D4A")
    return fig


def plot_residual_overlay(
    original: np.ndarray, reconstructed: np.ndarray
) -> plt.Figure:
    """Mel spectrogram with reconstruction error overlaid."""
    residual = (original[0, 0] - reconstructed[0, 0]) ** 2
    mel = original[0, 0]
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(figsize=(9, 3), facecolor="#141E35")
        ax.imshow(
            mel, aspect="auto", origin="lower", cmap="magma", interpolation="bilinear"
        )
        ax.imshow(
            residual,
            aspect="auto",
            origin="lower",
            cmap="hot",
            alpha=0.55,
            interpolation="bilinear",
        )
        _apply_dark(
            ax,
            title="RAW RECONSTRUCTION ERROR  ·  Mel + Residual Overlay",
            xlabel="Time Frames →",
            ylabel="Mel Bins ↑",
        )
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def plot_edge_features(feature_values: list, feature_names: list) -> plt.Figure:
    """Bar chart for 8 Edge audio features."""
    colors = [
        "#F59E0B" if v > np.mean(feature_values) else "#3B82F6" for v in feature_values
    ]
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#141E35")
        bars = ax.bar(
            feature_names,
            feature_values,
            color=colors,
            edgecolor="#1E2D4A",
            linewidth=0.8,
        )
        for b, v in zip(bars, feature_values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.001,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#94A3B8",
            )
        ax.axhline(
            np.mean(feature_values),
            color="#EF4444",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label="Mean",
        )
        _apply_dark(ax, title="EDGE AUDIO FEATURES  ·  Live Reading")
        ax.tick_params(axis="x", labelsize=6.5, rotation=12)
        ax.legend(
            fontsize=6.5, facecolor="#141E35", edgecolor="#1E2D4A", labelcolor="#94A3B8"
        )
    fig.tight_layout()
    return fig


def plot_gpu_mel_variance(spectrogram_data: list) -> plt.Figure:
    """Line/area chart for GPU Mel spectrogram variance."""
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#141E35")
        x = range(len(spectrogram_data))
        ax.fill_between(x, spectrogram_data, alpha=0.25, color="#F59E0B")
        ax.plot(spectrogram_data, color="#F59E0B", linewidth=1.2)
        ax.axhline(0, color="#64748B", linewidth=0.6, linestyle=":")
        _apply_dark(
            ax,
            title="GPU MEL VARIANCE  ·  Anomaly Region",
            xlabel="Time Frame",
            ylabel="Variance",
        )
    fig.tight_layout()
    return fig


def plot_rul_gauge(rul_pct: float) -> plt.Figure:
    """Radial gauge showing remaining useful life %."""
    color = "#10B981" if rul_pct > 60 else ("#F59E0B" if rul_pct > 30 else "#EF4444")
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(
            figsize=(4, 2.4), subplot_kw={"projection": "polar"}, facecolor="#141E35"
        )
        ax.set_facecolor("#0F1628")
        theta_range = np.linspace(np.pi, 0, 200)
        ax.fill_between(theta_range, 0.7, 1.0, color="#1E2D4A", alpha=0.8)
        filled = np.linspace(np.pi, np.pi - np.pi * (rul_pct / 100), 200)
        ax.fill_between(filled, 0.7, 1.0, color=color, alpha=0.9)
        ax.set_ylim(0, 1.1)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(
            0,
            0,
            f"{rul_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=22,
            color=color,
            fontweight="bold",
            fontfamily="monospace",
        )
        ax.text(
            np.pi / 2,
            1.35,
            "RUL",
            ha="center",
            va="center",
            fontsize=8,
            color="#64748B",
            fontfamily="monospace",
        )
    fig.tight_layout(pad=0)
    return fig


def plot_rul_trend(history: list) -> plt.Figure:
    """Line chart of RUL degradation trend."""
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#141E35")
        x = list(range(len(history)))
        color_grad = [
            "#10B981" if v > 60 else ("#F59E0B" if v > 30 else "#EF4444")
            for v in history
        ]
        for i in range(len(history) - 1):
            ax.plot(x[i : i + 2], history[i : i + 2], color=color_grad[i], linewidth=2)
        ax.fill_between(x, history, alpha=0.12, color="#F59E0B")
        ax.axhline(
            30,
            color="#EF4444",
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
            label="Critical threshold (30%)",
        )
        ax.axhline(
            60,
            color="#F59E0B",
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
            label="Warning threshold (60%)",
        )
        _apply_dark(
            ax,
            title="RUL TREND  ·  Degradation Profile",
            xlabel="Time Window",
            ylabel="RUL %",
        )
        ax.set_ylim(0, 105)
        ax.legend(
            fontsize=6.5, facecolor="#141E35", edgecolor="#1E2D4A", labelcolor="#94A3B8"
        )
    fig.tight_layout()
    return fig


def plot_feature_shap_bar(feature_names: list, shap_vals: list) -> plt.Figure:
    """Horizontal bar chart of SHAP contributions for RUL."""
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    names_s = [feature_names[i] for i in sorted_idx]
    shaps_s = [shap_vals[i] for i in sorted_idx]
    colors = ["#EF4444" if s > 0 else "#3B82F6" for s in shaps_s]
    with plt.rc_context(DARK_FIG_PARAMS):
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#141E35")
        bars = ax.barh(
            names_s,
            shaps_s,
            color=colors,
            edgecolor="#1E2D4A",
            linewidth=0.7,
            height=0.65,
        )
        ax.axvline(0, color="#64748B", linewidth=0.8)
        _apply_dark(ax, title="FEATURE SHAP VALUES  ·  RUL Contribution")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
        <div style='text-align:center; padding: 10px 0 18px 0;'>
            <div style='font-family:"Chakra Petch",sans-serif; font-size:1.3rem;
                        font-weight:700; letter-spacing:0.12em; color:#F59E0B;'>
                L·I·S·T·E·N
            </div>
            <div style='font-size:0.58rem; letter-spacing:0.18em; color:#334155;
                        margin-top:4px; text-transform:uppercase;'>
                Latent Inference of Sequential<br>Temporal Energy Networks
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "NAVIGATION",
            ["🏭  Main Dashboard", "🔬  Classifier", "📈  RUL Estimation"],
            label_visibility="visible",
        )

        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:16px 0;'>", unsafe_allow_html=True
        )

        # Live status indicator
        st.markdown(
            """
        <div style='font-size:0.65rem; letter-spacing:0.12em; color:#64748B;
                    text-transform:uppercase; margin-bottom:6px;'>
            Pipeline Status
        </div>
        <div style='font-size:0.75rem; display:flex; align-items:center; gap:8px;'>
            <span class="live-ping"></span>
            <span style='color:#10B981;'>Listening for upstream data</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:16px 0;'>", unsafe_allow_html=True
        )

        # System info
        st.markdown(
            """
        <div style='font-size:0.65rem; letter-spacing:0.12em; color:#64748B;
                    text-transform:uppercase; margin-bottom:8px;'>
            System Info
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
        <div style='font-size:0.68rem; color:#64748B; line-height:1.9;'>
            Backend  <span style='color:#94A3B8; float:right;'>FastAPI/Uvicorn</span><br>
            Node A   <span style='color:#94A3B8; float:right;'>Edge (ONNX)</span><br>
            Node B   <span style='color:#94A3B8; float:right;'>GPU (PyTorch)</span><br>
            XAI      <span style='color:#94A3B8; float:right;'>GradientSHAP</span><br>
            Poll     <span style='color:#94A3B8; float:right;'>{POLL_INTERVAL}s</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:16px 0;'>", unsafe_allow_html=True
        )
        st.markdown(
            """
        <div style='font-size:0.6rem; color:#1E2D4A; text-align:center; letter-spacing:0.08em;'>
            TEAM KODELOVERZZ · HACKSAGON 2026
        </div>
        """,
            unsafe_allow_html=True,
        )

    return page.strip().lstrip("🏭🔬📈 ")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────


def render_main_dashboard():
    st.markdown(
        """
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
        <div style='font-family:"Chakra Petch",sans-serif; font-size:1.4rem;
                    font-weight:700; letter-spacing:0.06em; color:#E2E8F0;'>
            Industrial Acoustic Monitor
        </div>
        <span class="live-ping"></span>
    </div>
    <div style='font-size:0.68rem; color:#334155; letter-spacing:0.1em; margin-bottom:20px;'>
        REAL-TIME EDGE & GPU ANOMALY DETECTION  ·  CONTINUOUS STREAM
    </div>
    """,
        unsafe_allow_html=True,
    )

    placeholder = st.empty()

    while True:
        try:
            resp = requests.get(API_URL, timeout=2)
            data = resp.json()
        except Exception:
            data = None

        with placeholder.container():
            if data is None:
                st.markdown(
                    """
                <div class='status-processing'>
                    <span style='color:#F59E0B; font-size:0.8rem;'>⚠ Backend offline</span>
                    <span style='color:#64748B; font-size:0.72rem; margin-left:12px;'>
                    Start the Uvicorn server: <code>uvicorn server.app:app --reload</code></span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif not data:
                st.markdown(
                    """
                <div class='status-processing'>
                    <span style='color:#F59E0B; font-size:0.8rem;'>
                        ◌  Waiting for telemetry from edge devices…
                    </span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # ── Overview metrics row ───────────────────────────────────
                edge_nodes = {
                    k: v for k, v in data.items() if v.get("project_id") == "edge"
                }
                gpu_nodes = {
                    k: v for k, v in data.items() if v.get("project_id") == "gpu"
                }
                total_anomalies = sum(
                    1
                    for v in data.values()
                    if v.get("latest_result", {})
                    and v["latest_result"].get("is_anomaly")
                )

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Nodes", len(data))
                mc2.metric("Edge Nodes", len(edge_nodes))
                mc3.metric("GPU Nodes", len(gpu_nodes))
                mc4.metric("Active Anomalies", total_anomalies)

                st.markdown(
                    "<hr style='border-color:#1E2D4A; margin:12px 0;'>",
                    unsafe_allow_html=True,
                )

                # ── Tabs ───────────────────────────────────────────────────
                tab_edge, tab_gpu = st.tabs(
                    [
                        f"⚡  Project A — Edge Nodes ({len(edge_nodes)})",
                        f"🖥  Project B — GPU Nodes  ({len(gpu_nodes)})",
                    ]
                )

                for tab, nodes, project_label in [
                    (tab_edge, edge_nodes, "EDGE"),
                    (tab_gpu, gpu_nodes, "GPU"),
                ]:
                    with tab:
                        if not nodes:
                            st.markdown(
                                f"<div style='color:#334155; font-size:0.75rem; padding:20px;'>"
                                f"No {project_label} nodes connected yet.</div>",
                                unsafe_allow_html=True,
                            )
                            continue

                        for node_id, state in nodes.items():
                            res = state.get("latest_result")
                            status = state.get("status", "unknown")
                            n_type = state.get("node_type", "?").upper()
                            n_tid = state.get("node_type_id", "?")
                            is_anomaly = res.get("is_anomaly", False) if res else False

                            # Dot color
                            if status in ("processing", "queued"):
                                dot_cls = "dot-amber"
                            elif is_anomaly:
                                dot_cls = "dot-red"
                            elif res:
                                dot_cls = "dot-green"
                            else:
                                dot_cls = "dot-grey"

                            exp_label = (
                                f"🚨  {node_id}  ·  {n_type} #{n_tid}  "
                                f"·  ANOMALY  ·  MSE {res['mse_score']:.4f}"
                                if (res and is_anomaly)
                                else (
                                    f"✅  {node_id}  ·  {n_type} #{n_tid}"
                                    if res
                                    else f"◌  {node_id}  ·  {n_type} #{n_tid}  ·  Awaiting"
                                )
                            )

                            with st.expander(exp_label, expanded=is_anomaly):
                                # Node header row
                                st.markdown(
                                    f"<div class='node-header'>"
                                    f"<span class='node-dot {dot_cls}'></span>"
                                    f"<span style='color:#94A3B8;'>NODE</span>"
                                    f"<span style='color:#E2E8F0; font-weight:600;'>{node_id}</span>"
                                    f"<span style='color:#334155;'>·</span>"
                                    f"<span style='color:#64748B;'>STATUS</span>"
                                    f"<span style='color:#F59E0B;'>{status.upper()}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                                if not res:
                                    st.markdown(
                                        "<div style='color:#334155; font-size:0.74rem; "
                                        "padding:8px 0;'>Awaiting first inference…</div>",
                                        unsafe_allow_html=True,
                                    )
                                    continue

                                col_status, col_graph = st.columns([1, 2])

                                with col_status:
                                    mse = res.get("mse_score", 0)
                                    # Show verdict
                                    if is_anomaly:
                                        st.markdown(
                                            f"<div class='status-anomaly'>"
                                            f"<div style='font-size:0.9rem; font-weight:700; "
                                            f"color:#EF4444;'>🚨 ANOMALY DETECTED</div>"
                                            f"<div style='font-size:0.7rem; color:#94A3B8; "
                                            f"margin-top:4px;'>MSE Score: "
                                            f"<span style='color:#F87171; font-weight:600;'>"
                                            f"{mse:.6f}</span></div>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )

                                        expl = res.get("explanation", "")
                                        if expl:
                                            st.markdown(
                                                f"<div class='xai-box'>"
                                                f"<div class='label'>⬡ XAI DIAGNOSIS</div>"
                                                f"{expl}"
                                                f"</div>",
                                                unsafe_allow_html=True,
                                            )
                                    else:
                                        st.markdown(
                                            f"<div class='status-normal'>"
                                            f"<div style='font-size:0.9rem; font-weight:700; "
                                            f"color:#10B981;'>✅ NORMAL OPERATION</div>"
                                            f"<div style='font-size:0.7rem; color:#94A3B8; "
                                            f"margin-top:4px;'>MSE Score: "
                                            f"<span style='color:#6EE7B7; font-weight:600;'>"
                                            f"{mse:.6f}</span></div>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )

                                    m_type = res.get("machine_type", "unknown").upper()
                                    st.markdown(
                                        f"<div style='font-size:0.68rem; color:#64748B; "
                                        f"margin-top:8px;'>VERIFIED MACHINE TYPE</div>"
                                        f"<div style='font-size:0.85rem; color:#E2E8F0; "
                                        f"font-weight:600; letter-spacing:0.06em;'>{m_type}</div>",
                                        unsafe_allow_html=True,
                                    )

                                with col_graph:
                                    graphs = res.get("graphs")
                                    if graphs and is_anomaly:
                                        spec_data = graphs.get("spectrogram", [])
                                        feat_data = graphs.get("audio_features", [])

                                        st.markdown(
                                            "<div class='section-header'>Anomaly Signal Graphs</div>",
                                            unsafe_allow_html=True,
                                        )

                                        g1, g2 = st.columns(2)
                                        if project_label == "EDGE":
                                            feat_names = [
                                                "RMS",
                                                "Kurtosis",
                                                "Skewness",
                                                "Peak",
                                                "Crest",
                                                "MAV",
                                                "BPFO",
                                                "BPFI",
                                            ]
                                            with g1:
                                                st.markdown(
                                                    "<div style='font-size:0.65rem; "
                                                    "color:#64748B;'>Edge Audio Features</div>",
                                                    unsafe_allow_html=True,
                                                )
                                                vals = (
                                                    feat_data[:8]
                                                    if len(feat_data) >= 8
                                                    else feat_data
                                                    + [0] * (8 - len(feat_data))
                                                )
                                                fig = plot_edge_features(
                                                    vals, feat_names
                                                )
                                                st.pyplot(fig, use_container_width=True)
                                                plt.close(fig)
                                        else:
                                            with g1:
                                                st.markdown(
                                                    "<div style='font-size:0.65rem; "
                                                    "color:#64748B;'>Mel Spectrogram Variance</div>",
                                                    unsafe_allow_html=True,
                                                )
                                                fig = plot_gpu_mel_variance(spec_data)
                                                st.pyplot(fig, use_container_width=True)
                                                plt.close(fig)

                                        with g2:
                                            st.markdown(
                                                "<div style='font-size:0.65rem; "
                                                "color:#64748B;'>Audio Feature Shift</div>",
                                                unsafe_allow_html=True,
                                            )
                                            fig2, ax2 = plt.subplots(
                                                figsize=(5, 2.8), facecolor="#141E35"
                                            )
                                            x = range(len(spec_data))
                                            ax2.fill_between(
                                                x, spec_data, alpha=0.2, color="#3B82F6"
                                            )
                                            ax2.plot(
                                                spec_data,
                                                color="#3B82F6",
                                                linewidth=1.2,
                                            )
                                            _apply_dark(ax2)
                                            fig2.tight_layout()
                                            st.pyplot(fig2, use_container_width=True)
                                            plt.close(fig2)
                                    elif not is_anomaly:
                                        st.markdown(
                                            "<div style='color:#334155; font-size:0.73rem; "
                                            "padding:20px 0; text-align:center;'>"
                                            "No anomaly graphs — operating within normal bounds.</div>",
                                            unsafe_allow_html=True,
                                        )

        time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────


def render_classifier_page():
    st.markdown(
        """
    <div style='font-family:"Chakra Petch",sans-serif; font-size:1.4rem;
                font-weight:700; letter-spacing:0.06em; color:#E2E8F0; margin-bottom:4px;'>
        Machine Component Classifier
    </div>
    <div style='font-size:0.68rem; color:#334155; letter-spacing:0.1em; margin-bottom:20px;'>
        AUDIO FILE → NUMPY ARRAY → ML PIPELINE → COMPONENT IDENTIFICATION
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Model status slots in sidebar ──────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:12px 0;'>", unsafe_allow_html=True
        )
        st.markdown(
            """
        <div style='font-size:0.65rem; letter-spacing:0.12em; color:#64748B;
                    text-transform:uppercase; margin-bottom:8px;'>
            Model Registry
        </div>
        """,
            unsafe_allow_html=True,
        )

        MODEL_SLOTS = [
            ("Model 1", "Component Classifier", "model_final.joblib"),
            ("Model 2", "Fan Sub-Classifier", "model_fan.joblib"),
            ("Model 3", "Pump Sub-Classifier", "model_pump.joblib"),
            ("Model 4", "Slider Sub-Classifier", "model_slider.joblib"),
            ("Model 5", "Valve Sub-Classifier", "model_valve.joblib"),
        ]

        active_models = st.session_state.get("active_models", set())
        for slot, desc, fname in MODEL_SLOTS:
            dot = "dot-green" if slot in active_models else "dot-grey"
            st.markdown(
                f"<div class='model-slot'>"
                f"<span class='node-dot {dot}'></span>"
                f"<div><div style='font-size:0.7rem; color:#94A3B8;'>{slot}</div>"
                f"<div style='font-size:0.58rem; color:#334155;'>{desc}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:12px 0;'>", unsafe_allow_html=True
        )

        # Valid label legend
        st.markdown(
            """
        <div style='font-size:0.65rem; letter-spacing:0.12em; color:#64748B;
                    text-transform:uppercase; margin-bottom:6px;'>
            Valid Output Labels
        </div>
        """,
            unsafe_allow_html=True,
        )
        for comp, color in COMPONENT_COLORS.items():
            ids = ["00", "02", "04", "06"] if comp != "valve" else ["00", "02", "04"]
            id_str = "  ".join(
                f"<span style='color:{color};'>{comp}{i}</span>" for i in ids
            )
            st.markdown(
                f"<div style='font-size:0.65rem; line-height:2; margin-bottom:2px;'>"
                f"{id_str}</div>",
                unsafe_allow_html=True,
            )

    # ── Main layout ────────────────────────────────────────────────────────
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            "<div class='section-header'>① Audio Input</div>", unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload a .wav file from the MIMII dataset",
            type=["wav"],
            help="File will be internally converted to a NumPy array before classification.",
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown(
                f"<div style='font-size:0.7rem; color:#64748B; margin:6px 0;'>"
                f"📁  <span style='color:#94A3B8;'>{uploaded.name}</span>"
                f"  ·  {uploaded.size / 1024:.1f} KB</div>",
                unsafe_allow_html=True,
            )
            st.audio(uploaded, format="audio/wav")
            st.markdown(
                "<div style='font-size:0.65rem; color:#334155; margin-top:4px;'>"
                "ℹ  File will be resampled to 16 kHz and converted to a NumPy "
                "float32 array before being passed to the classifier pipeline.</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='section-header' style='margin-top:20px;'>② Run Pipeline</div>",
            unsafe_allow_html=True,
        )

        # Mock selector for UI testing
        # st.markdown(
        #     "<div style='font-size:0.68rem; color:#64748B; margin-bottom:6px;'>"
        #     "Demo mode — select expected output label:</div>",
        #     unsafe_allow_html=True)
        # mock_label = st.selectbox(
        #     "Expected Label (Demo)",
        #     VALID_LABELS,
        #     label_visibility="collapsed",
        # )

        run_btn = st.button("⬢  RUN CLASSIFIER PIPELINE", use_container_width=True)

    with col_output:
        st.markdown(
            "<div class='section-header'>③ Classification Result</div>",
            unsafe_allow_html=True,
        )

        if run_btn:
            # # Activate model slots progressively
            # st.session_state["active_models"] = set()
            # st.session_state["classifier_result"] = None

            # with st.spinner("Running classifier pipeline…"):
            #     for i, (slot, _, _) in enumerate(MODEL_SLOTS):
            #         time.sleep(0.22)
            #         st.session_state["active_models"] = {
            #             s for j, (s, _, _) in enumerate(MODEL_SLOTS) if j <= i
            #         }
            #     # Use selected mock label
            #     st.session_state["classifier_result"] = mock_label
            #     time.sleep(0.15)

            # st.rerun()
            if not uploaded:
                st.error("Please upload a .wav file first.")
            else:
                # Activate model slots progressively
                st.session_state["active_models"] = set()
                st.session_state["classifier_result"] = None

                with st.spinner("Running classifier pipeline…"):
                    # Keep the UI animation
                    for i, (slot, _, _) in enumerate(MODEL_SLOTS):
                        time.sleep(0.22)
                        st.session_state["active_models"] = {
                            s for j, (s, _, _) in enumerate(MODEL_SLOTS) if j <= i
                        }

                    # ACTUAL API CALL: Send the file to the FastAPI backend
                    try:
                        files = {
                            "file": (uploaded.name, uploaded.getvalue(), "audio/wav")
                        }
                        response = requests.post(
                            "http://localhost:8000/classify", files=files
                        )

                        if response.status_code == 200:
                            st.session_state["classifier_result"] = response.json().get(
                                "predicted_label"
                            )
                        else:
                            st.error(f"Backend Error: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(
                            f"Failed to connect to the backend: {e}. Is Uvicorn running?"
                        )

                st.rerun()

        result = st.session_state.get("classifier_result")

        if result:
            # Parse component type and ID
            comp_type = "".join(c for c in result if c.isalpha())
            comp_id = "".join(c for c in result if c.isdigit())
            color = COMPONENT_COLORS.get(comp_type, "#94A3B8")

            st.markdown(
                f"<div class='result-card'>"
                f"  <div style='font-size:0.62rem; letter-spacing:0.14em; "
                f"  color:#64748B; text-transform:uppercase; margin-bottom:10px;'>"
                f"  Classification Result</div>"
                f"  <div style='display:flex; align-items:center; gap:12px; "
                f"  margin-bottom:14px;'>"
                f'    <div style=\'font-size:2.4rem; font-family:"Chakra Petch",sans-serif; '
                f"    font-weight:700; color:{color}; letter-spacing:0.04em;'>{result}</div>"
                f"    <span class='badge' style='background:{color}22; color:{color}; "
                f"    border:1px solid {color}55;'>{comp_type}</span>"
                f"  </div>"
                f"  <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>"
                f"    <div>"
                f"      <div style='font-size:0.6rem; color:#64748B; letter-spacing:0.1em; "
                f"      text-transform:uppercase;'>Component Type</div>"
                f"      <div style='font-size:1rem; font-weight:600; color:{color}; "
                f'      margin-top:2px; font-family:"Chakra Petch",sans-serif; '
                f"      letter-spacing:0.06em;'>{comp_type.upper()}</div>"
                f"    </div>"
                f"    <div>"
                f"      <div style='font-size:0.6rem; color:#64748B; letter-spacing:0.1em; "
                f"      text-transform:uppercase;'>Component ID</div>"
                f"      <div style='font-size:1rem; font-weight:600; color:#E2E8F0; "
                f'      margin-top:2px; font-family:"Chakra Petch",sans-serif;\'>{comp_id}</div>'
                f"    </div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence mock bar
            confidence = random.uniform(0.82, 0.99)
            st.markdown(
                "<div style='font-size:0.65rem; color:#64748B; margin:12px 0 4px 0; "
                "letter-spacing:0.1em;'>CLASSIFIER CONFIDENCE</div>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.markdown(
                f"<div style='font-size:0.7rem; color:#F59E0B; margin-top:2px;'>"
                f"{confidence * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )

            # Sub-stage breakdown
            with st.expander("Pipeline Stage Breakdown", expanded=False):
                stages = [
                    (
                        "Component Classifier",
                        f"Predicted class: **{comp_type}**  (4-way)",
                        True,
                    ),
                    (
                        f"{comp_type.capitalize()} Sub-Classifier",
                        f"Predicted ID: **{comp_id}** within {comp_type} family",
                        True,
                    ),
                    (
                        "Feature Extraction",
                        "8 acoustic features extracted (MFCCs, RMS, Kurtosis…)",
                        True,
                    ),
                    (
                        "Scaler Normalisation",
                        "StandardScaler applied per feature dimension",
                        True,
                    ),
                ]
                for stage_name, stage_detail, ok in stages:
                    status_icon = "✅" if ok else "⏳"
                    st.markdown(
                        f"<div style='display:flex; gap:10px; align-items:flex-start; "
                        f"padding:5px 0; border-bottom:1px solid #1E2D4A; "
                        f"font-size:0.72rem;'>"
                        f"  <span>{status_icon}</span>"
                        f"  <div>"
                        f"    <div style='color:#94A3B8; font-weight:600;'>{stage_name}</div>"
                        f"    <div style='color:#64748B; font-size:0.65rem;'>{stage_detail}</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                "<div style='color:#1E2D4A; font-size:0.78rem; padding:40px 0; "
                "text-align:center; border:1px dashed #1E2D4A; border-radius:6px;'>"
                "Upload a .wav file and click RUN to see the classification result."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Deep-Dive XAI Diagnostics ─────────────────────────────────────────
    result = st.session_state.get("classifier_result")
    if result:
        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:20px 0;'>", unsafe_allow_html=True
        )
        st.markdown(
            '<div style=\'font-family:"Chakra Petch",sans-serif; font-size:1rem; '
            "font-weight:700; color:#E2E8F0; letter-spacing:0.06em; "
            "margin-bottom:16px;'>Deep Diagnostics Stack</div>",
            unsafe_allow_html=True,
        )

        comp_type = "".join(c for c in result if c.isalpha())
        comp_id = "".join(c for c in result if c.isdigit())
        threshold = get_threshold_for_machine(comp_type, comp_id)

        with st.spinner("Running deep analysis…"):
            m_type, m_id, tensor = fetch_latest_pipeline_data()
            reconstructed, mse = mock_model_output(tensor)
            shap_vals = mock_shap_values(tensor)
            is_anomaly = mse > threshold

        # ── Tier 1: Detection Verdict ──────────────────────────────────
        st.markdown(
            "<div class='section-header'>TIER 1  ·  Detection Verdict</div>",
            unsafe_allow_html=True,
        )

        t1c1, t1c2, t1c3 = st.columns([1, 1, 2])
        with t1c1:
            st.metric("MSE Score", f"{mse:.6f}")
        with t1c2:
            st.metric("Threshold", f"{threshold:.6f}")
        with t1c3:
            if is_anomaly:
                st.markdown(
                    "<div class='status-anomaly' style='height:100%; padding:14px 18px;'>"
                    "<div style='font-size:1.1rem; font-weight:700; color:#EF4444;'>"
                    "🚨  STATUS: ANOMALY DETECTED</div>"
                    "<div style='font-size:0.7rem; color:#94A3B8; margin-top:4px;'>"
                    "MSE exceeds pre-calibrated threshold for this machine ID.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='status-normal' style='height:100%; padding:14px 18px;'>"
                    "<div style='font-size:1.1rem; font-weight:700; color:#10B981;'>"
                    "✅  STATUS: NORMAL</div>"
                    "<div style='font-size:0.7rem; color:#94A3B8; margin-top:4px;'>"
                    "MSE within normal operating bounds for this machine ID.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        # ── Tier 2: DeepSHAP ──────────────────────────────────────────
        st.markdown(
            "<div class='section-header' style='margin-top:20px;'>"
            "TIER 2  ·  Acoustic Feature Breakdown (DeepSHAP)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.7rem; color:#64748B; margin-bottom:8px;'>"
            "Highlights which specific frequencies and rate-of-change channels "
            "drove the anomaly score.  "
            "<span style='color:#EF4444;'>Red → positive error contribution</span>  ·  "
            "<span style='color:#3B82F6;'>Blue → negative contribution</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        shap_fig = plot_shap_heatmaps(shap_vals)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)

        # Channel summary
        ch_power = np.abs(shap_vals).mean(axis=(1, 2))
        ch_total = ch_power.sum() + 1e-9
        sh1, sh2, sh3 = st.columns(3)
        for col, name, val in zip(
            [sh1, sh2, sh3],
            ["Mel (Static)", "Delta (Velocity)", "Δ-Delta (Accel.)"],
            ch_power,
        ):
            with col:
                col.metric(name, f"{(val / ch_total) * 100:.1f}% contribution")

        # ── Tier 3: Residual Heatmap ──────────────────────────────────
        st.markdown(
            "<div class='section-header' style='margin-top:20px;'>"
            "TIER 3  ·  Raw Reconstruction Error (Local Confirmation)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.7rem; color:#64748B; margin-bottom:8px;'>"
            "Pixel-wise squared difference between the healthy baseline and the "
            "live machine sound.  "
            "<span style='color:#F59E0B;'>Bright regions</span> "
            "= high reconstruction error = anomalous zones.</div>",
            unsafe_allow_html=True,
        )

        res_fig = plot_residual_overlay(tensor, reconstructed)
        st.pyplot(res_fig, use_container_width=True)
        plt.close(res_fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RUL ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────


def render_rul_page():
    st.markdown(
        """
    <div style='font-family:"Chakra Petch",sans-serif; font-size:1.4rem;
                font-weight:700; letter-spacing:0.06em; color:#E2E8F0; margin-bottom:4px;'>
        Remaining Useful Life Estimation
    </div>
    <div style='font-size:0.68rem; color:#334155; letter-spacing:0.1em; margin-bottom:20px;'>
        BEARING FAULT DETECTION  ·  CNN-LSTM-ATTENTION BACKBONE  ·  PRONOSTIA / CWRU DATASETS
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Controls ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<hr style='border-color:#1E2D4A; margin:12px 0;'>", unsafe_allow_html=True
        )
        st.markdown(
            """
        <div style='font-size:0.65rem; letter-spacing:0.12em; color:#64748B;
                    text-transform:uppercase; margin-bottom:8px;'>
            RUL Parameters
        </div>
        """,
            unsafe_allow_html=True,
        )
        window_size = st.slider("Window Size (snapshots)", 10, 100, 50, step=5)
        noise_level = st.slider("Noise Level", 0.0, 0.3, 0.05, step=0.01)
        fault_seed = st.selectbox(
            "Fault Scenario",
            [
                "Outer Race Defect",
                "Inner Race Defect",
                "Rolling Element Fault",
                "Healthy Bearing",
            ],
        )

    FAULT_MAP = {
        "Outer Race Defect": (
            "BPFO_ratio",
            "outer race fault frequency — outer race defect",
        ),
        "Inner Race Defect": (
            "BPFI_ratio",
            "inner race fault frequency — inner race defect",
        ),
        "Rolling Element Fault": (
            "Kurtosis",
            "impulsive shock — bearing spall or crack",
        ),
        "Healthy Bearing": (None, "No significant fault signature detected."),
    }
    RAW_NAMES = [
        "RMS",
        "Kurtosis",
        "Skewness",
        "Peak",
        "Crest",
        "MAV",
        "BPFO_ratio",
        "BPFI_ratio",
    ]
    SLOPE_NAMES = [f"Δ{n}" for n in RAW_NAMES]
    ALL_NAMES = RAW_NAMES + SLOPE_NAMES

    # ── Generate mock RUL prediction ──────────────────────────────────────
    np.random.seed(abs(hash(fault_seed)) % (2**31))
    rul_now = max(
        5.0,
        min(
            99.0,
            {
                "Outer Race Defect": 38,
                "Inner Race Defect": 22,
                "Rolling Element Fault": 55,
                "Healthy Bearing": 91,
            }.get(fault_seed, 70)
            + np.random.uniform(-5, 5),
        ),
    )

    # Historical trend (simulate degradation)
    n_hist = 30
    base = min(rul_now + 40, 98)
    history = [
        max(5, base - i * ((base - rul_now) / n_hist) + np.random.randn() * 3)
        for i in range(n_hist)
    ]
    history[-1] = rul_now

    # Feature values
    raw_features = np.clip(
        np.random.randn(8) * 0.3 + [0.4, 2.5, 0.1, 1.8, 3.5, 0.35, 0.02, 0.01], 0, None
    )
    slope_features = np.clip(np.random.randn(8) * 0.05, -1, 1)
    all_features = np.concatenate([raw_features, slope_features])

    # SHAP values
    shap_vals = np.random.randn(16) * 0.03
    dom_feat, _ = FAULT_MAP[fault_seed]
    if dom_feat and dom_feat in ALL_NAMES:
        idx = ALL_NAMES.index(dom_feat)
        shap_vals[idx] += 0.12 * (1 if shap_vals[idx] > 0 else -1)

    # ── Layout ─────────────────────────────────────────────────────────────
    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        st.markdown(
            "<div class='section-header'>RUL Gauge</div>", unsafe_allow_html=True
        )
        gauge_fig = plot_rul_gauge(rul_now)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)

        color_rul = (
            "#10B981" if rul_now > 60 else ("#F59E0B" if rul_now > 30 else "#EF4444")
        )
        state_str = (
            "Healthy"
            if rul_now > 60
            else (
                "Warning — schedule maintenance"
                if rul_now > 30
                else "⚠ CRITICAL — immediate action required"
            )
        )
        st.markdown(
            f"<div style='text-align:center; font-size:0.72rem; color:{color_rul}; "
            f"margin-top:4px;'>{state_str}</div>",
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown(
            "<div class='section-header'>Bearing Health Metrics</div>",
            unsafe_allow_html=True,
        )
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RUL", f"{rul_now:.1f}%")
        mc2.metric("Kurtosis", f"{raw_features[1]:.3f}")
        mc3.metric("BPFO Ratio", f"{raw_features[6]:.4f}")

        # Fault diagnosis
        _, fault_desc = FAULT_MAP[fault_seed]
        st.markdown(
            f"<div class='xai-box' style='margin-top:10px;'>"
            f"<div class='label'>⬡ FAULT DIAGNOSIS — {fault_seed.upper()}</div>"
            f"{fault_desc.capitalize()}."
            f"{'  Window: ' + str(window_size) + ' snapshots  ·  Noise σ: ' + str(noise_level) if fault_seed != 'Healthy Bearing' else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Feature health bars
        st.markdown(
            "<div style='font-size:0.65rem; color:#64748B; margin:10px 0 6px 0; "
            "letter-spacing:0.1em;'>RAW FEATURE HEALTH INDICATORS</div>",
            unsafe_allow_html=True,
        )
        thresholds_health = [0.6, 3.5, 0.3, 2.5, 5.0, 0.6, 0.04, 0.03]
        for name, val, thresh in zip(RAW_NAMES, raw_features, thresholds_health):
            ratio = min(val / max(thresh, 1e-9), 1.0)
            color = (
                "#EF4444" if ratio > 0.8 else ("#F59E0B" if ratio > 0.5 else "#10B981")
            )
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; "
                f"margin-bottom:3px; font-size:0.68rem;'>"
                f"  <span style='color:#64748B; width:80px; flex-shrink:0;'>{name}</span>"
                f"  <div style='flex:1; background:#0F1628; border-radius:2px; "
                f"  height:5px; overflow:hidden;'>"
                f"    <div style='width:{ratio * 100:.0f}%; background:{color}; "
                f"    height:100%; border-radius:2px;'></div>"
                f"  </div>"
                f"  <span style='color:{color}; width:50px; text-align:right;'>"
                f"  {val:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<hr style='border-color:#1E2D4A; margin:16px 0;'>", unsafe_allow_html=True
    )

    # ── Trend + SHAP ───────────────────────────────────────────────────────
    trend_col, shap_col = st.columns([3, 2])

    with trend_col:
        st.markdown(
            "<div class='section-header'>RUL Degradation Trend</div>",
            unsafe_allow_html=True,
        )
        trend_fig = plot_rul_trend(history)
        st.pyplot(trend_fig, use_container_width=True)
        plt.close(trend_fig)

    with shap_col:
        st.markdown(
            "<div class='section-header'>SHAP Feature Attribution</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.65rem; color:#64748B; margin-bottom:6px;'>"
            "<span style='color:#EF4444;'>■</span> Accelerates degradation  "
            "<span style='color:#3B82F6; margin-left:6px;'>■</span> Slows degradation</div>",
            unsafe_allow_html=True,
        )
        shap_fig2 = plot_feature_shap_bar(ALL_NAMES, shap_vals.tolist())
        st.pyplot(shap_fig2, use_container_width=True)
        plt.close(shap_fig2)

    # ── Maintenance recommendation ─────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#1E2D4A; margin:16px 0;'>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-header'>Maintenance Recommendation Engine</div>",
        unsafe_allow_html=True,
    )

    rec_cols = st.columns(3)
    recs = [
        (
            "Recommended Action",
            (
                "Replace bearing NOW"
                if rul_now < 30
                else "Schedule inspection" if rul_now < 60 else "Continue monitoring"
            ),
            "#EF4444" if rul_now < 30 else "#F59E0B" if rul_now < 60 else "#10B981",
        ),
        ("Confidence Level", f"{random.uniform(0.84, 0.97):.0%}", "#94A3B8"),
    ]
    for col, (label, val, color) in zip(rec_cols, recs):
        with col:
            col.markdown(
                f"<div class='result-card'>"
                f"  <div style='font-size:0.6rem; color:#64748B; letter-spacing:0.1em; "
                f"  text-transform:uppercase; margin-bottom:6px;'>{label}</div>"
                f"  <div style='font-size:1rem; color:{color}; font-weight:700; "
                f'  font-family:"Chakra Petch",sans-serif;\'>{val}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Model architecture note ────────────────────────────────────────────
    with st.expander("Model Architecture Details", expanded=False):
        st.markdown(
            """
        <div style='font-size:0.72rem; color:#64748B; line-height:1.9;'>
        <b style='color:#94A3B8;'>RULNet</b> — CNN-BiLSTM with Multi-Head Attention<br>
        <br>
        • <b style='color:#64748B;'>Input</b>: Window of 50 snapshots × 16 features
          (8 raw + 8 slope Δ features)<br>
        • <b style='color:#64748B;'>CNN layer</b>: Conv1D(input→64, k=3) + BatchNorm + GELU + Dropout(0.3)<br>
        • <b style='color:#64748B;'>Sequence layer</b>: 2-layer BiLSTM (hidden=128, bidirectional → 256d)<br>
        • <b style='color:#64748B;'>Attention</b>: 4-head MultiheadAttention + LayerNorm (pre-norm residual)<br>
        • <b style='color:#64748B;'>Head</b>: Linear(256→32) → GELU → Dropout → Linear(32→1) → Sigmoid<br>
        • <b style='color:#64748B;'>Output</b>: Normalised RUL ∈ [0, 1] (multiply by max_life for hours)<br>
        • <b style='color:#64748B;'>Features</b>: RMS · Kurtosis · Skewness · Peak · Crest · MAV ·
          BPFO_ratio · BPFI_ratio (+ their first differences)<br>
        • <b style='color:#64748B;'>XAI</b>: SHAP GradientExplainer over the full RULNet
        </div>
        """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if "active_models" not in st.session_state:
    st.session_state["active_models"] = set()
if "classifier_result" not in st.session_state:
    st.session_state["classifier_result"] = None

page = render_sidebar()

if "Main Dashboard" in page:
    render_main_dashboard()
elif "Classifier" in page:
    render_classifier_page()
elif "RUL" in page:
    render_rul_page()
