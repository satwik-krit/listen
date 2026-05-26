"""
L.I.S.T.E.N. — RUL Dashboard  (app.py)
Run from the listen_person3/code/ folder:
    cd listen_person3/code
    streamlit run app.py

Required files in the same folder as app.py:
    predict.py, best_model.pt, scaler.pkl,
    window_size.npy, n_features.npy, n_raw.npy

NASA data expected at:  ../nasa_data/2nd_test/
"""

import os
import sys
import io

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Ensure predict.py is importable ────────────────
# Add the folder that contains app.py (and predict.py) to the path.
# This works regardless of where streamlit is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NASA data lives one level up from code/
_NASA_DIR = os.path.join(_HERE, "..", "nasa_data", "2nd_test")
_NASA_DIR = os.path.normpath(_NASA_DIR)


# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="L.I.S.T.E.N. — RUL Engine",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp { background-color: #0a0e17; color: #c8d6e5; }

section[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2d45;
}

.metric-card {
    background: linear-gradient(135deg, #0f1923 0%, #111d2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066ff);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a7fa5;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #4a7fa5;
    margin-top: 0.3rem;
    font-family: 'Share Tech Mono', monospace;
}

.badge-healthy {
    background: linear-gradient(90deg, #003d1f, #005c2e);
    border: 1px solid #00cc66; color: #00ff88;
    border-radius: 3px; padding: 0.5rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem; letter-spacing: 0.15em; display: inline-block;
}
.badge-warning {
    background: linear-gradient(90deg, #3d2800, #5c3d00);
    border: 1px solid #cc7700; color: #ffaa00;
    border-radius: 3px; padding: 0.5rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem; letter-spacing: 0.15em; display: inline-block;
}
.badge-critical {
    background: linear-gradient(90deg, #3d0000, #5c0000);
    border: 1px solid #cc0000; color: #ff4444;
    border-radius: 3px; padding: 0.5rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem; letter-spacing: 0.15em; display: inline-block;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { border-color: #cc0000; box-shadow: 0 0 0px #cc0000; }
    50%       { border-color: #ff4444; box-shadow: 0 0 12px #ff000055; }
}

.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.2em; color: #0077cc;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem; margin-bottom: 1rem; margin-top: 1.5rem;
}

.explanation-box {
    background: #0a141f;
    border-left: 3px solid #0077cc;
    padding: 1rem 1.2rem;
    border-radius: 0 4px 4px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem; line-height: 1.6; color: #8ab4cc;
}

.main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem; color: #00d4ff;
    letter-spacing: 0.08em; line-height: 1.1;
}
.main-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem; color: #3a6080;
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.2rem;
}

.stButton > button {
    background: linear-gradient(90deg, #003d66, #005c99);
    color: #00d4ff; border: 1px solid #0077cc; border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem; letter-spacing: 0.1em;
    padding: 0.5rem 1.2rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #005c99, #0077cc);
    border-color: #00d4ff; box-shadow: 0 0 12px #0077cc55;
}

hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# LOAD MODEL  — cached, runs only once per session
# FIX: removed os.chdir() which was breaking all
#      relative paths throughout the rest of the app.
#      We now use sys.path + absolute paths instead.
# ════════════════════════════════════════════════════
@st.cache_resource
def load_everything():
    try:
        import joblib
        from predict import load_model, predict_rul, extract_features_from_signal

        # All artifact files sit next to app.py in _HERE
        model       = load_model(
            model_path     = os.path.join(_HERE, "best_model.pt"),
            n_features_path= os.path.join(_HERE, "n_features.npy"),
        )
        scaler      = joblib.load(os.path.join(_HERE, "scaler.pkl"))
        window_size = int(np.load(os.path.join(_HERE, "window_size.npy"))[0])

        return model, scaler, window_size, extract_features_from_signal, predict_rul, None

    except Exception as e:
        return None, None, 50, None, None, str(e)


# ════════════════════════════════════════════════════
# LOAD DEMO SAMPLE
# FIX: _extract_fn removed from signature — calling
#      the imported function directly avoids Streamlit
#      trying to hash an un-hashable function object.
# ════════════════════════════════════════════════════
@st.cache_data
def load_demo_sample(sample_type: str, window_size: int):
    """
    Loads a window of features from NASA Run 2.
    Returns (raw_window np.array shape (W,8), label str) or (None, error_str).
    """
    # Import here so the cache key doesn't involve function objects
    from predict import extract_features_from_signal

    data_dir = _NASA_DIR
    if not os.path.exists(data_dir):
        return None, f"Demo data folder not found:\n{data_dir}"

    all_files = sorted(os.listdir(data_dir))
    if len(all_files) < window_size:
        return None, f"Not enough files in {data_dir} (found {len(all_files)}, need {window_size})"

    if sample_type == "critical":
        selected = all_files[-window_size:]
        label    = f"Near-Failure Sample — last {window_size} snapshots of NASA Run 2"
    elif sample_type == "warning":
        mid      = int(len(all_files) * 0.72)
        selected = all_files[mid: mid + window_size]
        label    = f"Mid-Degradation Sample — ~72% through NASA Run 2"
    else:  # healthy
        selected = all_files[:window_size]
        label    = f"Healthy Sample — first {window_size} snapshots of NASA Run 2"

    window_data = []
    for fname in selected:
        try:
            df  = pd.read_csv(os.path.join(data_dir, fname), sep="\t", header=None)
            sig = df.iloc[:, 0].values.astype(np.float32)
            window_data.append(extract_features_from_signal(sig))
        except Exception as e:
            return None, f"Failed reading {fname}: {e}"

    return np.array(window_data, dtype=np.float32), label


# ════════════════════════════════════════════════════
# RENDER RESULT
# ════════════════════════════════════════════════════
def render_result(result: dict):
    rul_pct   = result["rul_percent"]
    rul_norm  = result["rul_normalized"]
    status    = result["health_status"]
    top_feats = result["top_features"]
    expl      = result["explanation"]

    # ── Status row ───────────────────────────────────
    st.markdown('<div class="section-header">▸ BEARING HEALTH STATUS</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    badge_class = {"HEALTHY": "badge-healthy",
                   "WARNING": "badge-warning",
                   "CRITICAL": "badge-critical"}.get(status, "badge-warning")
    icon = {"HEALTHY": "●", "WARNING": "◆", "CRITICAL": "▲"}.get(status, "●")
    color = {"HEALTHY": "#00ff88",
             "WARNING": "#ffaa00",
             "CRITICAL": "#ff4444"}.get(status, "#ffaa00")

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">STATUS</div>'
            f'<div style="margin-top:0.5rem">'
            f'<span class="{badge_class}">{icon} {status}</span>'
            f'</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">HEALTH INDEX</div>'
            f'<div class="metric-value" style="color:{color}">'
            f'{rul_pct:.1f}<span style="font-size:1.2rem">%</span>'
            f'</div>'
            f'<div class="metric-sub">normalized remaining life</div>'
            f'</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">RUL SCORE</div>'
            f'<div class="metric-value" style="color:#00d4ff">'
            f'{rul_norm:.3f}'
            f'</div>'
            f'<div class="metric-sub">0.000 = failure &nbsp;/&nbsp; 1.000 = healthy</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Health bar (matplotlib, dark themed) ─────────
    bar_color = {"HEALTHY": "#00cc66",
                 "WARNING": "#cc7700",
                 "CRITICAL": "#cc0000"}.get(status, "#cc7700")

    fig_bar, ax_bar = plt.subplots(figsize=(10, 0.45))
    fig_bar.patch.set_facecolor('#0a141f')
    ax_bar.set_facecolor('#0d1a27')
    ax_bar.barh([0], [100],   color='#0d1a27',  height=0.8, edgecolor='#1e3a5f')
    ax_bar.barh([0], [rul_pct], color=bar_color, height=0.8, alpha=0.9)
    ax_bar.set_xlim(0, 100)
    ax_bar.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # ── SHAP + Explanation ───────────────────────────
    st.markdown('<div class="section-header">▸ FAULT INDICATOR ANALYSIS (SHAP)</div>',
                unsafe_allow_html=True)

    col_shap, col_expl = st.columns([1.2, 1])

    with col_shap:
        names  = [f[0] for f in top_feats]
        scores = [f[1] for f in top_feats]

        feat_colors = []
        for n in names:
            if "Kurtosis" in n or "Crest" in n or "Peak" in n:
                feat_colors.append("#ff4444")
            elif "BPFO" in n:
                feat_colors.append("#ff8800")
            elif "BPFI" in n:
                feat_colors.append("#ffaa00")
            elif "RMS" in n or "MAV" in n:
                feat_colors.append("#00aaff")
            else:
                feat_colors.append("#44aaff")

        fig_shap, ax = plt.subplots(figsize=(7, 3.5))
        fig_shap.patch.set_facecolor('#0a141f')
        ax.set_facecolor('#0a141f')

        ax.barh(range(len(names)), scores[::-1],
                color=feat_colors[::-1], edgecolor='none',
                height=0.55, alpha=0.9)

        max_score = max(scores) if scores else 1
        for i, score in enumerate(scores[::-1]):
            ax.text(score + max_score * 0.02, i, f'{score:.4f}',
                    va='center', ha='left', color='#8ab4cc',
                    fontsize=8, fontfamily='monospace')

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], color='#8ab4cc',
                           fontsize=9, fontfamily='monospace')
        ax.set_xlabel('Mean |SHAP Value|', color='#4a7fa5',
                      fontsize=8, fontfamily='monospace')
        ax.set_title('Feature Importance — Why This Prediction?',
                     color='#00d4ff', fontsize=9,
                     fontfamily='monospace', pad=10)
        ax.tick_params(colors='#4a7fa5', labelsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#1e3a5f')
        ax.spines['bottom'].set_color('#1e3a5f')
        ax.set_xlim(0, max_score * 1.3)
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

    with col_expl:
        st.markdown('<div class="section-header">▸ DIAGNOSTIC REPORT</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="explanation-box">{expl}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">▸ TOP INDICATORS</div>',
                    unsafe_allow_html=True)

        rank_colors = ["#ff4444", "#ff8800", "#ffaa00", "#00aaff", "#44aaff"]
        for i, (feat, score) in enumerate(top_feats):
            filled  = int((score / top_feats[0][1]) * 12) if top_feats[0][1] > 0 else 0
            bar_str = "█" * filled + "░" * (12 - filled)
            rc      = rank_colors[i]
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;'
                f'font-size:0.78rem;margin:0.25rem 0;color:#8ab4cc;">'
                f'<span style="color:{rc}">#{i+1}</span> '
                f'{feat:<16} '
                f'<span style="color:{rc}">{bar_str}</span> '
                f'<span style="color:#4a7fa5">{score:.4f}</span>'
                f'</div>', unsafe_allow_html=True)

        # Recommendation box
        st.markdown("<br>", unsafe_allow_html=True)
        rec_map = {
            "HEALTHY":  ("#003d1f", "#00cc66",
                         "✓ Continue normal operation. Schedule routine inspection in 30 days."),
            "WARNING":  ("#3d2800", "#cc7700",
                         "⚠ Increase monitoring frequency. Plan maintenance within 2 weeks."),
            "CRITICAL": ("#3d0000", "#cc0000",
                         "✕ Schedule immediate maintenance. High risk of unplanned failure."),
        }
        bg, border, msg = rec_map.get(status, rec_map["WARNING"])
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-radius:4px;padding:0.8rem 1rem;'
            f'font-family:\'Share Tech Mono\',monospace;'
            f'font-size:0.78rem;color:#c8d6e5;line-height:1.5;">'
            f'{msg}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════
def main():
    # ── Header ──────────────────────────────────────
    st.markdown(
        '<div class="main-title">L.I.S.T.E.N.</div>'
        '<div class="main-subtitle">'
        'Latent Inference of Sequential Temporal Energy Networks'
        ' — RUL Prediction Engine</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Load model ───────────────────────────────────
    model, scaler, window_size, extract_fn, predict_fn, err = load_everything()

    if err or model is None:
        st.error(f"⚠️ Model failed to load: {err}")
        st.markdown("""
**Make sure these files are all in the same folder as `app.py`:**
- `predict.py`
- `best_model.pt`
- `scaler.pkl`
- `window_size.npy`
- `n_features.npy`
- `n_raw.npy`

**Then run from that folder:**
```
cd listen_person3/code
streamlit run app.py
```
        """)
        return

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'color:#00d4ff;font-size:1rem;letter-spacing:0.1em;">'
            '⬡ SYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'font-size:0.72rem;color:#00cc66;margin-top:0.3rem;">'
            '● MODEL LOADED<br>● SHAP ENGINE ACTIVE'
            '</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'color:#4a7fa5;font-size:0.7rem;letter-spacing:0.12em;">'
            'ARCHITECTURE</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'font-size:0.72rem;color:#8ab4cc;line-height:1.8;">'
            f'CNN → BiLSTM → Attention<br>'
            f'Window: {window_size} snapshots<br>'
            f'Features: 16 per timestep<br>'
            f'Dataset: NASA IMS Bearing<br>'
            f'Bearing: Rexnord ZA-2115<br>'
            f'Speed: 2,000 RPM'
            f'</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'color:#4a7fa5;font-size:0.7rem;letter-spacing:0.12em;">'
            'FAULT FREQUENCIES</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'font-size:0.72rem;color:#8ab4cc;line-height:1.8;">'
            'BPFO: 161.1 Hz (outer race)<br>'
            'BPFI: 236.4 Hz (inner race)'
            '</div>', unsafe_allow_html=True)

    # ── Input section ────────────────────────────────
    st.markdown('<div class="section-header">▸ INPUT SOURCE</div>',
                unsafe_allow_html=True)

    input_mode = st.radio(
        "",
        ["🟢 Demo — Healthy Bearing",
         "🟡 Demo — Mid-Degradation",
         "🔴 Demo — Near-Failure Bearing",
         "📁 Upload Vibration File"],
        horizontal=True,
        label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    result     = None
    sample_lbl = ""

    # ── Demo path ────────────────────────────────────
    if "Demo" in input_mode:
        demo_map = {
            "🟢 Demo — Healthy Bearing":      "healthy",
            "🟡 Demo — Mid-Degradation":       "warning",
            "🔴 Demo — Near-Failure Bearing":  "critical",
        }
        sample_key = demo_map[input_mode]

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_demo = st.button("▶  RUN ANALYSIS", use_container_width=True)
        with col_info:
            desc = {
                "healthy":  "First 50 snapshots of NASA Run 2 — bearing in normal operation",
                "warning":  "Snapshots at ~72% through NASA Run 2 — early degradation phase",
                "critical": "Last 50 snapshots of NASA Run 2 — bearing near outer race failure",
            }[sample_key]
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;'
                f'font-size:0.78rem;color:#4a7fa5;padding-top:0.6rem;">'
                f'↳ {desc}</div>', unsafe_allow_html=True)

        if run_demo:
            with st.spinner("Extracting features and running inference..."):
                raw_window, sample_lbl = load_demo_sample(sample_key, window_size)

            if raw_window is None:
                st.error(f"Could not load demo data: {sample_lbl}")
            else:
                with st.spinner("Computing SHAP explanations..."):
                    result = predict_fn(raw_window, model, scaler)
                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace;'
                    f'font-size:0.72rem;color:#4a7fa5;margin-bottom:1rem;">'
                    f'SAMPLE: {sample_lbl}</div>', unsafe_allow_html=True)

    # ── Upload path ──────────────────────────────────
    else:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;'
            'font-size:0.78rem;color:#4a7fa5;margin-bottom:0.5rem;">'
            'Upload a single tab-separated vibration file '
            '(20,480 rows, 20 kHz). '
            'It will be treated as one snapshot replicated across the window.'
            '</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop vibration file here",
            type=["csv", "txt", "dat"],
            label_visibility="collapsed")

        if uploaded is not None:
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;'
                f'font-size:0.72rem;color:#4a7fa5;margin:0.5rem 0;">'
                f'FILE: {uploaded.name} | SIZE: {uploaded.size / 1024:.1f} KB'
                f'</div>', unsafe_allow_html=True)

            # Cache the bytes so re-runs don't re-read the file
            if st.session_state.get("upload_name") != uploaded.name:
                st.session_state["upload_bytes"] = uploaded.read()
                st.session_state["upload_name"]  = uploaded.name

            col_b1, _ = st.columns([1, 4])
            with col_b1:
                run_upload = st.button("▶  ANALYZE FILE", use_container_width=True)

            if run_upload:
                try:
                    raw_bytes = io.BytesIO(st.session_state["upload_bytes"])
                    df_up     = pd.read_csv(raw_bytes, sep="\t", header=None)
                    signal    = df_up.iloc[:, 0].values.astype(np.float32)

                    if len(signal) < 20480:
                        st.warning(f"File has {len(signal)} samples — expected 20,480. "
                                   "Results may be less accurate.")
                    signal = signal[:20480] if len(signal) >= 20480 \
                             else np.pad(signal, (0, 20480 - len(signal)))

                    st.info(f"Single file detected — "
                            f"replicating across {window_size} window positions.")

                    with st.spinner("Extracting features..."):
                        feats      = extract_fn(signal)
                        raw_window = np.tile(feats, (window_size, 1))

                    with st.spinner("Running model + SHAP..."):
                        result     = predict_fn(raw_window, model, scaler)
                    sample_lbl = uploaded.name

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ── Render result ────────────────────────────────
    if result is not None:
        st.markdown("---")
        render_result(result)

        with st.expander("▸ RAW FEATURE VALUES"):
            st.dataframe(
                pd.DataFrame({
                    "Rank":       [f"#{i+1}" for i in range(len(result["top_features"]))],
                    "Feature":    [f[0] for f in result["top_features"]],
                    "Importance": [f"{f[1]:.6f}" for f in result["top_features"]],
                }),
                use_container_width=True, hide_index=True)

        with st.expander("▸ TECHNICAL DETAILS"):
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;'
                f'font-size:0.78rem;color:#8ab4cc;line-height:2;">'
                f'RUL Normalized : {result["rul_normalized"]:.6f}<br>'
                f'RUL Percent    : {result["rul_percent"]}%<br>'
                f'Health Status  : {result["health_status"]}<br>'
                f'Window Size    : {window_size} snapshots<br>'
                f'Features/step  : 16 (8 raw + 8 slopes)<br>'
                f'Model          : CNN-BiLSTM-Attention<br>'
                f'Explainability : SHAP GradientExplainer<br>'
                f'Dataset        : NASA IMS Bearing (Rexnord ZA-2115)'
                f'</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem;'
            'font-family:\'Share Tech Mono\',monospace;'
            'color:#1e3a5f;font-size:0.85rem;letter-spacing:0.1em;">'
            '[ SELECT A DEMO OR UPLOAD A FILE AND CLICK RUN ANALYSIS ]'
            '</div>', unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;'
        'font-size:0.65rem;color:#1e3a5f;text-align:center;letter-spacing:0.1em;">'
        'L.I.S.T.E.N. © 2024 — Kode Loverzz — '
        'Rishab Satpathy · Chitresh Ranjan · Sanidhya Jain · Satwik Gupta'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
