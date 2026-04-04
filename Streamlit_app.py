
import streamlit as st
import torch
from pathlib import Path
from xai_outputs import run_xai, XAIOutputBundle
from XAI_layer import ProjectBRegistry

# ─────────────────────────────────────────────────────────────
# CONFIG  — edit these paths
# ─────────────────────────────────────────────────────────────
EDGE_DEPLOYMENTS = Path(r"C:\Users\risha\Downloads\listen\listen\edge_deployments")
MACHINE_TYPES    = ["fan", "pump", "slider", "valve"]
MACHINE_IDS      = ["id_00", "id_02", "id_04", "id_06"]

# ─────────────────────────────────────────────────────────────
# LOAD REGISTRY (cached — loads once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_registry() -> ProjectBRegistry:
    return ProjectBRegistry(
        root_dir      = EDGE_DEPLOYMENTS,
        n_background  = 50,
        shap_nsamples = 50,
        device        = "cpu",
    )

# ─────────────────────────────────────────────────────────────
# SIDEBAR  — controls
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Project B XAI", layout="wide")
st.title("Project B — XAI Anomaly Dashboard")

with st.sidebar:
    st.header("Settings")
    machine_type = st.selectbox("Machine Type", MACHINE_TYPES)
    machine_id   = st.selectbox("Machine ID",   MACHINE_IDS)
    uploaded     = st.file_uploader(
        "Upload spectrogram (.npy / .npz / .pt / .pth)",
        type=["npy", "npz", "pt", "pth"],
    )
    run_btn = st.button("▶  Run XAI", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# RUN  — triggered by button
# ─────────────────────────────────────────────────────────────
if run_btn and uploaded:
    # Save upload to a temp path (run_xai needs a file path)
    tmp_path = Path(f"/tmp/{uploaded.name}")
    tmp_path.write_bytes(uploaded.read())

    registry = load_registry()

    with st.spinner("Running inference + SHAP …"):
        bundle: XAIOutputBundle = run_xai(
            file_path    = tmp_path,
            machine_type = machine_type,
            machine_id   = machine_id,
            registry     = registry,
        )

    # ── OUTPUT 4 : Alert Banner ───────────────────────────────
    st.markdown(bundle.alert_banner.to_html(), unsafe_allow_html=True)
    st.write("")   # spacer

    # ── OUTPUT 3 : Score Readout + Gauge ─────────────────────
    col_gauge, col_score = st.columns([2, 1])
    with col_gauge:
        st.plotly_chart(bundle.score_readout.gauge_fig, use_container_width=True)
    with col_score:
        st.metric("Anomaly Score", f"{bundle.score_readout.anomaly_score:.6f}")
        st.metric("Threshold",     f"{bundle.score_readout.threshold:.6f}")
        st.metric("Decision",
                  "ANOMALY" if bundle.score_readout.is_anomaly else "NORMAL")

    st.divider()

    # ── OUTPUT 2 : Channel Verdict ────────────────────────────
    st.subheader("Channel Contribution (SHAP)")
    verdict_cols = st.columns(3)
    for i, (ch_name, pct) in enumerate(bundle.channel_verdict.items()):
        with verdict_cols[i]:
            st.metric(ch_name, f"{pct:.1f}%")
            st.progress(int(pct))

    st.divider()

    # ── OUTPUT 1 : 3-D SHAP per channel ──────────────────────
    st.subheader("3-D SHAP Attribution Map")
    st.plotly_chart(bundle.shap_3d_fig, use_container_width=True)

    st.divider()

    # ── OUTPUT 5 & 6 : Overlay + Heatmap side-by-side ─────────
    col_overlay, col_heatmap = st.columns(2)
    with col_overlay:
        st.subheader("3-Channel Residual Overlay")
        st.plotly_chart(bundle.overlay_fig, use_container_width=True)
    with col_heatmap:
        st.subheader("🗺️ Single Heatmap View")
        st.plotly_chart(bundle.heatmap_fig, use_container_width=True)

    # ── Raw JSON (collapsible) ────────────────────────────────
    with st.expander("Raw numeric output"):
        st.json({
            **bundle.score_readout.as_dict(),
            "channel_verdict": bundle.channel_verdict,
            "alert": bundle.alert_banner.as_dict(),
        })

elif run_btn and not uploaded:
    st.warning("Please upload a spectrogram file first.")
else:
    st.info("Upload a spectrogram file and press **Run XAI** to begin.")