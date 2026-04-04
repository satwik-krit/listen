
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# ── optional heavy deps (guarded) ────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
    warnings.warn("[xai_outputs] plotly not installed — figure outputs will be None.")

# ── local import ──────────────────────────────────────────────────────────────
from XAI_layer import ProjectBXAI, ProjectBRegistry, XAIResult

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = [
    "Mel (static frequency anomaly)",
    "Delta (rate of change)",
    "Delta-Delta (acceleration)",
]

CHANNEL_COLOURS = ["#636EFA", "#EF553B", "#00CC96"]  

SEVERITY_LEVELS = [
    # (upper_fraction_of_threshold, label, hex_colour, emoji)
    (0.80,  "NORMAL",       "#1f8b4c", "✅"),
    (1.00,  "BORDERLINE",   "#f0a500", "⚠️"),
    (1.50,  "ANOMALY",      "#e74c3c", "🚨"),
    (float("inf"), "CRITICAL", "#7b0000", "🔴"),
]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreReadout:
    """Numeric score metadata + a Plotly gauge figure."""
    anomaly_score: float
    threshold:     float
    is_anomaly:    bool
    gauge_fig:     object   # plotly.graph_objects.Figure | None

    def as_dict(self) -> dict:
        return {
            "anomaly_score": self.anomaly_score,
            "threshold":     self.threshold,
            "is_anomaly":    self.is_anomaly,
        }


@dataclass
class AlertBanner:
    """
    Severity classification for the UI banner.

    Fields
    ------
    severity  : 'NORMAL' | 'BORDERLINE' | 'ANOMALY' | 'CRITICAL'
    label     : human-readable text with emoji
    colour    : CSS hex colour (background)
    text_colour : CSS hex colour (foreground)
    emoji     : standalone emoji string
    """
    severity:     str
    label:        str
    colour:       str
    text_colour:  str
    emoji:        str

    def to_html(self) -> str:
        """Returns an inline-styled HTML div ready for st.markdown(…, unsafe_allow_html=True)."""
        return (
            f'<div style="background:{self.colour};color:{self.text_colour};'
            f'padding:14px 20px;border-radius:8px;font-size:1.3em;font-weight:700;'
            f'text-align:center;letter-spacing:0.05em;">'
            f'{self.emoji}  {self.label}'
            f'</div>'
        )

    def as_dict(self) -> dict:
        return {
            "severity":    self.severity,
            "label":       self.label,
            "colour":      self.colour,
            "text_colour": self.text_colour,
            "emoji":       self.emoji,
        }


@dataclass
class XAIOutputBundle:
    """
    Complete set of outputs produced by run_xai().
    Every field is independently consumable by Streamlit or any other UI.
    """
    # ── raw result (for programmatic access) ──────────────────────────────
    result:          XAIResult

    # ── 1. 3-D SHAP per channel ───────────────────────────────────────────
    shap_3d_fig:     object          # Plotly Figure | None

    # ── 2. Channel verdict ────────────────────────────────────────────────
    channel_verdict: dict            # {channel_name: pct_float}

    # ── 3. Score readout + gauge ──────────────────────────────────────────
    score_readout:   ScoreReadout

    # ── 4. Alert banner ───────────────────────────────────────────────────
    alert_banner:    AlertBanner

    # ── 5. 3-channel overlay on spectrogram ───────────────────────────────
    overlay_fig:     object          # Plotly Figure | None

    # ── 6. Single merged SHAP heatmap ─────────────────────────────────────
    heatmap_fig:     object          # Plotly Figure | None


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _classify_severity(score: float, threshold: float) -> AlertBanner:
    ratio = score / (threshold + 1e-12)
    for upper, label, colour, emoji in SEVERITY_LEVELS:
        if ratio <= upper:
            text_colour = "#ffffff" if label != "BORDERLINE" else "#1a1a1a"
            return AlertBanner(
                severity=label,
                label=f"{emoji}  {label}  (score={score:.5f}, threshold={threshold:.5f})",
                colour=colour,
                text_colour=text_colour,
                emoji=emoji,
            )
    # fallback (should never reach)
    return AlertBanner("CRITICAL", f" CRITICAL", "#7b0000", "#ffffff")


def _build_gauge(score: float, threshold: float, is_anomaly: bool) -> Optional[object]:
    if not _PLOTLY:
        return None

    max_val  = threshold * 2.5
    bar_col  = "#e74c3c" if is_anomaly else "#1f8b4c"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": threshold, "increasing": {"color": "#e74c3c"},
                 "decreasing": {"color": "#1f8b4c"}},
        gauge = {
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar":  {"color": bar_col},
            "steps": [
                {"range": [0,              threshold * 0.80], "color": "#d5f5e3"},
                {"range": [threshold*0.80, threshold],        "color": "#fdebd0"},
                {"range": [threshold,      threshold * 1.50], "color": "#fadbd8"},
                {"range": [threshold*1.50, max_val],          "color": "#922b21"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.85,
                "value": threshold,
            },
        },
        title = {"text": "Anomaly Score", "font": {"size": 18}},
        number= {"font": {"size": 28}, "valueformat": ".5f"},
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_shap_3d(shap_map: np.ndarray) -> Optional[object]:
    """3-D surface plot: one translucent surface per channel."""
    if not _PLOTLY:
        return None

    C, H, W = shap_map.shape
    x = np.arange(W)
    y = np.arange(H)

    fig = go.Figure()
    for c in range(C):
        z = shap_map[c]                       # (H, W)
        fig.add_trace(go.Surface(
            z=z, x=x, y=y,
            colorscale=[[0, "#2c3e50"], [0.5, CHANNEL_COLOURS[c]], [1, "#ffffff"]],
            opacity=0.72,
            showscale=(c == 0),
            name=CHANNEL_NAMES[c],
            hovertemplate=(
                f"<b>{CHANNEL_NAMES[c]}</b><br>"
                "Freq bin: %{y}<br>Time bin: %{x}<br>SHAP: %{z:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="3-D SHAP Attribution Map (per channel)",
        scene=dict(
            xaxis_title="Time bins",
            yaxis_title="Frequency bins",
            zaxis_title="SHAP value",
            bgcolor="rgba(0,0,0,0)",
        ),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_overlay(input_spec: np.ndarray, residual_map: np.ndarray) -> Optional[object]:
   
    if not _PLOTLY:
        return None

    C, H, W = input_spec.shape
    fig = make_subplots(
        rows=1, cols=C,
        subplot_titles=CHANNEL_NAMES,
        horizontal_spacing=0.04,
    )

    for c in range(C):
        spec_norm = (input_spec[c] - input_spec[c].min()) / (
            input_spec[c].ptp() + 1e-8
        )
        res_norm  = (residual_map[c] - residual_map[c].min()) / (
            residual_map[c].ptp() + 1e-8
        )

        # Base spectrogram (grey)
        fig.add_trace(
            go.Heatmap(
                z=spec_norm,
                colorscale="gray",
                showscale=False,
                name=f"Spec ch{c}",
                hoverinfo="skip",
            ),
            row=1, col=c + 1,
        )
        # Residual overlay (coloured, semi-transparent via opacity)
        fig.add_trace(
            go.Heatmap(
                z=res_norm,
                colorscale="Hot",
                opacity=0.55,
                showscale=(c == C - 1),
                colorbar=dict(title="Residual", len=0.8) if c == C - 1 else None,
                name=CHANNEL_NAMES[c],
                hovertemplate=(
                    f"<b>{CHANNEL_NAMES[c]}</b><br>"
                    "Freq: %{y}  Time: %{x}<br>Residual: %{z:.4f}<extra></extra>"
                ),
            ),
            row=1, col=c + 1,
        )

    fig.update_layout(
        title="3-Channel Residual Overlay on Spectrogram",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def _build_heatmap(shap_map: np.ndarray, residual_2d: np.ndarray) -> Optional[object]:
   
    if not _PLOTLY:
        return None

    merged_shap = np.abs(shap_map).mean(axis=0)        # (H, W)
    merged_shap_norm = (merged_shap - merged_shap.min()) / (
        merged_shap.ptp() + 1e-8
    )
    residual_norm = (residual_2d - residual_2d.min()) / (
        residual_2d.ptp() + 1e-8
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Mean |SHAP| Attribution", "Mean Residual (reconstruction error)"],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=merged_shap_norm,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title="|SHAP|", len=0.45, y=0.77),
            hovertemplate="Freq: %{y}  Time: %{x}<br>|SHAP|: %{z:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=residual_norm,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="Residual", len=0.45, y=0.23),
            hovertemplate="Freq: %{y}  Time: %{x}<br>Residual: %{z:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Single Heatmap View — SHAP & Residual",
        height=560,
        margin=dict(l=10, r=60, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SPEC LOADER 
# ─────────────────────────────────────────────────────────────────────────────

def load_spec(file_path: Union[str, Path]) -> torch.Tensor:
    
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:       # (H, W, 3) → (3, H, W)
            arr = arr.transpose(2, 0, 1)
        spec = torch.from_numpy(arr)

    elif suffix == ".npz":
        data = np.load(path)
        key  = list(data.keys())[0]
        arr  = data[key].astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)
        spec = torch.from_numpy(arr)

    elif suffix in {".pt", ".pth"}:
        spec = torch.load(path, map_location="cpu", weights_only=True)
        if spec.dtype != torch.float32:
            spec = spec.float()
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. Use .npy, .npz, .pt, or .pth"
        )

    # Strip batch dimension if present
    if spec.dim() == 4:
        spec = spec.squeeze(0)

    if spec.dim() != 3 or spec.shape[0] != 3:
        raise ValueError(
            f"Expected tensor of shape (3, H, W), got {tuple(spec.shape)}"
        )

    return spec


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_xai(
    file_path:    Union[str, Path],
    machine_type: str,
    machine_id:   str,
    *,
    # supply EITHER a pre-loaded registry OR individual model paths
    registry:           Optional[ProjectBRegistry] = None,
    model_path:         Optional[Union[str, Path]]  = None,
    stats_path:         Optional[Union[str, Path]]  = None,
    threshold_path:     Optional[Union[str, Path]]  = None,
    background_specs:   Optional[torch.Tensor]      = None,
    n_background:       int  = 50,
    shap_nsamples:      int  = 50,
    device:             str  = "cpu",
    normalise:          bool = True,
) -> XAIOutputBundle:
   
    # ── 1. Load spectrogram ───────────────────────────────────────────────
    spec = load_spec(file_path)

    # ── 2. Get XAI result ─────────────────────────────────────────────────
    if registry is not None:
        result: XAIResult = registry.explain(machine_type, machine_id, spec,
                                             normalise=normalise)
    elif model_path and stats_path and threshold_path:
        xai = ProjectBXAI(
            model_path       = model_path,
            stats_path       = stats_path,
            threshold_path   = threshold_path,
            background_specs = background_specs,
            n_background     = n_background,
            shap_nsamples    = shap_nsamples,
            device           = device,
        )
        result = xai.explain(spec, normalise=normalise)
    else:
        raise ValueError(
            "Provide either `registry` or all three of "
            "`model_path`, `stats_path`, `threshold_path`."
        )

    # ── 3. Build individual outputs ───────────────────────────────────────

    # 3a. Alert banner
    alert_banner = _classify_severity(result.anomaly_score, result.threshold)

    # 3b. Score readout + gauge
    gauge_fig    = _build_gauge(result.anomaly_score, result.threshold, result.is_anomaly)
    score_readout = ScoreReadout(
        anomaly_score = result.anomaly_score,
        threshold     = result.threshold,
        is_anomaly    = result.is_anomaly,
        gauge_fig     = gauge_fig,
    )

    # 3c. 3-D SHAP figure
    shap_3d_fig = _build_shap_3d(result.shap_map)

    # 3d. channel verdict (already a dict from xai_layer)
    channel_verdict = result.channel_verdict

    # 3e. 3-channel overlay
    overlay_fig = _build_overlay(result.input_spec, result.residual_map)

    # 3f. single merged heatmap
    heatmap_fig = _build_heatmap(result.shap_map, result.residual_2d)

    return XAIOutputBundle(
        result          = result,
        shap_3d_fig     = shap_3d_fig,
        channel_verdict = channel_verdict,
        score_readout   = score_readout,
        alert_banner    = alert_banner,
        overlay_fig     = overlay_fig,
        heatmap_fig     = heatmap_fig,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: individual render functions (call independently if needed)
# ─────────────────────────────────────────────────────────────────────────────

def render_shap_3d(result: XAIResult) -> Optional[object]:
    """Return the 3-D SHAP Plotly figure from an XAIResult."""
    return _build_shap_3d(result.shap_map)

def render_gauge(result: XAIResult) -> Optional[object]:
    """Return the score gauge Plotly figure from an XAIResult."""
    return _build_gauge(result.anomaly_score, result.threshold, result.is_anomaly)

def render_alert(result: XAIResult) -> AlertBanner:
    """Return the AlertBanner from an XAIResult."""
    return _classify_severity(result.anomaly_score, result.threshold)

def render_overlay(result: XAIResult) -> Optional[object]:
    """Return the 3-channel overlay Plotly figure from an XAIResult."""
    return _build_overlay(result.input_spec, result.residual_map)

def render_heatmap(result: XAIResult) -> Optional[object]:
    """Return the single merged heatmap Plotly figure from an XAIResult."""
    return _build_heatmap(result.shap_map, result.residual_2d)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (python xai_outputs.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    BASE   = Path(r"C:\Users\risha\Downloads\listen\listen\edge_deployments")
    FOLDER = BASE / "valve_id_06"

    H, W = 128, 128
    torch.manual_seed(0)

    # Save a dummy .npy test file
    dummy = np.random.rand(3, H, W).astype(np.float32)
    test_npy = Path("_test_spec.npy")
    np.save(test_npy, dummy)

    print("Running run_xai() self-test …")
    bundle = run_xai(
        file_path    = test_npy,
        machine_type = "valve",
        machine_id   = "id_06",
        model_path     = FOLDER / "cnn_ae_best.pth",
        stats_path     = FOLDER / "global_stats.json",
        threshold_path = FOLDER / "threshold_B.txt",
        background_specs = torch.rand(10, 3, H, W),
        n_background     = 5,
        shap_nsamples    = 10,
    )

    print("\n── Score Readout ─────────────────────────────────")
    print(f"  Anomaly score : {bundle.score_readout.anomaly_score:.6f}")
    print(f"  Threshold     : {bundle.score_readout.threshold:.6f}")
    print(f"  Is anomaly    : {bundle.score_readout.is_anomaly}")

    print("\n── Alert Banner ──────────────────────────────────")
    print(f"  Severity : {bundle.alert_banner.severity}")
    print(f"  Label    : {bundle.alert_banner.label}")
    print(f"  HTML     : {bundle.alert_banner.to_html()[:80]}…")

    print("\n── Channel Verdict ───────────────────────────────")
    for ch, pct in bundle.channel_verdict.items():
        print(f"  {ch[:40]:40s}  {pct:5.1f}%")

    print("\n── Plotly Figures ────────────────────────────────")
    for name, fig in [
        ("shap_3d_fig", bundle.shap_3d_fig),
        ("overlay_fig", bundle.overlay_fig),
        ("heatmap_fig", bundle.heatmap_fig),
        ("gauge_fig",   bundle.score_readout.gauge_fig),
    ]:
        status = f"{len(fig.data)} trace(s)" if fig else "None (plotly missing)"
        print(f"  {name:20s} → {status}")

    test_npy.unlink(missing_ok=True)
    print("\n✓  xai_outputs self-test complete.")