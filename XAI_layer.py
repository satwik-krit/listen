"""
xai_layer.py  —  Project B XAI Layer
======================================
Sits on top of trained CAE models and produces three explainability
outputs for any incoming spectrogram tensor:

    1. anomaly_score   – scalar MSE vs threshold
    2. residual_map    – (3, H, W) per-channel reconstruction error
    3. shap_map        – (3, H, W) GradientSHAP attribution map

All outputs are plain numpy arrays — the UI layer decides how to render them.

Architecture recovered from weights
-------------------------------------
    Encoder : Conv2d  3 → 32 → 64 → 128 → 256 → 256   (5 × MaxPool2d)
    Decoder : mirror with MaxUnpool2d + Conv2d
    Input   : (batch, 3, H, W)  — 3-channel Mel / Delta / Delta-Delta

NOTE on SHAP backend
---------------------
DeepExplainer is incompatible with MaxUnpool2d (gradient-hook / inplace
LeakyReLU conflicts). We use shap.GradientExplainer, which works via
SmoothGrad-style gradient sampling and is equally valid for attribution.
Output shapes are identical — the UI sees the same (3, H, W) array either way.

Usage
------
    from xai_layer import ProjectBXAI

    xai = ProjectBXAI(
        model_path       = "cnn_ae_best.pth",
        stats_path       = "global_stats.json",
        threshold_path   = "threshold_B.txt",
        background_specs = normal_tensor,      # (N, 3, H, W) normalised
        device           = "cuda",
    )

    result = xai.explain(spec_tensor)          # (3, H, W)

    result.anomaly_score    -> float
    result.is_anomaly        -> bool
    result.residual_map      -> np.ndarray (3, H, W)
    result.residual_2d       -> np.ndarray (H, W)      mean across channels
    result.shap_map          -> np.ndarray (3, H, W)
    result.shap_per_channel  -> np.ndarray (3,)         mean |SHAP| per channel
    result.channel_verdict   -> dict  channel_name -> contribution %
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1.  MODEL  (reconstructed from saved weight shapes)
# ─────────────────────────────────────────────────────────────

class CAEModel(nn.Module):
    """
    Convolutional Autoencoder matching the .pth weight layout.

    Key detail: LeakyReLU uses inplace=False throughout so that
    GradientSHAP's backward hooks can differentiate without errors.
    """

    def __init__(self):
        super().__init__()

        def enc_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=False),
            )

        def dec_block(ic, oc, last=False):
            if last:
                return nn.Sequential(
                    nn.Conv2d(ic, oc, 3, padding=1),
                )
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=False),
            )

        self.encoder = nn.ModuleDict({
            "enc1": enc_block(3,   32),
            "enc2": enc_block(32,  64),
            "enc3": enc_block(64,  128),
            "enc4": enc_block(128, 256),
            "enc5": enc_block(256, 256),
        })
        self.decoder = nn.ModuleDict({
            "dec1": dec_block(256, 256),
            "dec2": dec_block(256, 128),
            "dec3": dec_block(128, 64),
            "dec4": dec_block(64,  32),
            "dec5": dec_block(32,  3, last=True),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes, indices = [], []

        for name in ["enc1", "enc2", "enc3", "enc4", "enc5"]:
            sizes.append(x.shape)
            x = self.encoder[name](x)
            x, idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            indices.append(idx)

        for i, name in enumerate(["dec1", "dec2", "dec3", "dec4", "dec5"]):
            x = F.max_unpool2d(x, indices[-(i + 1)],
                               kernel_size=2, stride=2,
                               output_size=sizes[-(i + 1)])
            x = self.decoder[name](x)

        return x


def load_cae(model_path: str | Path, device: str = "cpu") -> CAEModel:
    """Load CAEModel from a .pth state-dict, remapping decoder key indices."""
    state = torch.load(model_path, map_location=device, weights_only=True)
    model = CAEModel().to(device)

    # Saved decoder keys: decoder.decN.1.weight, decoder.decN.2.weight
    # Our dec_block:       [0]=Conv, [1]=BN, [2]=LReLU
    # → shift saved index by -1
    remapped = {}
    for k, v in state.items():
        if k.startswith("decoder."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                parts[2] = str(int(parts[2]) - 1)
                k = ".".join(parts)
        remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[load_cae] Missing    : {missing}")
    if unexpected:
        print(f"[load_cae] Unexpected : {unexpected}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# 2.  MSE WRAPPER  (scalar target for GradientSHAP)
# ─────────────────────────────────────────────────────────────

class _MSEWrapper(nn.Module):
    """Wraps CAE so output is per-sample MSE as (B, 1). Required by GradientExplainer."""
    def __init__(self, cae: CAEModel):
        super().__init__()
        self.cae = cae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.cae(x)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)  # (B, 1)


# ─────────────────────────────────────────────────────────────
# 3.  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class XAIResult:
    # Detection
    anomaly_score:    float
    threshold:        float
    is_anomaly:       bool

    # Spectrogram tensors (numpy float32)
    input_spec:       np.ndarray          # (3, H, W)
    reconstructed:    np.ndarray          # (3, H, W)

    # Residual map
    residual_map:     np.ndarray          # (3, H, W)  per-channel squared error
    residual_2d:      np.ndarray          # (H, W)     mean across channels

    # SHAP attribution
    shap_map:         np.ndarray          # (3, H, W)  raw SHAP values
    shap_per_channel: np.ndarray          # (3,)       mean |SHAP| per channel
    channel_verdict:  dict = field(default_factory=dict)
    # e.g. {"Mel (static frequency anomaly)": 42.1,
    #        "Delta (rate of change)": 35.7,
    #        "Delta-Delta (acceleration)": 22.2}

    def summary(self) -> str:
        lines = [
            f"Anomaly Score : {self.anomaly_score:.6f}  "
            f"(threshold {self.threshold:.6f})",
            f"Decision      : "
            f"{'ANOMALY DETECTED' if self.is_anomaly else 'NORMAL'}",
            "",
            "Channel SHAP contribution:",
        ]
        for ch, pct in self.channel_verdict.items():
            bar = "=" * int(pct / 5)
            lines.append(f"  {ch:42s}  {pct:5.1f}%  [{bar}]")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 4.  MAIN XAI CLASS
# ─────────────────────────────────────────────────────────────

class ProjectBXAI:
    """
    XAI layer for a single machine-type / machine-id model.

    Parameters
    ----------
    model_path        : path to cnn_ae_best.pth
    stats_path        : path to global_stats.json
    threshold_path    : path to threshold_B.txt
    background_specs  : (N, 3, H, W) torch.Tensor of NORMAL spectrograms
                        already channel-normalised. Pass None to skip SHAP.
    n_background      : how many background samples to use for SHAP
    shap_nsamples     : gradient samples per SHAP call (lower = faster)
    device            : "cuda" | "cpu"
    """

    CHANNEL_NAMES = [
        "Mel (static frequency anomaly)",
        "Delta (rate of change)",
        "Delta-Delta (acceleration)",
    ]

    def __init__(
        self,
        model_path:        str | Path,
        stats_path:        str | Path,
        threshold_path:    str | Path,
        background_specs:  Optional[torch.Tensor] = None,
        n_background:      int = 50,
        shap_nsamples:     int = 50,
        device:            str = "cpu",
    ):
        self.device        = torch.device(device)
        self.shap_nsamples = shap_nsamples

        self.model = load_cae(model_path, device)

        with open(stats_path) as f:
            s = json.load(f)
        self._ch_mean = torch.tensor(s["ch_mean"], dtype=torch.float32).view(3, 1, 1)
        self._ch_std  = torch.tensor(s["ch_std"],  dtype=torch.float32).view(3, 1, 1)

        self.threshold = float(Path(threshold_path).read_text().strip())

        self._explainer = None
        if background_specs is not None:
            self._setup_shap(background_specs, n_background)

    # ── public api ──────────────────────────────────────────

    def normalise(self, spec: torch.Tensor) -> torch.Tensor:
        """(3, H, W) -> channel-normalised (3, H, W)."""
        return (spec - self._ch_mean) / (self._ch_std + 1e-8)

    def explain(
        self,
        spec:      torch.Tensor,
        normalise: bool = True,
    ) -> XAIResult:
        """
        Run full XAI pipeline on one spectrogram.

        Parameters
        ----------
        spec      : (3, H, W) or (1, 3, H, W) torch.Tensor
        normalise : apply channel normalisation before inference

        Returns
        -------
        XAIResult with all arrays as numpy float32, ready for the UI
        """
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)
        if normalise:
            spec = self.normalise(spec.squeeze(0)).unsqueeze(0)
        spec = spec.to(self.device)

        with torch.no_grad():
            recon = self.model(spec)

        sq_err        = (spec - recon) ** 2
        anomaly_score = sq_err.mean().item()
        residual_map  = sq_err.squeeze(0).cpu().numpy()   # (3, H, W)
        residual_2d   = residual_map.mean(axis=0)         # (H, W)

        shap_map, shap_per_ch, verdict = self._run_shap(spec)

        return XAIResult(
            anomaly_score    = anomaly_score,
            threshold        = self.threshold,
            is_anomaly       = anomaly_score > self.threshold,
            input_spec       = spec.squeeze(0).cpu().numpy(),
            reconstructed    = recon.squeeze(0).cpu().detach().numpy(),
            residual_map     = residual_map,
            residual_2d      = residual_2d,
            shap_map         = shap_map,
            shap_per_channel = shap_per_ch,
            channel_verdict  = verdict,
        )

    # ── internal ────────────────────────────────────────────

    def _setup_shap(self, background: torch.Tensor, n: int):
        try:
            import shap as _shap
        except ImportError:
            warnings.warn("[XAI] shap not installed — SHAP maps disabled.")
            return

        idx = torch.randperm(len(background))[:n]
        bg  = background[idx].to(self.device)

        wrapper = _MSEWrapper(self.model).to(self.device)
        wrapper.eval()
        self._explainer    = _shap.GradientExplainer(wrapper, bg)
        self._shap_wrapper = wrapper

    def _run_shap(self, spec_gpu: torch.Tensor):
        H, W  = spec_gpu.shape[2], spec_gpu.shape[3]
        zeros = np.zeros((3, H, W), dtype=np.float32)
        zero_v = {n: 0.0 for n in self.CHANNEL_NAMES}

        if self._explainer is None:
            return zeros, np.zeros(3, dtype=np.float32), zero_v

        try:
            sv  = self._explainer.shap_values(spec_gpu, nsamples=self.shap_nsamples)
            arr = np.array(sv).squeeze()    # -> (3, H, W)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
        except Exception as e:
            warnings.warn(f"[XAI] SHAP computation failed: {e}")
            return zeros, np.zeros(3, dtype=np.float32), zero_v

        per_ch = np.abs(arr).mean(axis=(1, 2))   # (3,)
        total  = per_ch.sum() + 1e-12
        pcts   = (per_ch / total) * 100

        verdict = {
            name: round(float(pcts[i]), 2)
            for i, name in enumerate(self.CHANNEL_NAMES)
        }

        return arr.astype(np.float32), per_ch.astype(np.float32), verdict


# ─────────────────────────────────────────────────────────────
# 5.  REGISTRY  (all 16 machine-type/id combos)
# ─────────────────────────────────────────────────────────────

# ... [Keep Architecture and Result Dataclass the same as your file] ...

class ProjectBRegistry:
    """
    Loads all (machine_type, machine_id) XAI instances from your 
    edge_deployments directory.
    """
    # Updated to match the types found in your directory
    MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
    # Updated to include all possible IDs (00, 02, 04, 06)
    MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

    def __init__(
        self,
        root_dir:         str | Path,
        background_specs: Optional[dict[str, torch.Tensor]] = None,
        n_background:     int = 50,
        shap_nsamples:    int = 50,
        device:           str = "cpu",
    ):
        self.device  = device
        self._models: dict[tuple[str, str], ProjectBXAI] = {}
        root = Path(root_dir)

        print(f"[Registry] Scanning {root} for models...")

        for mtype in self.MACHINE_TYPES:
            for mid in self.MACHINE_IDS:
                # The path logic now matches: edge_deployments/valve_id_06
                folder_name = f"{mtype}_{mid}"
                d = root / folder_name 
                
                model_file = d / "cnn_ae_best.pth"
                
                if not model_file.exists():
                    # Optional: print(f"  - Skipping {folder_name}: No model found.")
                    continue
                
                print(f"  + Loading {folder_name}...")
                
                bg = background_specs.get(mtype) if background_specs else None
                
                self._models[(mtype, mid)] = ProjectBXAI(
                    model_path       = model_file,
                    stats_path       = d / "global_stats.json",
                    threshold_path   = d / "threshold_B.txt",
                    background_specs = bg,
                    n_background     = n_background,
                    shap_nsamples    = shap_nsamples,
                    device           = device,
                )

        print(f"[Registry] Successfully loaded {len(self._models)} models.")

    def explain(self, machine_type: str, machine_id: str, spec: torch.Tensor, normalise: bool = True) -> XAIResult:
        key = (machine_type, machine_id)
        if key not in self._models:
            raise KeyError(f"No model loaded for {machine_type}_{machine_id}. Found: {self.available()}")
        return self._models[key].explain(spec, normalise=normalise)

    def available(self) -> list[str]:
        return [f"{t}_{i}" for t, i in self._models.keys()]


# ─────────────────────────────────────────────────────────────
# 6.  UPDATED SELF-TEST (Uses your specific paths)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Define your base path
    BASE_PATH = Path(r"C:\Users\risha\Downloads\listen\listen\edge_deployments")
    # 2. Pick a specific subfolder for a quick test
    TEST_FOLDER = BASE_PATH / "valve_id_06"

    print("=" * 65)
    print("Project B XAI Layer — Path Corrected Self-Test")
    print("=" * 65)

    if not TEST_FOLDER.exists():
        print(f"ERROR: Could not find folder {TEST_FOLDER}")
        print("Please check your C:\\ drive path.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create dummy data for testing
        H, W = 128, 128
        torch.manual_seed(42)
        bg_normal = torch.rand(10, 3, H, W)
        test_spec = torch.rand(3, H, W)

        try:
            xai = ProjectBXAI(
                model_path       = TEST_FOLDER / "cnn_ae_best.pth",
                stats_path       = TEST_FOLDER / "global_stats.json",
                threshold_path   = TEST_FOLDER / "threshold_B.txt",
                background_specs = bg_normal,
                n_background     = 5,
                shap_nsamples    = 10,
                device           = device,
            )

            result = xai.explain(test_spec, normalise=True)
            print(result.summary())
            print("\n✓ XAI logic and paths are now working correctly.")

        except FileNotFoundError as e:
            print(f"Path Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")