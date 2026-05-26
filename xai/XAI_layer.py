"""
xai_layer.py  —  Project B XAI Layer
   
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

class Decoder(nn.Module):
    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=False), # False for GradientSHAP
        )
    def __init__(self) -> None:
        super().__init__()
        self.dec1 = self._up_block(256, 256) 
        self.dec2 = self._up_block(256, 128) 
        self.dec3 = self._up_block(128,  64) 
        self.dec4 = self._up_block( 64,  32) 
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(), 
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec5(self.dec4(self.dec3(self.dec2(self.dec1(z)))))

class CAEModel(nn.Module):
    """
    Convolutional Autoencoder matching the .pth weight layout.

    Key detail: LeakyReLU uses inplace=False throughout so that
    GradientSHAP's backward hooks can differentiate without errors.
    """

    def __init__(self):
        super().__init__()
        # Parallel encoders with Group Normalization (8 groups)
        self.enc_low = nn.Sequential(
            self._block(3, 32, stride=2),   # 64 -> 32
            self._block(32, 64, stride=2),  # 32 -> 16
            self._block(64, 128, stride=2), # 16 -> 8
            self._block(128, 256, stride=2),# 8 -> 4
            self._block(256, 256, stride=2),# 4 -> 2
        )
        self.enc_mid = nn.Sequential(
            self._block(3, 32, stride=2),   # 52 -> 26
            self._block(32, 64, stride=2),  # 26 -> 13
            self._block(64, 128, stride=2), # 13 -> 7
            self._block(128, 256, stride=2),# 7 -> 4
            self._block(256, 256, stride=2),# 4 -> 2
        )
        self.enc_high = nn.Sequential(
            self._block(3, 32, stride=2),   # 12 -> 6
            self._block(32, 64, stride=2),  # 6 -> 3
            self._block(64, 128, stride=2), # 3 -> 2
            self._block(128, 256, stride=1),# 2 -> 2
            self._block(256, 256, stride=1),# 2 -> 2
        )
        
        self.bottleneck_linear = nn.Linear(6144, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.1)
        
        self.decoder_projection = nn.Linear(256, 4096)
        self.decoder = Decoder()

    def _block(self, in_ch, out_ch, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def encode(self, x):
        import librosa
        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=11025)
        idx_2k = int(np.argmin(np.abs(mel_freqs - 2000)))
        idx_8k = int(np.argmin(np.abs(mel_freqs - 8000)))
        
        # Row-wise Mel slicing
        low_img = x[:, :, :idx_2k, :]
        mid_img = x[:, :, idx_2k:idx_8k, :]
        high_img = x[:, :, idx_8k:, :]
        
        low_enc = self.enc_low(low_img)
        mid_enc = self.enc_mid(mid_img)
        high_enc = self.enc_high(high_img)
        
        assert low_enc.shape[2] > 0 and low_enc.shape[3] > 0, f"Low-freq encoder contracted: {low_enc.shape}"
        assert mid_enc.shape[2] > 0 and mid_enc.shape[3] > 0, f"Mid-freq encoder contracted: {mid_enc.shape}"
        assert high_enc.shape[2] > 0 and high_enc.shape[3] > 0, f"High-freq encoder contracted: {high_enc.shape}"
        
        # Flatten and concatenate parallel maps
        low_flat = low_enc.reshape(low_enc.size(0), -1)
        mid_flat = mid_enc.reshape(mid_enc.size(0), -1)
        high_flat = high_enc.reshape(high_enc.size(0), -1)
        
        fused = torch.cat([low_flat, mid_flat, high_flat], dim=1)
        
        compressed = self.bottleneck_linear(fused)
        compressed = self.layer_norm(compressed)
        compressed = self.dropout(compressed)
        return compressed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        dec_proj = self.decoder_projection(z)
        dec_in = dec_proj.reshape(dec_proj.size(0), 256, 4, 4)
        return self.decoder(dec_in)

def load_cae(model_path: str | Path, device: str = "cpu") -> CAEModel:
    """Load CAEModel from a .pth state-dict."""
    state = torch.load(model_path, map_location=device, weights_only=True)
    model = CAEModel().to(device)
    model.load_state_dict(state, strict=True)
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