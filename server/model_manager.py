"""
server/model_manager.py  — L.I.S.T.E.N. Master Pipeline
Handles: Feature extraction → Classification → Autoencoder inference → XAI
Project A (Edge): 8 audio features  → ONNX autoencoder → PFERD XAI
Project B (GPU):  3-ch Mel tensor   → PyTorch CAE      → GradientSHAP XAI
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

warnings.filterwarnings("ignore")

# Make sure the project root is on sys.path so all sibling modules resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

# ─────────────────────────────────────────────────────────────────────────────
# CAE model (mirrors xai/XAI_layer.py — duplicated here so server/ is
# self-contained and can be run without the xai/ package on the path)
# ─────────────────────────────────────────────────────────────────────────────


class _CAEModel(nn.Module):
    def __init__(self):
        super().__init__()

        def _enc(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=False),
            )

        def _dec(ic, oc, last=False):
            if last:
                return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1))
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=False),
            )

        self.encoder = nn.ModuleDict(
            {
                "enc1": _enc(3, 32),
                "enc2": _enc(32, 64),
                "enc3": _enc(64, 128),
                "enc4": _enc(128, 256),
                "enc5": _enc(256, 256),
            }
        )
        self.decoder = nn.ModuleDict(
            {
                "dec1": _dec(256, 256),
                "dec2": _dec(256, 128),
                "dec3": _dec(128, 64),
                "dec4": _dec(64, 32),
                "dec5": _dec(32, 3, last=True),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes, indices = [], []
        for name in ["enc1", "enc2", "enc3", "enc4", "enc5"]:
            sizes.append(x.shape)
            x = self.encoder[name](x)
            x, idx = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(idx)
        for i, name in enumerate(["dec1", "dec2", "dec3", "dec4", "dec5"]):
            x = F.max_unpool2d(x, indices[-(i + 1)], 2, 2, output_size=sizes[-(i + 1)])
            x = self.decoder[name](x)
        return x


class _MSEWrapper(nn.Module):
    """Thin wrapper so GradientExplainer has a scalar (MSE) output."""

    def __init__(self, cae: _CAEModel):
        super().__init__()
        self.cae = cae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.cae(x)) ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)


def _load_cae(path: Path) -> _CAEModel:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = _CAEModel()
    # Remap decoder indices (saved model has 1-indexed layers in some versions)
    remapped = {}
    for k, v in state.items():
        if k.startswith("decoder."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                parts[2] = str(int(parts[2]) - 1)
                k = ".".join(parts)
        remapped[k] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EDGE XAI — PFERD (Per-Feature Error Reconstruction Dictionary)
# ─────────────────────────────────────────────────────────────────────────────

_EDGE_FEATURE_NAMES = [
    "ZCR",
    "RMS Mean",
    "RMS Variance",
    "Median Pitch",
    "Spectral Centroid",
    "Rolloff",
    "MFCC-1",
    "MFCC-2",
]

_EDGE_FAULT_DICT = {
    "ZCR": "Bearing friction or surface roughness detected",
    "RMS Mean": "Amplitude surge — mechanical looseness or impact",
    "RMS Variance": "Unstable vibration — intermittent fault developing",
    "Median Pitch": "Shaft speed deviation — possible rotational imbalance",
    "Spectral Centroid": "Frequency content shifted — wear toward higher bands",
    "Rolloff": "High-frequency spike — metal contact or cracking",
    "MFCC-1": "Tonal character changed — resonance frequency shifted",
    "MFCC-2": "Spectral shape distorted — multiple fault signatures",
}

_GPU_FAULT_DICT = {
    "Mel (static)": "static frequency anomaly — likely bearing wear or resonance shift",
    "Delta (velocity)": "rate-of-change anomaly — transient fault or mechanical impact",
    "Delta-Delta (accel)": "acceleration anomaly — rapid onset fault or cavitation",
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MANAGER
# ─────────────────────────────────────────────────────────────────────────────


class ModelManager:

    def __init__(self):
        print("[MODEL MANAGER] Initialised — lazy-loading active.")
        self._edge: dict[str, dict] = {}  # key → {session, scaler, threshold}
        self._gpu: dict[str, dict] = {}  # key → {model, ch_mean, ch_std, threshold}
        self._noise: dict[str, Optional[np.ndarray]] = {}

    # ── key / directory helpers ───────────────────────────────────────────

    @staticmethod
    def _key(machine_type: str, machine_id: str) -> str:
        return f"{machine_type}{machine_id}"

    @staticmethod
    def _edge_dir(machine_type: str, machine_id: str) -> Path:
        return config.MODEL_DIR_EDGE / f"{machine_type}_id_{machine_id}"

    @staticmethod
    def _gpu_dir(machine_type: str, machine_id: str) -> Path:
        return config.MODEL_DIR_GPU / f"{machine_type}_id_{machine_id}"

    # ── lazy loaders ──────────────────────────────────────────────────────

    def _ensure_edge(self, machine_type: str, machine_id: str):
        key = self._key(machine_type, machine_id)
        if key in self._edge:
            return
        import onnxruntime as ort

        d = self._edge_dir(machine_type, machine_id)
        print(f"[LAZY] Edge model: {key} from {d}")
        session = ort.InferenceSession(
            str(d / "edge_ae.onnx"), providers=["CPUExecutionProvider"]
        )
        scaler = joblib.load(d / "scaler.pkl")
        # threshold = float((d / "threshold.txt").read_text().strip())
        # Use utf-8-sig to safely consume any hidden BOM characters, and strip accidental quotes
        # Read file, split at the first '#' to remove comments, then strip whitespace/quotes
        raw_text = (d / "threshold.txt").read_text(encoding="utf-8-sig")
        clean_thresh = raw_text.split("#")[0].strip().strip("'\"")
        threshold = float(clean_thresh)
        # raw_thresh = (
        #     (d / "threshold.txt").read_text(encoding="utf-8-sig").strip().strip("'\"")
        # )
        # threshold = float(raw_thresh)
        self._edge[key] = {"session": session, "scaler": scaler, "threshold": threshold}
        print(f"[✓] Edge {key.upper()} ready  (thr={threshold:.5f})")

    def _ensure_gpu(self, machine_type: str, machine_id: str):
        key = self._key(machine_type, machine_id)
        if key in self._gpu:
            return
        d = self._gpu_dir(machine_type, machine_id)
        print(f"[LAZY] GPU model: {key} from {d}")
        model = _load_cae(d / "cnn_ae_best.pth")
        with open(d / "global_stats.json") as f:
            stats = json.load(f)
        # threshold = float((d / "threshold_B.txt").read_text().strip())
        # Sanitize the GPU threshold file as well
        # Safely parse GPU threshold by ignoring comments
        raw_text_B = (d / "threshold_B.txt").read_text(encoding="utf-8-sig")
        clean_thresh_B = raw_text_B.split("#")[0].strip().strip("'\"")
        threshold = float(clean_thresh_B)
        # raw_thresh_gpu = (
        #     (d / "threshold_B.txt").read_text(encoding="utf-8-sig").strip().strip("'\"")
        # )
        # threshold = float(raw_thresh_gpu)
        ch_mean = torch.tensor(stats["ch_mean"], dtype=torch.float32).view(3, 1, 1)
        ch_std = torch.tensor(stats["ch_std"], dtype=torch.float32).view(3, 1, 1)
        self._gpu[key] = {
            "model": model,
            "ch_mean": ch_mean,
            "ch_std": ch_std,
            "threshold": threshold,
        }
        print(f"[✓] GPU  {key.upper()} ready  (thr={threshold:.5f})")

    def _get_noise(self, machine_type: str, machine_id: str) -> Optional[np.ndarray]:
        key = self._key(machine_type, machine_id)
        if key in self._noise:
            return self._noise[key]
        # Search for a pre-computed master noise profile
        candidates = [
            _ROOT
            / "scalers"
            / f"6_dB_{machine_type}"
            / f"id_{machine_id}"
            / "master_noise.pkl",
            _ROOT / "scalers" / f"6_dB_valve" / "id_00" / "master_noise.pkl",
        ]
        for p in candidates:
            if p.exists():
                self._noise[key] = joblib.load(p)
                print(f"[NOISE] Loaded master noise for {key} from {p.parent}")
                return self._noise[key]
        self._noise[key] = None
        return None

    # ── audio loading + feature extraction ───────────────────────────────

    def _load_audio(
        self, file_path: str, machine_type: str = "valve", machine_id: str = "00"
    ) -> tuple:
        """Load WAV → noise-reduce → return (y, sr)."""
        import librosa

        y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)
        noise = self._get_noise(machine_type, machine_id)
        if noise is not None:
            try:
                import noisereduce

                y = noisereduce.reduce_noise(
                    y=y, sr=sr, y_noise=noise, prop_decrease=1.0
                )
            except Exception as e:
                print(f"[WARN] Noise reduce failed: {e}")
        return y, sr

    def _audio_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """8 statistical audio features for classification and Edge pipeline."""
        from feature_extraction.features import extract_audio_features

        return extract_audio_features(y, sr)

    def _mel_tensor(self, y: np.ndarray, sr: int) -> torch.Tensor:
        """3-channel (Mel, Δ, ΔΔ) tensor for GPU pipeline, shape (3, N_MELS, W)."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-sample min-max → [0, 1]
        lo, hi = log_mel.min(), log_mel.max()
        log_mel = (log_mel - lo) / (hi - lo + 1e-8)

        # Pad / trim to fixed width
        W = config.FIXED_WIDTH
        if log_mel.shape[1] >= W:
            log_mel = log_mel[:, :W]
        else:
            log_mel = np.pad(log_mel, ((0, 0), (0, W - log_mel.shape[1])))

        delta = librosa.feature.delta(log_mel)
        delta2 = librosa.feature.delta(log_mel, order=2)

        spec = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)
        return torch.from_numpy(spec)

    # ── classification ────────────────────────────────────────────────────

    def _classify(self, audio_features: np.ndarray) -> tuple[str, str]:
        """Return (machine_type, machine_id) e.g. ('valve', '04')."""
        from classify.classifier import predict

        # Reshape the 1D array to a 2D array (1 sample, 8 features)
        result = predict(audio_features.reshape(1, -1))

        # Note: If predict() returns a NumPy array rather than a plain string,
        # you may also need to extract the first element like this:
        if isinstance(result, np.ndarray):
            result = result[0]

        machine_type, machine_id = result.split(" ")
        return machine_type, machine_id.zfill(2)

    # ── Edge inference ────────────────────────────────────────────────────

    def _run_edge(
        self, audio_features: np.ndarray, machine_type: str, machine_id: str
    ) -> tuple:
        key = self._key(machine_type, machine_id)
        entry = self._edge[key]

        scaled = (
            entry["scaler"].transform(audio_features.reshape(1, -1)).astype(np.float32)
        )
        inp_name = entry["session"].get_inputs()[0].name
        recon = entry["session"].run(None, {inp_name: scaled})[0]

        mse = float(np.mean((scaled - recon) ** 2))
        return scaled[0], recon[0], mse, entry["threshold"]

    # ── GPU inference ─────────────────────────────────────────────────────

    def _run_gpu(self, spec: torch.Tensor, machine_type: str, machine_id: str) -> tuple:
        key = self._key(machine_type, machine_id)
        entry = self._gpu[key]

        normed = (spec - entry["ch_mean"]) / (entry["ch_std"] + 1e-8)
        x = normed.unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            recon = entry["model"](x)

        sq_err = (x - recon) ** 2
        mse = float(sq_err.mean().item())
        return x, recon, sq_err, mse, entry["threshold"]

    # ── Edge XAI (PFERD) ──────────────────────────────────────────────────

    def _xai_edge(self, inp: np.ndarray, rec: np.ndarray) -> tuple[str, dict]:
        errors = np.abs(inp - rec)
        ranked = sorted(
            zip(_EDGE_FEATURE_NAMES, errors, inp, rec), key=lambda t: t[1], reverse=True
        )

        top_name, top_err, top_actual, top_rec = ranked[0]
        explanation = (
            f"Primary fault indicator: {top_name} (error={top_err:.4f}, "
            f"actual={top_actual:.4f}, expected={top_rec:.4f}). "
            f"{_EDGE_FAULT_DICT.get(top_name, 'Abnormal signal pattern')}."
        )
        if len(ranked) > 1:
            sec = ", ".join(f"{r[0]} ({r[1]:.4f})" for r in ranked[1:3])
            explanation += f" Contributing features: {sec}."

        graphs = {
            "spectrogram": errors.tolist(),  # error per feature (bar heights)
            "audio_features": inp.tolist(),  # raw actual values
            "feature_names": _EDGE_FEATURE_NAMES,
            "reconstructed": rec.tolist(),
        }
        return explanation, graphs

    # ── GPU XAI (GradientSHAP + residual) ────────────────────────────────

    def _xai_gpu(
        self, x: torch.Tensor, sq_err: torch.Tensor, machine_type: str, machine_id: str
    ) -> tuple[str, dict]:

        key = self._key(machine_type, machine_id)
        model = self._gpu[key]["model"]

        residual_2d = sq_err.squeeze(0).mean(dim=0).cpu().numpy()  # (H, W)
        residual_per_frame = residual_2d.mean(axis=0).tolist()  # (W,)
        residual_per_freq = residual_2d.mean(axis=1).tolist()  # (H,)

        # --- GradientSHAP ---
        shap_pct = [33.3, 33.3, 33.4]  # fallback equal split
        try:
            import shap as _shap

            wrapper = _MSEWrapper(model)
            wrapper.eval()
            bg = torch.zeros(1, *x.shape[1:])
            explainer = _shap.GradientExplainer(wrapper, bg)
            sv = explainer.shap_values(x, nsamples=30)
            arr = np.array(sv).squeeze()  # (3, H, W)
            if arr.ndim == 3:
                per_ch = np.abs(arr).mean(axis=(1, 2))
                total = per_ch.sum() + 1e-12
                shap_pct = (per_ch / total * 100).tolist()
        except Exception as e:
            print(f"[XAI-GPU] GradientSHAP skipped ({e.__class__.__name__})")

        ch_names = ["Mel (static)", "Delta (velocity)", "Delta-Delta (accel)"]
        dominant = ch_names[int(np.argmax(shap_pct))]
        severity = (
            "HIGH" if float(np.array(residual_per_frame).max()) > 0.15 else "MODERATE"
        )

        explanation = (
            f"{severity} reconstruction error. "
            f"Primary channel: {dominant}. "
            f"Suggests {_GPU_FAULT_DICT.get(dominant, 'abnormal acoustic pattern')}."
        )

        graphs = {
            "spectrogram": [float(v) for v in residual_per_frame[:150]],
            "audio_features": [float(v) for v in shap_pct],
            "shap_channels": dict(zip(ch_names, [round(v, 1) for v in shap_pct])),
            "residual_freq": [float(v) for v in residual_per_freq],
            "residual_mean": float(residual_2d.mean()),
        }
        return explanation, graphs

    # ── MASTER PIPELINE (called by FastAPI worker) ────────────────────────

    def process_pipeline(
        self, file_path: str, project_id: str, reported_node_type: str
    ) -> dict:
        """
        Full pipeline: WAV → classify → branch (edge | gpu) → XAI → result dict.
        """
        try:
            # ── Step 1: Load audio (noise-reduced) ───────────────────────
            # Use a best-effort noise profile during the pre-classify phase
            y, sr = self._load_audio(file_path, reported_node_type, "00")

            # ── Step 2: Extract 8 audio features (lightweight, for classifier) ──
            audio_features = self._audio_features(y, sr)

            # ── Step 3: Classify machine type + ID ───────────────────────
            try:
                machine_type, machine_id = self._classify(audio_features)
            except Exception as ce:
                # Classifier failure → fall back to reported type, id_00
                print(f"[WARN] Classifier failed: {ce} — using reported type")
                machine_type = reported_node_type.lower()
                machine_id = "00"
            print(
                f"[PIPELINE] Classified → {machine_type} id_{machine_id}  [{project_id.upper()}]"
            )

            # ── Step 4: Reload audio with the now-known noise profile ─────
            y, sr = self._load_audio(file_path, machine_type, machine_id)
            audio_features = self._audio_features(
                y, sr
            )  # re-extract after noise reduce

            # ── Step 5: Branch by project_id ─────────────────────────────
            if project_id == "edge":
                return self._pipeline_edge(audio_features, machine_type, machine_id)
            else:
                spec = self._mel_tensor(y, sr)
                return self._pipeline_gpu(spec, machine_type, machine_id)

        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    # ── edge sub-pipeline ─────────────────────────────────────────────────

    def _pipeline_edge(
        self, audio_features: np.ndarray, machine_type: str, machine_id: str
    ) -> dict:
        try:
            self._ensure_edge(machine_type, machine_id)
        except Exception as e:
            return {
                "error": f"Edge model load failed for {machine_type}/{machine_id}: {e}"
            }

        inp, rec, mse, threshold = self._run_edge(
            audio_features, machine_type, machine_id
        )
        is_anomaly = bool(mse > threshold)

        result: dict = {
            "machine_type": machine_type,
            "machine_id": machine_id,
            "is_anomaly": is_anomaly,
            "mse_score": round(mse, 6),
            "threshold": round(threshold, 6),
            "explanation": None,
            "graphs": None,
        }

        if is_anomaly:
            explanation, graphs = self._xai_edge(inp, rec)
            result["explanation"] = explanation
            result["graphs"] = graphs

        return result

    # ── gpu sub-pipeline ──────────────────────────────────────────────────

    def _pipeline_gpu(
        self, spec: torch.Tensor, machine_type: str, machine_id: str
    ) -> dict:
        try:
            self._ensure_gpu(machine_type, machine_id)
        except Exception as e:
            return {
                "error": f"GPU model load failed for {machine_type}/{machine_id}: {e}"
            }

        x, recon, sq_err, mse, threshold = self._run_gpu(spec, machine_type, machine_id)
        is_anomaly = bool(mse > threshold)

        result: dict = {
            "machine_type": machine_type,
            "machine_id": machine_id,
            "is_anomaly": is_anomaly,
            "mse_score": round(mse, 6),
            "threshold": round(threshold, 6),
            "explanation": None,
            "graphs": None,
        }

        if is_anomaly:
            explanation, graphs = self._xai_gpu(x, sq_err, machine_type, machine_id)
            result["explanation"] = explanation
            result["graphs"] = graphs

        return result
