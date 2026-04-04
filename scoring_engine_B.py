import json
import os
import warnings
from pathlib import Path
from typing import Optional

import librosa
import matplotlib
matplotlib.use("Agg")          # MUST be before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARCHITECTURE  (must exactly match training)
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _conv_block(3,   32),
            _conv_block(32,  64),
            _conv_block(64,  128),
            _conv_block(128, 256),
            _conv_block(256, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _up_block(256, 256),
            _up_block(256, 128),
            _up_block(128,  64),
            _up_block( 64,  32),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class CNNAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCORER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# BLUEPRINT 2: Z-Score Normalization Keys
_REQUIRED_STAT_KEYS = {"ch_mean", "ch_std"}


class ProjectBScorer:
    """
    Loads a trained CNNAutoencoder from a deployment folder and scores
    individual .npy audio-feature files for anomalies using Pure MSE.

    Deployment folder must contain:
        cnn_ae_best.pth       — model weights
        global_stats.json     — per-channel Z-score stats (ch_mean, ch_std)
        threshold_B.txt       — decision threshold (first line)
        scorer_config.json    — (Optional) runtime config: img_size, is_power_spec
    """

    # ── construction ────────────────────────────────────────────────────────

    def __init__(
        self,
        deployment_folder: str | os.PathLike,
        device: Optional[str] = None, # Defaults to GPU if available
        output_dir: Optional[str | os.PathLike] = None,
    ) -> None:
        self._deployment = Path(deployment_folder)
        
        # BLUEPRINT 1: Hardware Activation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Heatmaps go here; defaults to <deployment>/heatmaps/
        self._output_dir = Path(output_dir) if output_dir else self._deployment / "heatmaps"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔧  Initialising scorer: {self._deployment.name}")

        # ── config (img_size, mel format) ───────────────────────────
        cfg_path = self._deployment / "scorer_config.json"
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = json.load(f)
            self.img_size      = int(cfg.get("img_size", 128))
            self.is_power_spec = bool(cfg.get("is_power_spec", True))
        else:
            warnings.warn(
                "scorer_config.json not found — falling back to hardcoded defaults "
                "(img_size=128, is_power_spec=True). ",
                stacklevel=2,
            )
            self.img_size      = 128
            self.is_power_spec = True

        # ── global stats (Z-Score) ─────────────────────────────────────────
        stats_path = self._deployment / "global_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"global_stats.json not found in {self._deployment}")
        with stats_path.open() as f:
            self._stats = json.load(f)
        
        # Validate keys
        missing_ch = _REQUIRED_STAT_KEYS - self._stats.keys()
        if missing_ch:
            raise KeyError(f"global_stats.json is missing keys: {missing_ch}")
            
        self.ch_mean = np.array(self._stats["ch_mean"], dtype=np.float32)
        self.ch_std  = np.array(self._stats["ch_std"], dtype=np.float32)

        # ── threshold ──────────────────────────────────────────────────────
        thresh_path = self._deployment / "threshold_B.txt"
        if not thresh_path.exists():
            raise FileNotFoundError(f"threshold_B.txt not found in {self._deployment}")
        with thresh_path.open() as f:
            line = f.readline().strip()
        try:
            self.threshold = float(line)
        except ValueError:
            raise ValueError(f"threshold_B.txt contains non-numeric value: '{line!r}'")

        # ── model ──────────────────────────────────────────────────────────
        model_path = self._deployment / "cnn_ae_best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"cnn_ae_best.pth not found in {self._deployment}")
        self._model = CNNAutoencoder().to(self.device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        # Lock model permanently in eval mode
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        print(
            f"✅  Ready | threshold={self.threshold:.6f} | "
            f"img={self.img_size}px | device={self.device}"
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _load_and_check(self, npy_path: str | os.PathLike) -> np.ndarray:
        path = Path(npy_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file, got: {path.suffix}")
        return np.load(path).astype(np.float32)

    # ── public API ──────────────────────────────────────────────────────────

    def preprocess(self, npy_path: str | os.PathLike) -> torch.Tensor:
        """
        Raw audio feature array → Z-score normalised 3-channel tensor (1, 3, H, W).
        """
        mel = self._load_and_check(npy_path)

        # ── shape normalisation ────────────────────────────────────────────
        mel = mel.squeeze()
        if mel.ndim == 1:
            mel = mel.reshape(1, -1)          # (1, T) — degenerate but safe
        if mel.ndim != 2:
            raise ValueError(
                f"Cannot reduce mel array of shape {mel.shape} to 2-D. "
                "Expected (n_mels, T) or squeeze-able to it."
            )

        # ── dB conversion ─────────────────────────────────────────────────
        db: np.ndarray = (
            librosa.power_to_db(mel, ref=np.max) if self.is_power_spec else mel
        )

        # ── delta features (safe on any 2-D array) ────────────────────────
        delta  = librosa.feature.delta(db)
        delta2 = librosa.feature.delta(db, order=2)

        # BLUEPRINT 2: Z-score Normalization (replaces min/max clipping)
        db_norm     = (db - self.ch_mean[0]) / self.ch_std[0]
        delta_norm  = (delta - self.ch_mean[1]) / self.ch_std[1]
        delta2_norm = (delta2 - self.ch_mean[2]) / self.ch_std[2]

        # ── stack & interpolate ─────────────────────────────────────────────
        stacked = np.stack([db_norm, delta_norm, delta2_norm], axis=0) # (3, H, W)

        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 3, H, W)
        return F.interpolate(
            tensor, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )

    def _save_heatmap(
        self,
        diff: np.ndarray,
        stem: str,
    ) -> Path:
        """
        Saves a pixel-wise reconstruction-error heatmap.
        Uses a fresh Figure object (thread-safe; no global pyplot state).
        """
        heatmap_path = self._output_dir / f"{stem}_heatmap.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        try:
            ax.imshow(diff, cmap="hot", aspect="auto")
            ax.axis("off")
            fig.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        finally:
            plt.close(fig)   # always released, even if savefig raises
        return heatmap_path

    @torch.no_grad()
    def score_sample(
        self,
        npy_path: str | os.PathLike,
        save_heatmap: bool = False,
    ) -> dict:
        """
        Score a single .npy sample using Hackathon Pure MSE.
        """
        tensor = self.preprocess(npy_path).to(self.device, non_blocking=True)
        
        # BLUEPRINT 3: Mixed Precision for 2x Inference Speed
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            recon = self._model(tensor)
            
            # MATCH HACKATHON TRAINING LOGIC: Pure MSE
            score = F.mse_loss(recon, tensor).item()

        heatmap_path: Optional[Path] = None
        if save_heatmap:
            # Explicitly cast to float32 before sending back to numpy/CPU
            diff = ((tensor[0, 0].float() - recon[0, 0].float()) ** 2).cpu().numpy()
            heatmap_path = self._save_heatmap(diff, stem=Path(npy_path).stem)

        return {
            "score":        score,
            "threshold":    self.threshold,
            "is_anomaly":   score > self.threshold,
            "heatmap_path": heatmap_path,
        }

    def score_batch(
        self,
        npy_paths: list[str | os.PathLike],
        save_heatmaps: bool = False,
    ) -> list[dict]:
        """
        Score multiple files one-at-a-time to keep memory bounded.
        """
        return [
            self.score_sample(p, save_heatmap=save_heatmaps)
            for p in npy_paths
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Replace these paths if testing locally
    deployment_dir = r"C:\Users\risha\Downloads\listen\listen\edge_deployments\fan_id_00"
    test_file      = r"C:\Users\risha\Downloads\listen\listen\split_output\test\model_B\abnormal_sample.npy"

    try:
        # device=None will automatically use GPU if CUDA is available
        scorer = ProjectBScorer(deployment_dir, device=None) 
    except Exception as e:
        print(f"Failed to initialize scorer: {e}", file=sys.stderr)
        sys.exit(1)

    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}", file=sys.stderr)
        sys.exit(1)

    result = scorer.score_sample(test_file, save_heatmap=True)

    print("\n" + "=" * 36)
    print(" INFERENCE RESULT")
    print("=" * 36)
    print(f"  Score     : {result['score']:.6f}")
    print(f"  Threshold : {result['threshold']:.6f}")
    print(f"  Status    : {'ANOMALY' if result['is_anomaly'] else ' NORMAL'}")
    if result["heatmap_path"]:
        print(f"  Heatmap   : {result['heatmap_path']}")
    print("=" * 36)