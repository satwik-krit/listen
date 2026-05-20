import json
import os
import warnings
import pickle
from pathlib import Path
from typing import Optional

import librosa
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARCHITECTURE  (must exactly match training)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Identity()

class Decoder(nn.Module):
    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
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

class CNNAutoencoder(nn.Module):
    def __init__(self) -> None:
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
            nn.LeakyReLU(0.2, inplace=True),
        )

    def encode(self, x):
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
        
        # Shape assertions
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

    def forward(self, x):
        z = self.encode(x)
        dec_proj = self.decoder_projection(z)
        dec_in = dec_proj.reshape(dec_proj.size(0), 256, 4, 4)
        return self.decoder(dec_in)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SCORER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_STAT_KEYS = {"ch_mean", "ch_std"}

class ProjectBScorer:

    # ── construction ────────────────────────────────────────────────────────

    def __init__(
        self,
        deployment_folder: str | os.PathLike,
        device: Optional[str] = None,
        output_dir: Optional[str | os.PathLike] = None,
    ) -> None:
        self._deployment = Path(deployment_folder)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self._output_dir = Path(output_dir) if output_dir else self._deployment / "heatmaps"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialising scorer: {self._deployment.name}")

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
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        # ── OneClassSVM loader ─────────────────────────────────────────────
        svm_path = self._deployment / "one_class_svm.pkl"
        if svm_path.exists():
            with svm_path.open("rb") as f:
                self._svm = pickle.load(f)
            print(f"[✓] OneClassSVM ready from {svm_path.name}")
        else:
            self._svm = None
            print("[!] No OneClassSVM found — kinematic predictions disabled.")

        # Bivariate Alert Persistence Trackers
        self.yellow_consecutive = 0
        self.red_consecutive = 0

        print(
            f"Ready | threshold={self.threshold:.6f} | "
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
        mel = self._load_and_check(npy_path)

        mel = mel.squeeze()
        if mel.ndim == 1:
            mel = mel.reshape(1, -1)
        if mel.ndim != 2:
            raise ValueError(
                f"Cannot reduce mel array of shape {mel.shape} to 2-D. "
                "Expected (n_mels, T) or squeeze-able to it."
            )

        db: np.ndarray = (
            librosa.power_to_db(mel, ref=np.max) if self.is_power_spec else mel
        )

        delta  = librosa.feature.delta(db)
        delta2 = librosa.feature.delta(db, order=2)

        db_norm     = (db - self.ch_mean[0]) / self.ch_std[0]
        delta_norm  = (delta - self.ch_mean[1]) / self.ch_std[1]
        delta2_norm = (delta2 - self.ch_mean[2]) / self.ch_std[2]

        stacked = np.stack([db_norm, delta_norm, delta2_norm], axis=0)

        tensor = torch.from_numpy(stacked).unsqueeze(0)
        return F.interpolate(
            tensor, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )

    def _save_heatmap(
        self,
        diff: np.ndarray,
        stem: str,
    ) -> Path:
        heatmap_path = self._output_dir / f"{stem}_heatmap.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        try:
            ax.imshow(diff, cmap="hot", aspect="auto")
            ax.axis("off")
            fig.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        finally:
            plt.close(fig)
        return heatmap_path

    @torch.no_grad()
    def score_sample(
        self,
        npy_path: str | os.PathLike,
        X_scalar_features: Optional[np.ndarray] = None,
        save_heatmap: bool = False,
    ) -> dict:
    
        tensor = self.preprocess(npy_path).to(self.device, non_blocking=True)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            recon = self._model(tensor)
            
            # Local Anomaly Pooling using MaxPool2d(8, 8)
            loss_map = ((tensor - recon) ** 2).mean(dim=1, keepdim=True)
            max_pool = nn.MaxPool2d(kernel_size=8, stride=8)
            pooled_loss = max_pool(loss_map)
            spatial_score = float(pooled_loss.max().item())

        # SVM Anomaly check
        svm_anomaly = False
        if self._svm is not None and X_scalar_features is not None:
            feats = X_scalar_features.reshape(1, -1)
            pred = self._svm.predict(feats)[0]
            svm_anomaly = (pred == -1)

        spatial_anomaly = spatial_score > self.threshold
        
        yellow_triggered = spatial_anomaly or svm_anomaly
        red_triggered = spatial_anomaly and svm_anomaly

        # RED Alert persistence: P >= 3
        if red_triggered:
            self.red_consecutive += 1
        else:
            self.red_consecutive = 0

        # YELLOW Alert persistence: P >= 8
        if yellow_triggered:
            self.yellow_consecutive += 1
        else:
            self.yellow_consecutive = 0

        # Determine Alert State
        if self.red_consecutive >= 3:
            alert_state = "RED"
        elif self.yellow_consecutive >= 8:
            alert_state = "YELLOW"
        else:
            alert_state = "NOMINAL"

        heatmap_path: Optional[Path] = None
        if save_heatmap:
            diff = ((tensor[0, 0].float() - recon[0, 0].float()) ** 2).cpu().numpy()
            heatmap_path = self._save_heatmap(diff, stem=Path(npy_path).stem)

        return {
            "score":        spatial_score,
            "threshold":    self.threshold,
            "is_anomaly":   alert_state != "NOMINAL",
            "alert_state":  alert_state,
            "heatmap_path": heatmap_path,
        }

    def score_batch(
        self,
        npy_paths: list[str | os.PathLike],
        X_scalar_batch: Optional[list[np.ndarray]] = None,
        save_heatmaps: bool = False,
    ) -> list[dict]:
        results = []
        for i, p in enumerate(npy_paths):
            feats = X_scalar_batch[i] if X_scalar_batch is not None else None
            results.append(self.score_sample(p, X_scalar_features=feats, save_heatmap=save_heatmaps))
        return results

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    deployment_dir = r"C:\Users\risha\Downloads\listen\listen\edge_deployments\fan_id_00"
    test_file      = r"C:\Users\risha\Downloads\listen\listen\split_output\test\model_B\abnormal_sample.npy"

    try:
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
    print(f"  Alert State: {result.get('alert_state', 'NOMINAL')}")
    print(f"  Status    : {'ANOMALY' if result['is_anomaly'] else ' NORMAL'}")
    if result["heatmap_path"]:
        print(f"  Heatmap   : {result['heatmap_path']}")
    print("=" * 36)