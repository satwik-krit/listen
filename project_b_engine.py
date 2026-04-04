
import os
import json
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_msssim import ssim 
import librosa

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
CFG = {
    "wav_dir": "./healthy_wavs",       
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "clip_duration_sec": 2.0,          
    "img_size": 128,                   
    "base_channels": 32,               
    "bottleneck_channels": 256,
    "epochs": 80,
    "batch_size": 16,                  
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,              
    "grad_clip_norm": 1.0,             
    "val_split": 0.15,
    "early_stopping_patience": 15,
    "num_workers": 4,
    "seed": 42,
    "loss_mse_weight": 0.5,
    "loss_ssim_weight": 0.5,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────
# GLOBAL NORMALISATION (Independent Channel Bounds)
# ─────────────────────────────────────────────────────────────────
def compute_global_stats(wav_paths: list[str]) -> dict:
    print("  Computing independent global bounds for Mel, Delta, and Delta-Delta...")
    
    stats = {
        "ch0": {"min": float('inf'), "max": float('-inf')},
        "ch1": {"min": float('inf'), "max": float('-inf')},
        "ch2": {"min": float('inf'), "max": float('-inf')}
    }

    for path in wav_paths:
        try:
            y, sr = librosa.load(path, sr=CFG["sample_rate"])
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], n_mels=CFG["n_mels"], norm='slaney'
            )
            db = librosa.power_to_db(mel, ref=np.max)
            delta = librosa.feature.delta(db)
            delta2 = librosa.feature.delta(db, order=2)

            if db.min() < stats["ch0"]["min"]: stats["ch0"]["min"] = float(db.min())
            if db.max() > stats["ch0"]["max"]: stats["ch0"]["max"] = float(db.max())
            
            if delta.min() < stats["ch1"]["min"]: stats["ch1"]["min"] = float(delta.min())
            if delta.max() > stats["ch1"]["max"]: stats["ch1"]["max"] = float(delta.max())
            
            if delta2.min() < stats["ch2"]["min"]: stats["ch2"]["min"] = float(delta2.min())
            if delta2.max() > stats["ch2"]["max"]: stats["ch2"]["max"] = float(delta2.max())
            
        except Exception as e:
            print(f"    Skipping {path}: {e}")

    print(f"  ✓ Mel Bounds  : [{stats['ch0']['min']:.2f}, {stats['ch0']['max']:.2f}]")
    print(f"  ✓ D1 Bounds   : [{stats['ch1']['min']:.2f}, {stats['ch1']['max']:.2f}]")
    print(f"  ✓ D2 Bounds   : [{stats['ch2']['min']:.2f}, {stats['ch2']['max']:.2f}]")

    return stats


def wav_to_3ch_tensor(y: np.ndarray, sr: int, global_stats: dict, img_size: int = 128) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], n_mels=CFG["n_mels"], norm='slaney'
    )
    db = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(db)
    delta2 = librosa.feature.delta(db, order=2)

    def global_normalize(arr, c_min, c_max):
        return np.clip((arr - c_min) / (c_max - c_min + 1e-8), 0.0, 1.0)

    db_norm = global_normalize(db, global_stats["ch0"]["min"], global_stats["ch0"]["max"])
    d1_norm = global_normalize(delta, global_stats["ch1"]["min"], global_stats["ch1"]["max"])
    d2_norm = global_normalize(delta2, global_stats["ch2"]["min"], global_stats["ch2"]["max"])

    stacked = torch.tensor(np.stack([db_norm, d1_norm, d2_norm], axis=0), dtype=torch.float32).unsqueeze(0)
    resized = F.interpolate(stacked, size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)

    return resized 

# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class MelSpectrogramDataset(Dataset):
    def __init__(self, wav_paths: list[str], global_stats: dict):
        self.global_stats = global_stats
        self.tensors: list[torch.Tensor] = []
        self._load_and_process_all(wav_paths)

    def _load_and_process_all(self, paths: list[str]):
        clip_len = int(CFG["clip_duration_sec"] * CFG["sample_rate"])
        print("  Pre-computing 3-Channel Tensors into RAM...")
        for path in paths:
            try:
                y, _ = librosa.load(path, sr=CFG["sample_rate"])
                for start in range(0, len(y) - clip_len, clip_len):
                    clip = y[start : start + clip_len]
                    tensor = wav_to_3ch_tensor(clip, CFG["sample_rate"], self.global_stats, CFG["img_size"])
                    self.tensors.append(tensor)
            except Exception:
                continue
        print(f"  ✓ Pre-computation complete: {len(self.tensors)} tensors ready.")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]

# ─────────────────────────────────────────────────────────────────
# MODEL (With 1x1 Final Conv)
# ─────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)

class TransposeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class EnterpriseAutoencoder(nn.Module):
    def __init__(self, base_ch: int = 32, bottleneck_ch: int = 256):
        super().__init__()
        b = base_ch

        self.encoder = nn.Sequential(
            ConvBlock(3,    b),        
            ConvBlock(b,    b*2),      
            ConvBlock(b*2,  b*4),      
            ConvBlock(b*4,  bottleneck_ch),  
        )

        self.decoder = nn.Sequential(
            TransposeBlock(bottleneck_ch, b*4),   
            TransposeBlock(b*4, b*2),             
            TransposeBlock(b*2, b),               
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Optimized: 1x1 final convolution reduces boundary artifacts
            nn.Conv2d(b, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),              
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

# ─────────────────────────────────────────────────────────────────
# LOSS (Fixed SSIM API)
# ─────────────────────────────────────────────────────────────────
def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    mse = F.mse_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=1.0).mean() # Fixed API call
    ssim_loss_val = 1.0 - ssim_val
    
    total = CFG["loss_mse_weight"] * mse + CFG["loss_ssim_weight"] * ssim_loss_val
    return total, {"mse": mse.item(), "ssim": ssim_loss_val.item(), "total": total.item()}

# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP (With Early Stopping)
# ─────────────────────────────────────────────────────────────────
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    t0 = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        train_stats = {"mse": 0.0, "ssim": 0.0, "total": 0.0}
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss, stats = combined_loss(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
            optimizer.step()
            for k in train_stats: train_stats[k] += stats[k] * len(batch)

        train_stats = {k: v / len(train_loader.dataset) for k, v in train_stats.items()}

        model.eval()
        val_stats = {"mse": 0.0, "ssim": 0.0, "total": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                _, stats = combined_loss(recon, batch)
                for k in val_stats: val_stats[k] += stats[k] * len(batch)

        val_stats = {k: v / len(val_loader.dataset) for k, v in val_stats.items()}
        scheduler.step()

        # Early Stopping & Checkpointing
        if val_stats["total"] < best_val_loss:
            best_val_loss = val_stats["total"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_ae.pth")
            ckpt_flag = " ← ✓ saved"
        else:
            ckpt_flag = ""
            if epoch > 10 and val_stats["total"] > best_val_loss * 1.01:
                patience_counter += 1

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{CFG['epochs']} | Train={train_stats['total']:.5f} | Val={val_stats['total']:.5f} | Pat={patience_counter}/{CFG['early_stopping_patience']} {ckpt_flag}")

        if patience_counter > CFG["early_stopping_patience"]:
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break

    print(f"\n✓ Training complete in {time.time()-t0:.1f}s | best_val_loss={best_val_loss:.6f}")
    return history

# ─────────────────────────────────────────────────────────────────
# THRESHOLD CALCULATION
# ─────────────────────────────────────────────────────────────────
def compute_alarm_threshold(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    model.to(DEVICE)
    heatmap_means = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            heatmaps = (recon - batch).pow(2)               
            means = heatmaps.mean(dim=[1, 2, 3]).cpu().numpy()  
            heatmap_means.extend(means.tolist())

    hm = np.array(heatmap_means)
    threshold = float(np.percentile(hm, 95))
    sigma2 = float(np.mean(hm) + 2 * np.std(hm))

    print(f"\n── Heatmap Mean Distribution (Validation) ──")
    print(f"   Median : {np.median(hm):.6f}")
    print(f"   95th   : {threshold:.6f}  ← ALARM_THRESHOLD  (1σ tripwire)")
    print(f"   2σ line: {sigma2:.6f}  (Critical alarm)")

    with open("threshold_b.txt", "w") as f:
        f.write(f"THRESHOLD_1SIGMA = {threshold:.8f}\n")
        f.write(f"THRESHOLD_2SIGMA = {sigma2:.8f}\n")

    return threshold

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" PROJECT B — 2D ENTERPRISE MEL-SPECTROGRAM AUTOENCODER")
    print("=" * 70)

    wav_paths = []
    if os.path.isdir(CFG["wav_dir"]):
        wav_paths = glob.glob(os.path.join(CFG["wav_dir"], "**", "*.wav"), recursive=True)
        print(f"\n✓ Found {len(wav_paths)} .wav files in {CFG['wav_dir']}")

    print("\n── Step 1: Global Stats (3-Channel) ──")
    if wav_paths:
        global_stats = compute_global_stats(wav_paths)
    else:
        # Fallback bounds for dry runs
        global_stats = {
            "ch0": {"min": -80.0, "max": 0.0},
            "ch1": {"min": -20.0, "max": 20.0},
            "ch2": {"min": -20.0, "max": 20.0}
        }
        print(f"  ⚠ Run with real .wav files for accurate limits.")

    with open("global_stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    print(f"\n── Step 2: Dataset Building ──")
    full_dataset = MelSpectrogramDataset(wav_paths, global_stats)

    val_size  = max(1, int(len(full_dataset) * CFG["val_split"]))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

    print(f"\n── Step 3: Model Setup ──")
    model = EnterpriseAutoencoder(base_ch=CFG["base_channels"], bottleneck_ch=CFG["bottleneck_channels"]).to(DEVICE)
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n── Step 4: Training ──")
    history = train_model(model, train_loader, val_loader)

    print(f"\n── Step 5: Post-Processing & Exports ──")
    model.load_state_dict(torch.load("best_ae.pth", map_location=DEVICE))
    
    compute_alarm_threshold(model, val_loader)

    model.eval()
    dummy = torch.randn(1, 3, CFG["img_size"], CFG["img_size"]).to(DEVICE)
    torch.onnx.export(
        model, dummy, "enterprise_ae.onnx", 
        opset_version=13, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"✓ ONNX Exported → enterprise_ae.onnx")

    print("\n" + "=" * 70)
    print(" DEPLOYMENT CHECKLIST")
    print("=" * 70)
    print(f"  [✓] global_stats.json    — Independent bounds for Ch0/Ch1/Ch2")
    print(f"  [✓] best_ae.pth          — Native PyTorch GPU weights")
    print(f"  [✓] enterprise_ae.onnx   — Optimized standard format export")
    print(f"  [✓] threshold_b.txt      — Dashboard 1σ/2σ tripwire values")
    print("=" * 70)

if __name__ == "__main__":
    main()