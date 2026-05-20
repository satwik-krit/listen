import os
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Added for progress bars

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SPLIT_DIR  = r"C:\Users\risha\Downloads\listen\listen\split_output"
OUTPUT_DIR = r"C:\Users\risha\Downloads\listen\listen\edge_deployments"

# BLUEPRINT 1: Hardware & Data Loading (CUDA enabled)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15                  # Fast Hackathon limit
BATCH_SIZE = 64              # Maximize CPU core usage
IMG_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA UTILITIES & DATASET
# ─────────────────────────────────────────────
def parse_machine_id(meta_entry: str):
    parts = meta_entry.split("|")
    snr_machine = parts[0]
    machine_id  = parts[1]
    machine_type = snr_machine.split("_dB_")[-1] if "_dB_" in snr_machine else snr_machine
    return machine_type, machine_id

def group_by_machine_indices(y, meta):
    """Groups ONLY the indices to keep RAM usage near zero."""
    groups = {}
    for i, m in enumerate(meta):
        mtype, mid = parse_machine_id(m)
        key = f"{mtype}_{mid}"
        if key not in groups:
            groups[key] = {"indices": [], "y": []}
        groups[key]["indices"].append(i)
        groups[key]["y"].append(y[i])
        
    for key in groups:
        groups[key]["indices"] = np.array(groups[key]["indices"], dtype=np.int32)
        groups[key]["y"] = np.array(groups[key]["y"], dtype=np.int32)
    return groups

class MachineMelDataset(Dataset):
    def __init__(self, X_mel, stats):
        self.X = X_mel
        # BLUEPRINT 2: Preprocessing (Z-score Normalization)
        self.ch_mean = np.array(stats["ch_mean"], dtype=np.float32)
        self.ch_std  = np.array(stats["ch_std"], dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mel = self.X[idx].copy()
        for c in range(3):
            # Z-Score Normalization
            mel[c] = (mel[c] - self.ch_mean[c]) / self.ch_std[c]
             
        t = torch.from_numpy(mel).unsqueeze(0)
        # Interpolate to target size, then squeeze back to (C, H, W)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return t.squeeze(0)

# ─────────────────────────────────────────────
# 3. ARCHITECTURE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 3. ARCHITECTURE & HYBRID LOSS SYSTEM
# ─────────────────────────────────────────────
import librosa
from pytorch_msssim import ssim

def log_cosh_loss(recon, target):
    # Stable Log-Cosh implementation: |d| + softplus(-2|d|) - log 2
    d = recon - target
    abs_d = torch.abs(d)
    return (abs_d + F.softplus(-2.0 * abs_d) - np.log(2.0)).mean()

def harmonized_loss(recon, original):
    loss_logcosh = log_cosh_loss(recon, original)
    ssim_val = ssim(recon, original, data_range=1.0, size_average=True)
    loss_ssim = 1.0 - ssim_val
    return 0.8 * loss_logcosh + 0.2 * loss_ssim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholders to satisfy any legacy reflection/attributes queries
        self.enc1 = nn.Identity()

class Decoder(nn.Module):
    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def __init__(self):
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
    def forward(self, z):
        return self.dec5(self.dec4(self.dec3(self.dec2(self.dec1(z)))))

class CNNAutoencoder(nn.Module):
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
        
        # Hard 256-dimensional compression bottleneck layer
        # 256 channels * 3 heads * 2 height * 4 width = 6144
        self.bottleneck_linear = nn.Linear(6144, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.1)
        
        # Decoder projection block
        self.decoder_projection = nn.Linear(256, 4096)
        self.decoder = Decoder()
        self._init_weights()

    def _block(self, in_ch, out_ch, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self):
        # Xavier Uniform for linear bottleneck
        nn.init.xavier_uniform_(self.bottleneck_linear.weight)
        if self.bottleneck_linear.bias is not None:
            nn.init.zeros_(self.bottleneck_linear.bias)
            
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        if self.decoder_projection.bias is not None:
            nn.init.zeros_(self.decoder_projection.bias)
        
        # Kaiming Uniform for conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        # Exact Mel-Bin boundary calculations programmatically
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
        
        # Shape assertions to guarantee downsampled sizes never contract to 0 rows
        assert low_enc.shape[2] > 0 and low_enc.shape[3] > 0, f"Low-freq encoder contracted: {low_enc.shape}"
        assert mid_enc.shape[2] > 0 and mid_enc.shape[3] > 0, f"Mid-freq encoder contracted: {mid_enc.shape}"
        assert high_enc.shape[2] > 0 and high_enc.shape[3] > 0, f"High-freq encoder contracted: {high_enc.shape}"
        
        # Flatten and concatenate parallel maps
        low_flat = low_enc.reshape(low_enc.size(0), -1)
        mid_flat = mid_enc.reshape(mid_enc.size(0), -1)
        high_flat = high_enc.reshape(high_enc.size(0), -1)
        
        fused = torch.cat([low_flat, mid_flat, high_flat], dim=1)
        
        # Bottleneck compression
        compressed = self.bottleneck_linear(fused)
        compressed = self.layer_norm(compressed)
        compressed = self.dropout(compressed)
        return compressed

    def forward(self, x):
        z = self.encode(x)
        dec_proj = self.decoder_projection(z)
        dec_in = dec_proj.reshape(dec_proj.size(0), 256, 4, 4)
        return self.decoder(dec_in)

def get_lr_multiplier(epoch, num_epochs=15, warmup_epochs=5):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    else:
        progress = float(epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

def train_and_save_svm(X_scalar_healthy, out_dir):
    from sklearn.svm import OneClassSVM
    import pickle
    
    # Grid Search over gamma parameter
    param_grid = [1e-4, 1e-3, 1e-2, 1e-1, "scale", "auto"]
    best_gamma = "scale"
    best_score = float("-inf")
    
    for gamma in param_grid:
        svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=gamma)
        svm.fit(X_scalar_healthy)
        preds = svm.predict(X_scalar_healthy)
        outlier_ratio = np.mean(preds == -1)
        score = -abs(outlier_ratio - 0.01)
        if score > best_score:
            best_score = score
            best_gamma = gamma
            
    best_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=best_gamma)
    best_svm.fit(X_scalar_healthy)
    
    with open(os.path.join(out_dir, "one_class_svm.pkl"), "wb") as f:
        pickle.dump(best_svm, f)
    print(f"      ✓ Trained OneClassSVM (gamma={best_gamma}) saved!")

def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     Model B — Hackathon Survival Training Pipeline       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Device Active: {DEVICE}")
    
    train_folder = os.path.join(SPLIT_DIR, "train", "model_B")
    
    # THE RAM FIX: Memory map the array directly from NVMe drive
    print("\n[ 1 / 3 ] Memory-Mapping X_mel.npy...")
    X_mmap = np.load(os.path.join(train_folder, "X_mel.npy"), mmap_mode='r')
    X_scalar_all = np.load(os.path.join(train_folder, "X_scalar.npy")).astype(np.float32)
    y_all = np.load(os.path.join(train_folder, "y.npy")).astype(np.int32)
    with open(os.path.join(train_folder, "meta.txt")) as f:
        meta = [l.strip() for l in f]

    print("[ 2 / 3 ] Grouping by machine identity (Indices only)...")
    groups = group_by_machine_indices(y_all, meta)
    print(f"          Found {len(groups)} unique machines.\n")

    print("[ 3 / 3 ] Training CNN autoencoders...")
    
    # Check max threads available to avoid OS bottlenecks
    num_workers = 0  # Forces data loading to stay on the main thread

    for machine_name, data in sorted(groups.items()):
        indices = data["indices"]
        y_machine = data["y"]
        
        # Unsupervised: Train ONLY on healthy (normal) data
        normal_mask = (y_machine == 0)
        normal_indices = indices[normal_mask]
        
        if len(normal_indices) < 10:
            print(f"  ⚠ Skipping {machine_name}: Not enough normal data.")
            continue
             
        print(f"\n  ── {machine_name.upper()}  ({len(normal_indices)} normal samples)")
        out_dir = os.path.join(OUTPUT_DIR, machine_name)
        os.makedirs(out_dir, exist_ok=True)

        # Pull ONLY this machine's normal data into RAM (~200 MB maximum)
        X_normal = X_mmap[normal_indices].copy()
        X_scalar_machine = X_scalar_all[normal_indices]

        # Train OneClassSVM with hyperparameter grid-search CV
        train_and_save_svm(X_scalar_machine, out_dir)

        # BLUEPRINT 2: Stats Update (Z-score math)
        ch_mean = [float(X_normal[:, c, :, :].mean()) for c in range(3)]
        ch_std  = [float(X_normal[:, c, :, :].std()) + 1e-8 for c in range(3)] 
        stats = {"ch_mean": ch_mean, "ch_std": ch_std}
        
        with open(os.path.join(out_dir, "global_stats.json"), "w") as f:
            json.dump(stats, f)

        X_tr, X_val = train_test_split(X_normal, test_size=0.15, random_state=42)
        
        # BLUEPRINT 1: Supercharge DataLoaders
        train_loader = DataLoader(
            MachineMelDataset(X_tr, stats), 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True             
        )
        val_loader = DataLoader(
            MachineMelDataset(X_val, stats), 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        model = CNNAutoencoder().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Linear lr warmup + cosine decay scheduler
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: get_lr_multiplier(ep, num_epochs=EPOCHS, warmup_epochs=5))
        
        # BLUEPRINT 3: Initialize the AMP Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
        
        best_val = float('inf')
        t0 = time.time()
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            # PROGRESS BAR INTEGRATION (Train)
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
            
            for batch in train_pbar:
                # Hutchinson Trace requires requires_grad=True
                batch = batch.to(DEVICE, non_blocking=True)
                batch.requires_grad_(True)
                assert batch.requires_grad, "Graph Tracker Failure: Input tensor requires_grad is False."

                opt.zero_grad(set_to_none=True) 
                
                # BLUEPRINT 3: Mixed Precision Forward Pass
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                    encoded = model.encode(batch)
                    dec_proj = model.decoder_projection(encoded)
                    dec_in = dec_proj.reshape(dec_proj.size(0), 256, 4, 4)
                    recon = model.decoder(dec_in)
                    
                    # Harmonized reconstruction loss
                    loss_recon = harmonized_loss(recon, batch)
                    
                # Hutchinson Trace Estimator for Contractive Jacobian Penalty
                v = torch.randn_like(encoded)
                vjp = torch.autograd.grad(encoded, batch, grad_outputs=v, create_graph=True)[0]
                loss_contractive = 1e-4 * (vjp ** 2).sum()
                
                loss = loss_recon + loss_contractive
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                train_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())
                
            train_loss /= len(train_loader)
            scheduler.step()
            
            model.eval()
            val_loss = 0.0
            val_scores = []
            
            # PROGRESS BAR INTEGRATION (Val)
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  ", leave=False)
            
            with torch.no_grad():
                for batch in val_pbar:
                    batch = batch.to(DEVICE, non_blocking=True)
                    
                    # Also use AMP in validation for faster inference
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                        recon = model(batch)
                        for i in range(len(batch)):
                            s_loss = harmonized_loss(recon[i:i+1], batch[i:i+1])
                            val_scores.append(s_loss.item())
                            val_loss += s_loss.item()
            
            val_loss /= len(val_loader.dataset)
            
            # Clean print for epoch summary
            print(f"      Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            if val_loss < best_val:
                best_val = val_loss
                
                # BLUEPRINT 4: Mean + 3*Std Threshold calculation
                val_mean = np.mean(val_scores)
                val_std  = np.std(val_scores)
                best_threshold = val_mean + (3 * val_std)
                
                torch.save(model.state_dict(), os.path.join(out_dir, 'cnn_ae_best.pth'))
                with open(os.path.join(out_dir, "threshold_B.txt"), "w") as f:
                    f.write(f"{best_threshold:.8f}\n")
                    
        print(f"     ✓ Done in {time.time()-t0:.1f}s | Best Val: {best_val:.5f} | Threshold: {best_threshold:.5f}")
        
        # FREE UP RAM BEFORE THE NEXT MACHINE STARTS
        del X_normal, X_tr, X_val, train_loader, val_loader, model
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n🎉 Training Complete! All models saved to Edge Deployments.")

if __name__ == "__main__":
    main()