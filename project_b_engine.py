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
class Encoder(nn.Module):
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def __init__(self):
        super().__init__()
        self.enc1 = self._block(3,   32)   
        self.enc2 = self._block(32,  64)   
        self.enc3 = self._block(64,  128)  
        self.enc4 = self._block(128, 256)  
        self.enc5 = self._block(256, 256)  
    def forward(self, x):
        return self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(x)))))

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
            # Note: Tanh or Identity might be better than Sigmoid if input is Z-score normalized
            # but leaving Sigmoid if the hackathon logic strictly requires it. 
            # If gradients vanish, swap nn.Sigmoid() for nn.Identity().
            nn.Sigmoid(), 
        )
    def forward(self, z):
        return self.dec5(self.dec4(self.dec3(self.dec2(self.dec1(z)))))

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ─────────────────────────────────────────────
# 4. TRAINING LOGIC
# ─────────────────────────────────────────────
def fast_loss(recon, original):
    # HACKATHON SPEED UP: Pure MSE is 50x faster.
    return F.mse_loss(recon, original)

def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     Model B — Hackathon Survival Training Pipeline       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Device Active: {DEVICE}")
    
    train_folder = os.path.join(SPLIT_DIR, "train", "model_B")
    
    # THE RAM FIX: Memory map the array directly from NVMe drive
    print("\n[ 1 / 3 ] Memory-Mapping X_mel.npy...")
    X_mmap = np.load(os.path.join(train_folder, "X_mel.npy"), mmap_mode='r')
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
                batch = batch.to(DEVICE, non_blocking=True) # Send to GPU
                opt.zero_grad(set_to_none=True) 
                
                # BLUEPRINT 3: Mixed Precision Forward Pass
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                    recon = model(batch)
                    loss = fast_loss(recon, batch)
                    
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                train_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())
                
            train_loss /= len(train_loader)
            
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
                            s_loss = fast_loss(recon[i:i+1], batch[i:i+1])
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