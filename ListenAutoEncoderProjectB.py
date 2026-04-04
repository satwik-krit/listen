
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_msssim import ssim 

# ─────────────────────────────────────────────
# 1. DYNAMIC DATASET STATS
# ─────────────────────────────────────────────
def compute_global_stats(root_dir, sr=22050, duration=10.0, n_mels=128):
    print("Computing global normalization bounds...")
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".wav")]
    global_min = float('inf')
    global_max = float('-inf')
    
    for f in files:
        y, _ = librosa.load(f, sr=sr, duration=duration)
        # Added norm='slaney' for perceptual uniformity
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, norm='slaney')
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        if mel_db.min() < global_min: global_min = mel_db.min()
        if mel_db.max() > global_max: global_max = mel_db.max()
        
    print(f"Stats Locked - Min DB: {global_min:.2f}, Max DB: {global_max:.2f}")
    return global_min, global_max

# ─────────────────────────────────────────────
# 2. DATASET PIPELINE
# ─────────────────────────────────────────────
class MelSpectrogramDataset(Dataset):
    def __init__(self, root_dir, global_min_db, global_max_db, sr=22050, duration=10.0, 
                 n_mels=128, n_fft=2048, hop_length=512, target_size=(128, 128), augment=False):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".wav")]
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_size = target_size 
        self.augment = augment
        
        self.global_min = global_min_db
        self.global_max = global_max_db

    def __len__(self):
        return len(self.files)

    def _load_and_augment(self, path):
        y, sr = librosa.load(path, sr=self.sr, duration=self.duration)
        if self.augment:
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
            y = y + (0.001 * np.random.randn(len(y)))
        return y, sr

    def _make_3channel(self, y):
        # Applied norm='slaney' here as well
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, norm='slaney'
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        delta  = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)

        def global_normalize(arr):
            norm = (arr - self.global_min) / (self.global_max - self.global_min + 1e-8)
            return np.clip(norm, 0.0, 1.0) 

        ch0 = torch.tensor(global_normalize(mel_db), dtype=torch.float32).unsqueeze(0)
        ch1 = torch.tensor(global_normalize(delta), dtype=torch.float32).unsqueeze(0)
        ch2 = torch.tensor(global_normalize(delta2), dtype=torch.float32).unsqueeze(0)

        stacked = torch.cat([ch0, ch1, ch2], dim=0) 
        return F.interpolate(stacked.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
    
    def __getitem__(self, idx):
        y, sr = self._load_and_augment(self.files[idx])
        return self._make_3channel(y)

# ─────────────────────────────────────────────
# 3. ENCODER & DECODER ARCHITECTURE
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
        self.enc5 = self._block(256, 256)  # 4x4 Bottleneck 

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
        z = self.encoder(x)
        return self.decoder(z)

# ─────────────────────────────────────────────
# 4. LOSS, TRAINING LOOP & INFERENCE
# ─────────────────────────────────────────────
def combined_loss(recon, original, alpha=0.5):
    mse_loss = nn.functional.mse_loss(recon, original)
    # Fixed SSIM API call
    ssim_val = ssim(recon, original, data_range=1.0, size_average=False).mean()
    ssim_loss = 1.0 - ssim_val
    return alpha * mse_loss + (1.0 - alpha) * ssim_loss

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    model.to(device)
    # Upgraded to AdamW
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    
    best_val = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = combined_loss(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader) # Average loss per batch
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # Critical memory fix
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = combined_loss(recon, batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader) # Average loss per batch
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val: 
            best_val = val_loss
            torch.save(model.state_dict(), 'best_ae.pth')
            print("  -> New best model saved!")

@torch.no_grad()
def compute_anomaly_score(model, spectrogram_tensor, device):
    model.eval() 
    x = spectrogram_tensor.unsqueeze(0).to(device)
    recon = model(x)
    
    pixel_error = ((x - recon) ** 2)          
    heatmap = pixel_error.mean(dim=1)      
    score = heatmap.mean().item()        

    return score, heatmap.squeeze().cpu().numpy()