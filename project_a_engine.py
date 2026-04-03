
import os
import glob
import time
import json
import pickle
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
CFG = {
    "input_dim": 8,           
    "bottleneck_dim": 3,      
    "learning_rate": 1e-3,
    "epochs": 150,
    "batch_size": 64,
    "patience": 20,           # Early stopping patience
    "seed": 42,
    "dataset_path": "./data/healthy_machine_audio" # Point this to your folder
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

FEATURE_NAMES = [
    "ZCR", "RMS", "Spectral_Centroid", "Spectral_BW", 
    "Spectral_Rolloff", "Spectral_Flatness", "MFCC_1", "MFCC_2"
]

# ─────────────────────────────────────────────────────────────────
# DATA EXTRACTION (Real Librosa Framing)
# ─────────────────────────────────────────────────────────────────
def load_healthy_features(data_dir: str) -> np.ndarray:
    audio_files = glob.glob(os.path.join(data_dir, "*.wav"))
    
    # Fallback to synthetic if folder is missing so your script doesn't crash
    if not audio_files:
        print("⚠ WARNING: No .wav files found. Generating synthetic fallback data...")
        rng = np.random.default_rng(CFG["seed"])
        base = rng.normal(0, 1, (2000, 8))
        return (base * np.array([0.8, 0.6, 0.7, 0.5, 0.6, 0.4, 1.0, 0.9])).astype(np.float32)

    print(f"Loading {len(audio_files)} audio files...")
    all_features = []
    
    for path in audio_files:
        y, sr = librosa.load(path, sr=22050) # Maintain standard SR
        
        # Extract features (returns shape [1, time_steps] or [n_mfcc, time_steps])
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        flat = librosa.feature.spectral_flatness(y=y)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
        
        # Stack vertically: shape becomes [time_steps, 8]
        file_feats = np.stack([
            zcr, rms, cent, bw, roll, flat, mfccs[0], mfccs[1]
        ], axis=1)
        
        all_features.append(file_feats)
        
    # Concatenate all files into one massive dataset: [total_time_steps, 8]
    final_dataset = np.vstack(all_features)
    return final_dataset.astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class EdgeAutoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        mid = (input_dim + bottleneck_dim) // 2 + 1  

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, bottleneck_dim),
            nn.Tanh(),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP (With Early Stopping)
# ─────────────────────────────────────────────────────────────────
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    criterion = nn.MSELoss()

    history = []
    best_loss = float('inf')
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        
        train_loss /= len(train_loader.dataset)
        history.append(train_loss)

        # Validation Phase (Overfit check)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * len(batch)
        val_loss /= len(val_loader.dataset)

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "edge_ae_best.pth") # Save best weights
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG['epochs']} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Patience: {patience_counter}")

        if patience_counter >= CFG["patience"]:
            print(f"⚠ Early stopping triggered at epoch {epoch}")
            break

    print(f"\n✓ Training complete in {time.time()-t0:.1f}s")
    
    # Generate Training Plot
    plt.figure(figsize=(8, 4))
    plt.plot(history, label="Train MSE")
    plt.yscale('log')
    plt.title('Autoencoder Reconstruction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('train_curve.png', dpi=150)
    plt.close()
    
    # Reload the best weights before exiting
    model.load_state_dict(torch.load("edge_ae_best.pth"))
    return history

# ─────────────────────────────────────────────────────────────────
# THRESHOLD CALCULATION (Using Validation Data)
# ─────────────────────────────────────────────────────────────────
def compute_alarm_threshold(model: nn.Module, val_scaled_data: np.ndarray) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    tensor = torch.tensor(val_scaled_data, dtype=torch.float32)

    with torch.no_grad():
        recon = model(tensor)
        per_sample_mse = criterion(recon, tensor).mean(dim=1).numpy()

    threshold = float(np.percentile(per_sample_mse, 99))

    print(f"\n── Reconstruction Error Distribution (Validation Data) ──")
    print(f"   Median : {np.median(per_sample_mse):.6f}")
    print(f"   95th   : {np.percentile(per_sample_mse, 95):.6f}")
    print(f"   99th   : {threshold:.6f}  ← ALARM_THRESHOLD")

    with open("threshold.txt", "w") as f:
        f.write(f"{threshold:.8f}\n")
        f.write(f"# 99th percentile MSE on healthy validation data\n")

    return threshold

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(" PROJECT A — 1D EDGE AUTOENCODER (PRODUCTION PIPELINE)")
    print("=" * 65)

    # 1. Load Data
    raw_features = load_healthy_features(CFG["dataset_path"])
    print(f"\n✓ Extracted features shape: {raw_features.shape}")

    # 2. Train/Val Split (Overfit Guard)
    print("\n── Step 1: Split & Scale ──")
    train_raw, val_raw = train_test_split(raw_features, test_size=0.2, random_state=CFG["seed"])
    
    # Fit scaler strictly on training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    val_scaled = scaler.transform(val_raw).astype(np.float32)

    # Save Scaler artifacts
    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open("scaler_params.json", "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)

    # 3. Build DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(train_scaled)), batch_size=CFG["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_scaled)), batch_size=CFG["batch_size"], shuffle=False, pin_memory=True)

    # 4. Model Setup
    model = EdgeAutoencoder(CFG["input_dim"], CFG["bottleneck_dim"])
    
    # 5. Train
    print(f"\n── Step 2: Training ──")
    train(model, train_loader, val_loader)

    # 6. Advanced ONNX Export
    print(f"\n── Step 3: Deployment Export ──")
    model.eval()
    dummy_input = torch.randn(1, CFG["input_dim"], dtype=torch.float32)
    
    torch.onnx.export(
        model, dummy_input, "edge_ae.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,         # TFLite safe
        do_constant_folding=True  # Optimizes constants
    )
    print(f"✓ ONNX exported (opset_version 12) → edge_ae.onnx")

    # 7. Calculate Threshold on Validation Set
    print(f"\n── Step 4: Threshold Calibration ──")
    threshold = compute_alarm_threshold(model, val_scaled)

    print("\n" + "=" * 65)
    print(" DEPLOYMENT CHECKLIST")
    print("=" * 65)
    print(f"  [✓] scaler_params.json  — embedded in edge firmware")
    print(f"  [✓] edge_ae.onnx        — convert to TFLite!")
    print(f"  [✓] train_curve.png     — UI proof for the judges")
    print(f"  [✓] ALARM_THRESHOLD     = {threshold:.6f}")
    print("=" * 65)

if __name__ == "__main__":
    main()