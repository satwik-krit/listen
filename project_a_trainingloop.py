
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
# 1. CONFIGURATION & DIRECTORIES
# ─────────────────────────────────────────────────────────────────
CFG = {
    "input_dim": 8,           
    "bottleneck_dim": 3,      
    "learning_rate": 1e-3,
    "epochs": 150,
    "batch_size": 64,
    "patience": 20,           
    "seed": 42,
}

# Assumes MIMII dataset structure: dataset/pump/id_00/normal/
BASE_DATASET_DIR = "./dataset"
OUTPUT_DIR = "./edge_deployments"

MACHINE_TYPES = ["pump", "fan", "valve", "slider"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

FEATURE_NAMES = [
    "ZCR", "RMS", "Spectral_Centroid", "Spectral_BW", 
    "Spectral_Rolloff", "Spectral_Flatness", "MFCC_1", "MFCC_2"
]

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 2. CORE COMPONENTS (Model & Extraction)
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

def load_healthy_features(data_dir: str) -> np.ndarray:
    audio_files = glob.glob(os.path.join(data_dir, "*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    all_features = []
    for path in audio_files:
        y, sr = librosa.load(path, sr=22050)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        flat = librosa.feature.spectral_flatness(y=y)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
        
        file_feats = np.stack([zcr, rms, cent, bw, roll, flat, mfccs[0], mfccs[1]], axis=1)
        all_features.append(file_feats)
        
    return np.vstack(all_features).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 3. TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────
def train_edge_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, out_dir: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    criterion = nn.MSELoss()

    history = []
    best_loss = float('inf')
    patience_counter = 0

    best_weights_path = os.path.join(out_dir, "edge_ae_best.pth")

    for epoch in range(1, CFG["epochs"] + 1):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * len(batch)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path) 
        else:
            patience_counter += 1

        if patience_counter >= CFG["patience"]:
            break

    # Save curve
    plt.figure(figsize=(6, 3))
    plt.plot(history, label="Train MSE")
    plt.yscale('log')
    plt.title('Training Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_curve.png'), dpi=100)
    plt.close()
    
    model.load_state_dict(torch.load(best_weights_path))

def compute_alarm_threshold(model: nn.Module, val_scaled_data: np.ndarray, out_dir: str) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    tensor = torch.tensor(val_scaled_data, dtype=torch.float32)

    with torch.no_grad():
        recon = model(tensor)
        per_sample_mse = criterion(recon, tensor).mean(dim=1).numpy()

    threshold = float(np.percentile(per_sample_mse, 99))

    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(f"{threshold:.8f}\n")
        f.write(f"# 99th percentile MSE on healthy validation data\n")

    return threshold

# ─────────────────────────────────────────────────────────────────
# 4. THE MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_master_pipeline():
    print("=" * 65)
    print(" PROJECT A — INITIATING 16-MACHINE AUTOMATION LOOP")
    print("=" * 65)

    t_global = time.time()
    success_count = 0

    for m_type in MACHINE_TYPES:
        for m_id in MACHINE_IDS:
            machine_name = f"{m_type}_{m_id}"
            data_dir = os.path.join(BASE_DATASET_DIR, m_type, m_id, "normal")
            out_dir = os.path.join(OUTPUT_DIR, machine_name)
            
            print(f"\n🚀 Processing: {machine_name.upper()}")
            
            if not os.path.exists(data_dir):
                print(f"  ⚠ Skipping: Directory {data_dir} not found.")
                continue
                
            os.makedirs(out_dir, exist_ok=True)

            try:
                # 1. Load & Split
                raw_features = load_healthy_features(data_dir)
                train_raw, val_raw = train_test_split(raw_features, test_size=0.2, random_state=CFG["seed"])
                
                # 2. Scale
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
                val_scaled = scaler.transform(val_raw).astype(np.float32)

                with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f: 
                    pickle.dump(scaler, f)
                with open(os.path.join(out_dir, "scaler_params.json"), "w") as f:
                    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)

                # 3. Loaders
                train_loader = DataLoader(TensorDataset(torch.tensor(train_scaled)), batch_size=CFG["batch_size"], shuffle=True)
                val_loader = DataLoader(TensorDataset(torch.tensor(val_scaled)), batch_size=CFG["batch_size"], shuffle=False)

                # 4. Train
                model = EdgeAutoencoder(CFG["input_dim"], CFG["bottleneck_dim"])
                train_edge_model(model, train_loader, val_loader, out_dir)

                # 5. Export ONNX
                model.eval()
                dummy_input = torch.randn(1, CFG["input_dim"], dtype=torch.float32)
                torch.onnx.export(
                    model, dummy_input, os.path.join(out_dir, "edge_ae.onnx"),
                    input_names=["input"], output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    opset_version=12, do_constant_folding=True
                )

                # 6. Threshold
                threshold = compute_alarm_threshold(model, val_scaled, out_dir)
                
                print(f"  ✓ {machine_name} Complete. Threshold: {threshold:.6f}")
                success_count += 1

            except Exception as e:
                print(f"  ❌ Error processing {machine_name}: {e}")

    print("\n" + "=" * 65)
    print(f" DONE. Successfully trained {success_count}/16 edge models.")
    print(f" Total time elapsed: {(time.time() - t_global)/60:.1f} minutes.")
    print(f" All deployments securely isolated in: {OUTPUT_DIR}/")
    print("=" * 65)

if __name__ == "__main__":
    run_master_pipeline()