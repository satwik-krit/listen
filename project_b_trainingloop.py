import os
import torch
# Import the engine we just built
from project_b_engine import compute_global_stats, MelSpectrogramDataset, CNNAutoencoder, train_model, compute_anomaly_score

# ─────────────────────────────────────────────────────────────────
# 1. SETUP YOUR DIRECTORIES
# ─────────────────────────────────────────────────────────────────
# Assume your MIMII dataset is unzipped like this:
# dataset/
#   ├── pump/
#   │   ├── id_00/
#   │   │   ├── normal/
#   │   │   └── abnormal/
#   │   ├── id_02/ ...
#   ├── fan/ ...

BASE_DATASET_DIR = "./dataset"
OUTPUT_DIR = "./trained_models"

MACHINE_TYPES = ["pump", "fan", "valve", "slider"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
# 2. THE MASTER LOOP
# ─────────────────────────────────────────────────────────────────
def run_master_training():
    for m_type in MACHINE_TYPES:
        for m_id in MACHINE_IDS:
            print("=" * 60)
            print(f"🚀 STARTING PIPELINE: {m_type.upper()} - {m_id.upper()}")
            print("=" * 60)
            
            # 1. Define Paths
            healthy_train_dir = os.path.join(BASE_DATASET_DIR, m_type, m_id, "normal")
            
            # Create a specific folder for this machine's outputs
            machine_out_dir = os.path.join(OUTPUT_DIR, f"{m_type}_{m_id}")
            os.makedirs(machine_out_dir, exist_ok=True)
            
            weights_path = os.path.join(machine_out_dir, "best_ae.pth")
            
            # 2. Get Machine-Specific Stats
            # A pump has totally different decibel ranges than a valve.
            global_min, global_max = compute_global_stats(healthy_train_dir)
            
            # Save these stats so the inference script can use them tomorrow
            with open(os.path.join(machine_out_dir, "norm_stats.txt"), "w") as f:
                f.write(f"{global_min},{global_max}")

            # 3. Load Data
            train_ds = MelSpectrogramDataset(healthy_train_dir, global_min, global_max, augment=True)
            
            # Note: For hackathons, if you don't have a separate validation set, 
            # you can split train_ds or just evaluate on training data to save time.
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
            
            # 4. Initialize Fresh Model
            model = CNNAutoencoder()
            
            # 5. Train
            print(f"Training {m_type} {m_id}...")
            # Modify your train_model function to accept the weights_path
            train_model(model, train_loader, val_loader=train_loader, epochs=50, device=device, save_path=weights_path)
            
            # 6. Calculate Threshold (Optional: Add this logic inside the loop)
            print(f"✅ Finished {m_type} {m_id}. Weights saved to {machine_out_dir}\n")

if __name__ == "__main__":
    run_master_training()