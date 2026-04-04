"""
Model B — CNN Autoencoder Training Pipeline
============================================
Loads directly from split_output produced by split_dataset.py:

  split_output/
    train/
      model_B/
        X_scalar.npy   (N_train, 8)        <- not used here, Model A's job
        X_mel.npy      (N_train, 3, 128, T) <- mel / delta / delta-delta already baked in
        y.npy          (N_train,)
        meta.txt
    test/
      model_B/
        X_mel.npy      (N_test,  3, 128, T)
        y.npy          (N_test,)
        meta.txt

Training is unsupervised on NORMAL samples only.
Anomaly score = combined MSE + (1 - SSIM) reconstruction loss.
Threshold     = 99th percentile on healthy validation data.

Output per machine-id:
  edge_deployments/{machine}_{id}/
    cnn_ae_best.pth
    cnn_ae.onnx
    global_stats.json      <- per-channel min/max fitted on train-normal
    threshold_B.txt
    train_curve_B.png

Requirements:
  pip install numpy torch torchvision pytorch-msssim scikit-learn tqdm matplotlib
"""

import os
import gc
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_msssim import ssim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CFG = {
    "img_size"      : 128,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-5,
    "epochs"        : 100,
    "batch_size"    : 16,
    "patience"      : 20,
    "val_split"     : 0.2,
    "loss_alpha"    : 0.5,      # weight: alpha*MSE + (1-alpha)*(1-SSIM)
    "seed"          : 42,
    "num_workers"   : 4,        # DataLoader workers
}

SPLIT_DIR  = r"C:\Users\risha\Downloads\listen\listen\split_output"
OUTPUT_DIR = r"C:\Users\risha\Downloads\listen\listen\edge_deployments"

DEVICE = torch.device("cpu")
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── Model (verbatim from your architecture doc) ───────────────────────────────

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
        self.enc5 = self._block(256, 256)   # 4×4 bottleneck

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
        return self.decoder(self.encoder(x))


# ── Loss ──────────────────────────────────────────────────────────────────────

def combined_loss(recon, original, alpha=CFG["loss_alpha"]):
    mse  = F.mse_loss(recon, original)
    ssim_val  = ssim(recon, original, data_range=1.0, size_average=False).mean()
    return alpha * mse + (1.0 - alpha) * (1.0 - ssim_val)


# ── Dataset (reads from the pre-split X_mel.npy) ─────────────────────────────

class SplitMelDataset(Dataset):
    """
    Reads a pre-split X_mel.npy (N, 3, 128, T) and y.npy.
    Filters to normal-only for training.
    Applies per-channel global normalisation to [0, 1].
    Resizes to (img_size, img_size).
    """
    def __init__(self, X_mel: np.ndarray, global_stats: dict, img_size: int = 128):
        """
        X_mel       : float32 array (N, 3, 128, T)  — already filtered to normal
        global_stats: {"ch_min": [c0,c1,c2], "ch_max": [c0,c1,c2]}
        """
        self.X         = X_mel
        self.img_size  = img_size
        self.ch_min    = np.array(global_stats["ch_min"], dtype=np.float32)
        self.ch_max    = np.array(global_stats["ch_max"], dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mel = self.X[idx].copy()                    # (3, 128, T)

        # Per-channel global normalisation
        for c in range(3):
            rng = self.ch_max[c] - self.ch_min[c] + 1e-8
            mel[c] = np.clip((mel[c] - self.ch_min[c]) / rng, 0.0, 1.0)

        t = torch.from_numpy(mel).unsqueeze(0)      # (1, 3, 128, T)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        return t.squeeze(0)                         # (3, 128, 128)


# ── Global stats ──────────────────────────────────────────────────────────────

def compute_global_stats(X_normal: np.ndarray) -> dict:
    """
    Compute per-channel min/max over the normal training subset.
    X_normal shape: (N, 3, 128, T)
    """
    ch_min, ch_max = [], []
    for c in range(X_normal.shape[1]):
        ch_min.append(float(X_normal[:, c, :, :].min()))
        ch_max.append(float(X_normal[:, c, :, :].max()))
    return {"ch_min": ch_min, "ch_max": ch_max}


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, out_dir: str):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_path  = os.path.join(out_dir, "cnn_ae_best.pth")
    best_val   = float("inf")
    patience_c = 0
    history    = {"train": [], "val": []}

    epoch_bar = tqdm(
        range(1, CFG["epochs"] + 1),
        desc      = "    Epochs",
        unit      = "ep",
        ncols     = 72,
        leave     = False,
    )

    for epoch in epoch_bar:
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = combined_loss(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                val_loss += combined_loss(recon, batch).item() * len(batch)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        epoch_bar.set_postfix({"val": f"{val_loss:.5f}"})

        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_c += 1

        if patience_c >= CFG["patience"]:
            tqdm.write(f"    Early stop at epoch {epoch}  (best val={best_val:.6f})")
            break

    # loss curve
    plt.figure(figsize=(7, 3))
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"],   label="Val")
    plt.yscale("log")
    plt.title("Training Curve — Model B CNN AE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curve_B.png"), dpi=100)
    plt.close()

    model.load_state_dict(torch.load(best_path, weights_only=True, map_location=DEVICE))
    return model


def compute_threshold(model, val_dataset, out_dir: str) -> float:
    """
    Compute 99th-percentile combined loss on the normal validation set.
    """
    model.eval()
    loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            # per-sample score
            for i in range(len(batch)):
                s = combined_loss(recon[i:i+1], batch[i:i+1]).item()
                scores.append(s)

    threshold = float(np.percentile(scores, 99))
    with open(os.path.join(out_dir, "threshold_B.txt"), "w") as f:
        f.write(f"{threshold:.8f}\n")
        f.write("# 99th-percentile combined loss on healthy val data\n")
    return threshold


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_split(split_root: str, split: str):
    folder = os.path.join(split_root, split, "model_B")
    X_mel = np.load(os.path.join(folder, "X_mel.npy"))        # (N, 3, 128, T)
    y     = np.load(os.path.join(folder, "y.npy")).astype(np.int32)
    with open(os.path.join(folder, "meta.txt")) as f:
        meta = [l.strip() for l in f]
    return X_mel, y, meta


def parse_machine_id(meta_entry: str):
    parts        = meta_entry.split("|")
    snr_machine  = parts[0]
    machine_id   = parts[1]
    machine_type = snr_machine.split("_dB_")[-1] if "_dB_" in snr_machine else snr_machine
    return machine_type, machine_id


def group_by_machine(X_mel, y, meta):
    groups = {}
    for i, m in enumerate(meta):
        mtype, mid = parse_machine_id(m)
        key = f"{mtype}_{mid}"
        if key not in groups:
            groups[key] = {"indices": [], "y": []}
        groups[key]["indices"].append(i)
        groups[key]["y"].append(y[i])
    for key in groups:
        groups[key]["y"] = np.array(groups[key]["y"], dtype=np.int32)
    return groups


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Model B — CNN Autoencoder Training Pipeline       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device    : {DEVICE}")
    print(f"  Split dir : {SPLIT_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print()

    # ── 1. Load full train split ──────────────────────────────────────────────
    print("[ 1 / 3 ]  Loading train split ...")
    X_mel_train, y_train, meta_train = load_split(SPLIT_DIR, "train")
    print(f"           {len(y_train)} samples  shape={X_mel_train.shape}  "
          f"(normal={(y_train==0).sum()}, abnormal={(y_train==1).sum()})\n")

    # ── 2. Group by machine-id ────────────────────────────────────────────────
    print("[ 2 / 3 ]  Grouping by machine identity ...")
    groups = group_by_machine(X_mel_train, y_train, meta_train)
    print(f"           {len(groups)} machine-IDs: {sorted(groups.keys())}\n")

    # ── 3. Train one CNN AE per machine-id ───────────────────────────────────
    print("[ 3 / 3 ]  Training CNN autoencoders ...\n")
    success = 0

    for machine_name, data in sorted(groups.items()):
        indices  = data["indices"]
        y_group  = data["y"]

        # Normal samples only
        normal_mask    = y_group == 0
        normal_indices = [indices[i] for i in range(len(indices)) if normal_mask[i]]

        if len(normal_indices) < 16:
            print(f"  ⚠  {machine_name}: too few normal samples ({len(normal_indices)}), skipping.")
            continue

        out_dir = os.path.join(OUTPUT_DIR, machine_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  ── {machine_name.upper()}  ({len(normal_indices)} normal samples)")

        # Slice only this machine's normal data (view, not copy)
        X_normal = X_mel_train[normal_indices]          # (n, 3, 128, T)

        # Train / val split
        n = len(X_normal)
        idx = np.arange(n)
        idx_tr, idx_val = train_test_split(idx, test_size=CFG["val_split"],
                                           random_state=CFG["seed"])

        # Global stats from train-normal only
        stats = compute_global_stats(X_normal[idx_tr])
        with open(os.path.join(out_dir, "global_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Datasets
        train_ds = SplitMelDataset(X_normal[idx_tr],  stats, CFG["img_size"])
        val_ds   = SplitMelDataset(X_normal[idx_val], stats, CFG["img_size"])

        train_loader = DataLoader(
            train_ds, batch_size=CFG["batch_size"], shuffle=True,
            num_workers=CFG["num_workers"], pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(CFG["num_workers"] > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG["batch_size"], shuffle=False,
            num_workers=0, pin_memory=False,
        )

        # Train
        model = CNNAutoencoder().to(DEVICE)
        model = train_model(model, train_loader, val_loader, out_dir)

        # ONNX export
        model.eval()
        dummy = torch.randn(1, 3, CFG["img_size"], CFG["img_size"],
                            dtype=torch.float32, device=DEVICE)
        torch.onnx.export(
            model, dummy,
            os.path.join(out_dir, "cnn_ae.onnx"),
            input_names   = ["input"],
            output_names  = ["output"],
            dynamic_axes  = {"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version = 17,
            do_constant_folding = True,
        )

        # Threshold on val set
        threshold = compute_threshold(model, val_ds, out_dir)
        print(f"     ✓  threshold_B = {threshold:.6f}\n")
        success += 1

        # Explicit cleanup between machines
        del model, train_ds, val_ds, train_loader, val_loader, X_normal
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Done. {success}/{len(groups)} models trained in {elapsed/60:.1f} min.")
    print(f"║  Deployments: {OUTPUT_DIR}")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()