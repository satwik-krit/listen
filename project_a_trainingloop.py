
import os
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CFG = {
    "input_dim"     : 8,
    "bottleneck_dim": 3,
    "learning_rate" : 1e-3,
    "epochs"        : 150,
    "batch_size"    : 64,
    "patience"      : 20,
    "seed"          : 42,
    "val_split"     : 0.2,    # fraction of train-normal used for validation
}

SPLIT_DIR  = r"C:\Users\risha\Downloads\listen\listen\split_output"
OUTPUT_DIR = r"C:\Users\risha\Downloads\listen\listen\edge_deployments"

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── Model ─────────────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split_root: str, split: str = "train"):
    """
    Load X.npy and y.npy from split_output/{split}/model_A/.
    Returns X (N,8) float32 and y (N,) int32.
    """
    folder = os.path.join(split_root, split, "model_A")
    X = np.load(os.path.join(folder, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(folder, "y.npy")).astype(np.int32)

    with open(os.path.join(folder, "meta.txt"), "r") as f:
        meta = [line.strip() for line in f.readlines()]

    return X, y, meta


def parse_machine_id(meta_entry: str):
    """
    meta format: {snr_machine}|{id_dir}|{label}|{stem}
    e.g.  -6_dB_slider|id_00|normal|00001043
    Returns (machine_type, machine_id)  e.g. ("slider", "id_00")
    """
    parts = meta_entry.split("|")
    snr_machine = parts[0]                                  # e.g. -6_dB_slider
    machine_id  = parts[1]                                  # e.g. id_00
    machine_type = snr_machine.split("_dB_")[-1] if "_dB_" in snr_machine else snr_machine
    return machine_type, machine_id


def group_by_machine(X, y, meta):
    """
    Group samples by (machine_type, machine_id).
    Returns dict: key -> {"X": ..., "y": ...}
    """
    groups = {}
    for i, m in enumerate(meta):
        mtype, mid = parse_machine_id(m)
        key = f"{mtype}_{mid}"
        if key not in groups:
            groups[key] = {"X": [], "y": []}
        groups[key]["X"].append(X[i])
        groups[key]["y"].append(y[i])

    for key in groups:
        groups[key]["X"] = np.stack(groups[key]["X"], axis=0)
        groups[key]["y"] = np.array(groups[key]["y"], dtype=np.int32)

    return groups


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, out_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    criterion = nn.MSELoss()
    best_path = os.path.join(out_dir, "edge_ae_best.pth")

    history    = []
    best_loss  = float("inf")
    patience_c = 0

    pbar = tqdm(
        range(1, CFG["epochs"] + 1),
        desc    = "    Epochs",
        unit    = "ep",
        ncols   = 68,
        leave   = False,
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] val={postfix}",
    )

    for epoch in pbar:
        # train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_loader.dataset)
        history.append(train_loss)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                val_loss += criterion(model(batch), batch).item() * len(batch)
        val_loss /= len(val_loader.dataset)

        pbar.set_postfix_str(f"{val_loss:.5f}")

        if val_loss < best_loss:
            best_loss  = val_loss
            patience_c = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_c += 1

        if patience_c >= CFG["patience"]:
            tqdm.write(f"    Early stop at epoch {epoch}  (best val={best_loss:.6f})")
            break

    # loss curve
    plt.figure(figsize=(6, 3))
    plt.plot(history, label="Train MSE")
    plt.yscale("log")
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curve.png"), dpi=100)
    plt.close()

    model.load_state_dict(torch.load(best_path, weights_only=True))
    return model


def compute_threshold(model, val_scaled, out_dir) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    tensor = torch.tensor(val_scaled, dtype=torch.float32)
    with torch.no_grad():
        per_sample_mse = criterion(model(tensor), tensor).mean(dim=1).numpy()

    threshold = float(np.percentile(per_sample_mse, 99))
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(f"{threshold:.8f}\n")
        f.write("# 99th-percentile MSE on healthy val data (all noise levels)\n")
    return threshold


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Model A — Edge Autoencoder Training Pipeline      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Split dir : {SPLIT_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print()

    # ── Load full train split ─────────────────────────────────────────────────
    print("[ 1 / 3 ]  Loading train split ...")
    X_train_all, y_train_all, meta_train = load_split(SPLIT_DIR, "train")
    print(f"           {len(X_train_all)} samples  "
          f"(normal={( y_train_all==0).sum()}, abnormal={(y_train_all==1).sum()})\n")

    # ── Group by machine-id ───────────────────────────────────────────────────
    print("[ 2 / 3 ]  Grouping by machine identity ...")
    groups = group_by_machine(X_train_all, y_train_all, meta_train)
    print(f"           {len(groups)} unique machine-IDs found: {sorted(groups.keys())}\n")

    # ── Train one AE per machine-id ───────────────────────────────────────────
    print("[ 3 / 3 ]  Training autoencoders ...\n")
    success = 0

    for machine_name, data in sorted(groups.items()):
        X_all = data["X"]
        y_all = data["y"]

        # Use NORMAL samples only for unsupervised training
        X_normal = X_all[y_all == 0]

        if len(X_normal) < 10:
            print(f"  ⚠  {machine_name}: too few normal samples ({len(X_normal)}), skipping.")
            continue

        out_dir = os.path.join(OUTPUT_DIR, machine_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  ── {machine_name.upper()}  "
              f"({len(X_normal)} normal samples for training)")

        # train / val split (normal only)
        X_tr, X_val = train_test_split(
            X_normal, test_size=CFG["val_split"], random_state=CFG["seed"]
        )

        # scale
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_sc = scaler.transform(X_val).astype(np.float32)

        with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(out_dir, "scaler_params.json"), "w") as f:
            json.dump(
                {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
                f, indent=2
            )

        # dataloaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr_sc)),
            batch_size=CFG["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_sc)),
            batch_size=CFG["batch_size"], shuffle=False
        )

        # train
        model = EdgeAutoencoder(CFG["input_dim"], CFG["bottleneck_dim"])
        model = train_model(model, train_loader, val_loader, out_dir)

        # ONNX export
        model.eval()
        dummy = torch.randn(1, CFG["input_dim"], dtype=torch.float32)
        torch.onnx.export(
            model, dummy,
            os.path.join(out_dir, "edge_ae.onnx"),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

        # threshold
        threshold = compute_threshold(model, X_val_sc, out_dir)
        print(f"     ✓  threshold = {threshold:.6f}\n")
        success += 1

    elapsed = time.perf_counter() - t0
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Done. {success}/{len(groups)} models trained in {elapsed/60:.1f} min.")
    print(f"║  Deployments: {OUTPUT_DIR}")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()