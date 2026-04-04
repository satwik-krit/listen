"""
Model A — Edge Autoencoder Evaluation Script
============================================
Loads the test split produced by split_dataset.py and evaluates every
trained model from edge_deployments/.

For each machine-id it reports:
  • AUC-ROC
  • Accuracy, Precision, Recall, F1  (at the 99th-percentile threshold)
  • Confusion matrix
  • Per-sample MSE histogram (saved as PNG)

A final summary table is printed and saved to results_summary.csv.

Usage:
  python test_model_A.py

Requirements:
  pip install numpy torch scikit-learn matplotlib tqdm
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
)
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SPLIT_DIR      = r"C:\Users\risha\Downloads\listen\listen\split_output"
DEPLOYMENT_DIR = r"C:\Users\risha\Downloads\listen\listen\edge_deployments"
RESULTS_DIR    = r"C:\Users\risha\Downloads\listen\listen\evaluation_results"

CFG = {
    "input_dim"     : 8,
    "bottleneck_dim": 3,
    "seed"          : 42,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
# ─────────────────────────────────────────────────────────────────────────────


# ── Model (must match training definition exactly) ────────────────────────────

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

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_split(split_dir: str):
    folder = os.path.join(split_dir, "test", "model_A")
    X = np.load(os.path.join(folder, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(folder, "y.npy")).astype(np.int32)
    with open(os.path.join(folder, "meta.txt"), "r") as f:
        meta = [line.strip() for line in f.readlines()]
    return X, y, meta


def parse_machine_id(meta_entry: str):
    parts        = meta_entry.split("|")
    snr_machine  = parts[0]
    machine_id   = parts[1]
    machine_type = snr_machine.split("_dB_")[-1] if "_dB_" in snr_machine else snr_machine
    return machine_type, machine_id


def group_by_machine(X, y, meta):
    groups = {}
    for i, m in enumerate(meta):
        mtype, mid = parse_machine_id(m)
        key = f"{mtype}_{mid}"
        if key not in groups:
            groups[key] = {"X": [], "y": []}
        groups[key]["X"].append(X[i])
        groups[key]["y"].append(y[i])
    for key in groups:
        groups[key]["X"] = np.stack(groups[key]["X"])
        groups[key]["y"] = np.array(groups[key]["y"], dtype=np.int32)
    return groups


def compute_mse_scores(model: nn.Module, X_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor)
        mse   = criterion(recon, tensor).mean(dim=1).numpy()
    return mse


def plot_results(machine_name, mse_scores, y_true, threshold, fpr, tpr, auc, out_dir):
    """Save a 2-panel plot: MSE histogram + ROC curve."""
    fig = plt.figure(figsize=(11, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle(f"Model A — {machine_name.upper()}", fontsize=13, fontweight="bold")

    # Panel 1: MSE distribution
    ax1 = fig.add_subplot(gs[0])
    mse_normal   = mse_scores[y_true == 0]
    mse_abnormal = mse_scores[y_true == 1]

    bins = np.linspace(0, np.percentile(mse_scores, 99.5), 60)
    ax1.hist(mse_normal,   bins=bins, alpha=0.65, color="#4CAF50", label="Normal",   density=True)
    ax1.hist(mse_abnormal, bins=bins, alpha=0.65, color="#F44336", label="Abnormal", density=True)
    ax1.axvline(threshold, color="black", linewidth=1.5, linestyle="--",
                label=f"Threshold ({threshold:.4f})")
    ax1.set_xlabel("Reconstruction MSE")
    ax1.set_ylabel("Density")
    ax1.set_title("MSE Distribution")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ROC curve
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{machine_name}_eval.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Model A — Evaluation on Test Split                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Split dir  : {SPLIT_DIR}")
    print(f"  Models dir : {DEPLOYMENT_DIR}")
    print(f"  Results    : {RESULTS_DIR}")
    print()

    # ── Load test data ────────────────────────────────────────────────────────
    print("[ 1 / 3 ]  Loading test split ...")
    X_test_all, y_test_all, meta_test = load_test_split(SPLIT_DIR)
    print(f"           {len(X_test_all)} samples  "
          f"(normal={(y_test_all==0).sum()}, abnormal={(y_test_all==1).sum()})\n")

    # ── Group by machine ──────────────────────────────────────────────────────
    print("[ 2 / 3 ]  Grouping by machine identity ...")
    groups = group_by_machine(X_test_all, y_test_all, meta_test)
    print(f"           {len(groups)} machine-IDs in test set\n")

    # ── Evaluate each model ───────────────────────────────────────────────────
    print("[ 3 / 3 ]  Evaluating models ...\n")

    summary_rows = []
    header = f"{'Machine':<20} {'N':>5} {'Nrm':>5} {'Abn':>5} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(header)
    print("─" * len(header))

    for machine_name in tqdm(sorted(groups.keys()), desc="  Models", unit="model", ncols=60):
        deploy_dir = os.path.join(DEPLOYMENT_DIR, machine_name)

        # Check all required files exist
        required = ["edge_ae_best.pth", "scaler.pkl", "threshold.txt"]
        if not all(os.path.exists(os.path.join(deploy_dir, f)) for f in required):
            tqdm.write(f"  ⚠  {machine_name}: missing deployment files, skipping.")
            continue

        X = groups[machine_name]["X"]
        y = groups[machine_name]["y"]

        if len(np.unique(y)) < 2:
            tqdm.write(f"  ⚠  {machine_name}: only one class in test set, skipping AUC.")
            continue

        # Load scaler
        with open(os.path.join(deploy_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X).astype(np.float32)

        # Load threshold
        with open(os.path.join(deploy_dir, "threshold.txt"), "r") as f:
            threshold = float(f.readline().strip())

        # Load model
        model = EdgeAutoencoder(CFG["input_dim"], CFG["bottleneck_dim"])
        model.load_state_dict(torch.load(
            os.path.join(deploy_dir, "edge_ae_best.pth"),
            weights_only=True, map_location="cpu"
        ))

        # Compute MSE anomaly scores
        mse_scores = compute_mse_scores(model, X_scaled)

        # Binary predictions: MSE > threshold => anomaly (1)
        y_pred = (mse_scores > threshold).astype(np.int32)

        # Metrics
        try:
            auc = roc_auc_score(y, mse_scores)
            fpr, tpr, _ = roc_curve(y, mse_scores)
        except ValueError:
            auc = float("nan")
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        cm   = confusion_matrix(y, y_pred)

        n_total   = len(y)
        n_normal  = (y == 0).sum()
        n_abnorm  = (y == 1).sum()

        # Print row
        row = (f"{machine_name:<20} {n_total:>5} {n_normal:>5} {n_abnorm:>5} "
               f"{auc:>6.3f} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        tqdm.write(row)

        # Save confusion matrix text
        cm_path = os.path.join(RESULTS_DIR, f"{machine_name}_cm.txt")
        with open(cm_path, "w") as f:
            f.write(f"Confusion Matrix — {machine_name}\n")
            f.write("Rows=Actual  Cols=Predicted  [Normal, Abnormal]\n\n")
            f.write(f"              Pred Normal  Pred Abnormal\n")
            f.write(f"True Normal   {cm[0,0]:>11}  {cm[0,1]:>13}\n")
            f.write(f"True Abnormal {cm[1,0]:>11}  {cm[1,1]:>13}\n\n")
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
            f.write(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
            f.write(f"\nAUC={auc:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}\n")
            f.write(f"Threshold used: {threshold:.8f}\n")

        # Save plot
        plot_results(machine_name, mse_scores, y, threshold, fpr, tpr, auc, RESULTS_DIR)

        summary_rows.append({
            "machine"  : machine_name,
            "n_total"  : n_total,
            "n_normal" : n_normal,
            "n_abnormal": n_abnorm,
            "auc"      : round(auc, 4),
            "accuracy" : round(acc, 4),
            "precision": round(prec, 4),
            "recall"   : round(rec, 4),
            "f1"       : round(f1, 4),
            "threshold": round(threshold, 8),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * len(header))

    if summary_rows:
        aucs = [r["auc"] for r in summary_rows if not np.isnan(r["auc"])]
        f1s  = [r["f1"]  for r in summary_rows]
        accs = [r["accuracy"] for r in summary_rows]

        print(f"  Mean AUC      : {np.mean(aucs):.4f}")
        print(f"  Mean F1       : {np.mean(f1s):.4f}")
        print(f"  Mean Accuracy : {np.mean(accs):.4f}")

        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        print(f"\n  Summary CSV   : {csv_path}")
        print(f"  Eval plots    : {RESULTS_DIR}\\*_eval.png")
        print(f"  Confusion mats: {RESULTS_DIR}\\*_cm.txt")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Evaluation complete. {len(summary_rows)} models evaluated.")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()