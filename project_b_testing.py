import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SPLIT_DIR      = r"C:\Users\risha\Downloads\listen\listen\split_output"
DEPLOYMENT_DIR = r"C:\Users\risha\Downloads\listen\listen\edge_deployments"
RESULTS_DIR    = r"C:\Users\risha\Downloads\listen\listen\evaluation_results_B"

CFG = {
    "img_size"   : 128,
    "batch_size" : 64,  
    "seed"       : 42,
}

#Hardware Activation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model──────────────────────────────────────

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

# ── Dataset ───────────────────────────────────────────────────────────────────

class SplitMelDataset(Dataset):
    def __init__(self, X_mel: np.ndarray, global_stats: dict, img_size: int = 128):
        self.X        = X_mel
        self.img_size = img_size
        # Z-score logic instead of min-max clipping
        self.ch_mean  = np.array(global_stats["ch_mean"], dtype=np.float32)
        self.ch_std   = np.array(global_stats["ch_std"], dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mel = self.X[idx].copy()
        for c in range(3):
            # Apply Z-score standardization
            mel[c] = (mel[c] - self.ch_mean[c]) / self.ch_std[c]
            
        t = torch.from_numpy(mel).unsqueeze(0)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        return t.squeeze(0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_split(split_dir: str):
    folder = os.path.join(split_dir, "test", "model_B")
    X_mel = np.load(os.path.join(folder, "X_mel.npy"))
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

@torch.no_grad()
def compute_scores(model, dataset, batch_size=64):

    num_workers = min(os.cpu_count() or 4, 8)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    scores  = []
    first_heatmap = None
    first_original = None
    first_recon = None

    model.eval()
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(DEVICE, non_blocking=True)
        
        # BLUEPRINT 3: Mixed Precision for 2x Inference Speed
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            recon = model(batch)
            # Fast Loss matching the training script (pure MSE)
            mse = F.mse_loss(recon, batch, reduction='none').mean(dim=[1, 2, 3])

        scores.extend(mse.cpu().numpy().tolist())

        # grab heatmap from very first sample for the plot
        if first_heatmap is None:
            # Cast back to float32 to ensure heatmap accuracy
            err = ((batch[0].float() - recon[0].float()) ** 2).mean(dim=0)
            first_heatmap = err.cpu().numpy()
            first_original = batch[0].float().cpu().numpy()
            first_recon    = recon[0].float().cpu().numpy()

    return np.array(scores), first_heatmap, first_original, first_recon


def plot_results(machine_name, scores, y_true, threshold,
                 fpr, tpr, auc,
                 first_heatmap, first_original, first_recon,
                 out_dir):
    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    fig.suptitle(f"Model B — {machine_name.upper()}", fontsize=13, fontweight="bold")

    # Panel 1: score distribution
    ax1 = fig.add_subplot(gs[0])
    s_norm = scores[y_true == 0]
    s_abn  = scores[y_true == 1]
    bins = np.linspace(0, np.percentile(scores, 99.5), 55)
    ax1.hist(s_norm, bins=bins, alpha=0.65, color="#4CAF50", label="Normal",   density=True)
    ax1.hist(s_abn,  bins=bins, alpha=0.65, color="#F44336", label="Abnormal", density=True)
    ax1.axvline(threshold, color="black", linewidth=1.5, linestyle="--",
                label=f"Thr ({threshold:.4f})")
    ax1.set_title("Score Distribution")
    ax1.set_xlabel("Reconstruction MSE")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ROC curve
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC={auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: example original (channel 0 = mel)
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(first_original[0], aspect="auto", origin="lower", cmap="magma")
    ax3.set_title("Example Input (mel ch)")
    ax3.axis("off")

    # Panel 4: reconstruction error heatmap
    ax4 = fig.add_subplot(gs[3])
    im = ax4.imshow(first_heatmap, aspect="auto", origin="lower", cmap="hot")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title("Reconstruction Error")
    ax4.axis("off")

    plt.savefig(os.path.join(out_dir, f"{machine_name}_eval_B.png"), dpi=120,
                bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print()
    print("       Model B — CNN AE Evaluation on Test Split          ")
    print(f"  Device     : {DEVICE}")
    print(f"  Split dir  : {SPLIT_DIR}")
    print(f"  Models dir : {DEPLOYMENT_DIR}")
    print(f"  Results    : {RESULTS_DIR}")
    print()

    # ── Load test data ────────────────────────────────────────────────────────
    print("[ 1 / 3 ]  Loading test split ")
    X_mel_test, y_test, meta_test = load_test_split(SPLIT_DIR)
    print(f"           {len(y_test)} samples  shape={X_mel_test.shape}  "
          f"(normal={(y_test==0).sum()}, abnormal={(y_test==1).sum()})\n")

    # ── Group by machine ──────────────────────────────────────────────────────
    print("[ 2 / 3 ]  Grouping by machine identity ")
    groups = group_by_machine(X_mel_test, y_test, meta_test)
    print(f"           {len(groups)} machine-IDs in test set\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[ 3 / 3 ]  Evaluating models \n")

    summary_rows = []
    header = (f"{'Machine':<20} {'N':>5} {'Nrm':>5} {'Abn':>5} "
              f"{'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(header)
    print("─" * len(header))

    for machine_name in tqdm(sorted(groups.keys()), desc="  Models", unit="model", ncols=60):
        deploy_dir = os.path.join(DEPLOYMENT_DIR, machine_name)

        required = ["cnn_ae_best.pth", "global_stats.json", "threshold_B.txt"]
        if not all(os.path.exists(os.path.join(deploy_dir, f)) for f in required):
            tqdm.write(f"  {machine_name}: missing deployment files, skipping.")
            continue

        indices = groups[machine_name]["indices"]
        y       = groups[machine_name]["y"]

        if len(np.unique(y)) < 2:
            tqdm.write(f"  {machine_name}: only one class in test set, skipping.")
            continue

        # Load artefacts
        with open(os.path.join(deploy_dir, "global_stats.json")) as f:
            stats = json.load(f)

        with open(os.path.join(deploy_dir, "threshold_B.txt")) as f:
            threshold = float(f.readline().strip())

        model = CNNAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(deploy_dir, "cnn_ae_best.pth"),
            weights_only=True, map_location=DEVICE,
        ))

        # Build test dataset for this machine
        X_machine = X_mel_test[indices]                      # (n, 3, 128, T)
        dataset   = SplitMelDataset(X_machine, stats, CFG["img_size"])

        # Score
        scores, heatmap, orig, recon_ex = compute_scores(
            model, dataset, CFG["batch_size"]
        )

        # Metrics
        y_pred = (scores > threshold).astype(np.int32)
        try:
            auc = roc_auc_score(y, scores)
            fpr, tpr, _ = roc_curve(y, scores)
        except ValueError:
            auc = float("nan")
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        cm   = confusion_matrix(y, y_pred)

        n_total  = len(y)
        n_normal = (y == 0).sum()
        n_abnorm = (y == 1).sum()

        row = (f"{machine_name:<20} {n_total:>5} {n_normal:>5} {n_abnorm:>5} "
               f"{auc:>6.3f} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        tqdm.write(row)

        # Confusion matrix file
        cm_path = os.path.join(RESULTS_DIR, f"{machine_name}_cm_B.txt")
        with open(cm_path, "w") as f:
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
            f.write(f"Confusion Matrix — {machine_name} (Model B)\n")
            f.write("Rows=Actual  Cols=Predicted  [Normal=0, Abnormal=1]\n\n")
            f.write(f"              Pred Normal  Pred Abnormal\n")
            f.write(f"True Normal   {tn:>11}  {fp:>13}\n")
            f.write(f"True Abnormal {fn:>11}  {tp:>13}\n\n")
            f.write(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
            f.write(f"\nAUC={auc:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  "
                    f"Rec={rec:.4f}  F1={f1:.4f}\n")
            f.write(f"Threshold used: {threshold:.8f}\n")

        # Plot
        plot_results(machine_name, scores, y, threshold,
                     fpr, tpr, auc, heatmap, orig, recon_ex, RESULTS_DIR)

        summary_rows.append({
            "machine"    : machine_name,
            "n_total"    : n_total,
            "n_normal"   : n_normal,
            "n_abnormal" : n_abnorm,
            "auc"        : round(auc,  4),
            "accuracy"   : round(acc,  4),
            "precision"  : round(prec, 4),
            "recall"     : round(rec,  4),
            "f1"         : round(f1,   4),
            "threshold"  : round(threshold, 8),
        })

        # Free GPU memory between machines
        del model, dataset
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── OutPut ───────────────────────────────────────────────────────────────
    print("\n" + "─" * len(header))
    if summary_rows:
        aucs = [r["auc"] for r in summary_rows if not np.isnan(r["auc"])]
        print(f"  Mean AUC      : {np.mean(aucs):.4f}")
        print(f"  Mean F1       : {np.mean([r['f1']  for r in summary_rows]):.4f}")
        print(f"  Mean Accuracy : {np.mean([r['accuracy'] for r in summary_rows]):.4f}")

        csv_path = os.path.join(RESULTS_DIR, "results_summary_B.csv")
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in summary_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

        print(f"\n  Summary CSV   : {csv_path}")
        print(f"  Eval plots    : {RESULTS_DIR}\\*_eval_B.png")
        print(f"  Confusion mats: {RESULTS_DIR}\\*_cm_B.txt")

    elapsed = time.perf_counter() - t0
    print()
    print(f"Evaluation complete. {len(summary_rows)} models evaluated in {elapsed/60:.1f} min.")
    print()


if __name__ == "__main__":
    main()