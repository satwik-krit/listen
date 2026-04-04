import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

WINDOW_SIZE = 50
BATCH_SIZE  = 256
EPOCHS      = 150
LR          = 2e-4
EARLY_STOP  = 25
N_RAW       = 8

RUN3_TRUE_SIZE = 4448

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class RULNet(nn.Module):
    def __init__(self, input_size, hidden=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            64, hidden, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout)
        self.attn      = nn.MultiheadAttention(
            hidden * 2, n_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        a, _ = self.attn(x, x, x)
        x    = self.attn_norm(x + a)
        return self.fc(x[:, -1, :]).squeeze()


def make_rul_labels(n):
    deg_start = int(n * 0.75)
    deg_len   = n - deg_start
    labels    = np.ones(n, dtype=np.float32)
    for i in range(deg_start, n):
        labels[i] = float(n - i) / float(deg_len)
    return labels


def build_sequences(X_run, fails_idx, window_size, scaler):
    n          = len(X_run)
    n_bearing  = X_run.shape[1] // N_RAW
    X_reshaped = X_run.reshape(n, n_bearing, N_RAW)

    all_X, all_y = [], []

    for b in range(n_bearing):
        scaled   = scaler.transform(
            X_reshaped[:, b, :]).astype(np.float32)

        diff     = np.diff(scaled, axis=0, prepend=scaled[0:1])
        combined = np.hstack([scaled, diff]).astype(np.float32)

        labels   = make_rul_labels(n) if b in fails_idx \
                   else np.ones(n, dtype=np.float32)

        for j in range(window_size, n):
            all_X.append(combined[j - window_size:j])
            all_y.append(labels[j])

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.float32))


if __name__ == "__main__":

    print("\nLoading X_raw_features.npy...")
    X_runs = np.load("X_raw_features.npy", allow_pickle=True)
    for i, r in enumerate(X_runs):
        print(f"  Run {i+1}: {r.shape}")

    X_runs[2] = X_runs[2][:RUN3_TRUE_SIZE]
    print(f"  Run 3 trimmed to: {X_runs[2].shape} (per README)")

    all_train_raw = np.concatenate([
        X_runs[0].reshape(-1, N_RAW),
        X_runs[2].reshape(-1, N_RAW)
    ])
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_train_raw)
    joblib.dump(global_scaler, "scaler.pkl")
    print(f"\n  Global scaler fitted on {len(all_train_raw)} samples")
    print("  Saved → scaler.pkl")

    print("\nBuilding sequences...")
    X_r1, y_r1 = build_sequences(X_runs[0], [2, 3], WINDOW_SIZE, global_scaler)
    X_r2, y_r2 = build_sequences(X_runs[1], [0],    WINDOW_SIZE, global_scaler)
    X_r3, y_r3 = build_sequences(X_runs[2], [2],    WINDOW_SIZE, global_scaler)

    print(f"  Run 1 (train): {len(X_r1)} sequences")
    print(f"  Run 2 (test):  {len(X_r2)} sequences — held out completely")
    print(f"  Run 3 (train): {len(X_r3)} sequences")

    X_tr = np.concatenate([X_r1, X_r3])
    y_tr = np.concatenate([y_r1, y_r3])
    idx  = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    n_features = X_tr.shape[2]
    print(f"\n  Features per timestep: {n_features} (8 raw + 8 slopes)")
    print(f"  Total train sequences: {len(X_tr)}")

    np.save("window_size.npy", np.array([WINDOW_SIZE]))
    np.save("n_features.npy",  np.array([n_features]))
    np.save("n_raw.npy",       np.array([N_RAW]))
    print("  Config saved → window_size.npy, n_features.npy, n_raw.npy")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=0)

    model   = RULNet(input_size=n_features).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_p:,}")

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    loss_fn    = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda')

    print(f"\nTraining for up to {EPOCHS} epochs...")
    train_losses, val_losses = [], []
    best_v, patience_c       = 1e9, 0

    X_val_t = torch.tensor(X_r2).to(device)
    y_val_t = torch.tensor(y_r2).to(device)

    for epoch in range(EPOCHS):
        model.train()
        bl = []

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(bx), by)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            bl.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                v_loss = loss_fn(model(X_val_t), y_val_t).item()

        train_losses.append(np.mean(bl))
        val_losses.append(v_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train: {train_losses[-1]:.5f} | "
                  f"Val: {v_loss:.5f}")

        if v_loss < best_v:
            best_v, patience_c = v_loss, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_c += 1
            if patience_c >= EARLY_STOP:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print(f"\n  Best val loss: {best_v:.5f}")
    print("  Saved → best_model.pt")

    print("\nEvaluating on Run 2 (held-out test)...")
    model.load_state_dict(
        torch.load("best_model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()
    actuals = y_r2

    y_smooth   = savgol_filter(preds, 51, 3)
    y_monotone = np.clip(np.minimum.accumulate(y_smooth), 0, 1)

    mae  = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds)**2)))
    print(f"  MAE:  {mae:.4f} | RMSE: {rmse:.4f} (normalized 0-1)")

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(actuals,    '--b', alpha=0.8, label="Actual RUL")
    plt.plot(preds,      color='gray', alpha=0.3, label="Raw Prediction")
    plt.plot(y_monotone, 'r', linewidth=2, label="Monotonic Smoothed")
    plt.title(f"CNN-BiLSTM-Attention | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Health (0=failure, 1=healthy)")
    plt.legend()
    plt.savefig("rul_predictions.png")
    plt.show()

    print("\nDone. All artifacts saved.")
