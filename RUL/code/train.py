import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import joblib
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

WINDOW_SIZE = 50
BATCH_SIZE  = 256
EPOCHS      = 150
LR          = 2e-4
EARLY_STOP  = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class RULNet(nn.Module):
    def __init__(self, input_size, hidden=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden * 2, n_heads, batch_first=True)
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
        x = self.attn_norm(x + a)
        return self.fc(x[:, -1, :]).squeeze()

def make_rul_labels(n):
    """
    Standard Piecewise Linear RUL:
    Healthy (1.0) until 75% of life, then linear decay to 0.0[cite: 8].
    """
    deg_start = int(n * 0.75)
    deg_len   = n - deg_start
    labels    = np.ones(n, dtype=np.float32)
    for i in range(deg_start, n):
        labels[i] = float(n - i) / float(deg_len)
    return labels

def build_unrolled_sequences(X_run, fails_idx):
    """
    Treats each bearing as an independent run to solve 'slot bias'.
    Saves a master scaler for the real-time dashboard.
    """
    n = len(X_run)
    X_reshaped = X_run.reshape(n, 4, 6)
    all_X, all_y = [], []
    
    for b in range(4):
        sc = MinMaxScaler()
        scaled = sc.fit_transform(X_reshaped[:, b, :]).astype(np.float32)
        
        if not os.path.exists("scaler.pkl"):
            joblib.dump(sc, "scaler.pkl")
            print("Master Scaler (scaler.pkl) saved.")
            
        diff = np.diff(scaled, axis=0, prepend=scaled[0:1])
        combined = np.hstack([scaled, diff]).astype(np.float32)

        if b in fails_idx:
            labels = make_rul_labels(n)
        else:
            labels = np.ones(n, dtype=np.float32)
            
        for j in range(WINDOW_SIZE, n):
            all_X.append(combined[j - WINDOW_SIZE:j])
            all_y.append(labels[j])
            
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)

if __name__ == "__main__":
    print("\nLoading X_raw_features.npy...")
    X_runs = np.load("X_raw_features.npy", allow_pickle=True)

    print("Building unrolled sequences...")
    X_r1, y_r1 = build_unrolled_sequences(X_runs[0], [2, 3])
    X_r3, y_r3 = build_unrolled_sequences(X_runs[2], [2])
    X_val_all, y_val_all = build_unrolled_sequences(X_runs[1], [0])

    X_tr = np.concatenate([X_r1, X_r3])
    y_tr = np.concatenate([y_r1, y_r3])
    idx = np.random.permutation(len(X_tr))
    
    np.save("window_size.npy", np.array([WINDOW_SIZE]))
    np.save("n_features.npy", np.array([12]))

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr[idx]), torch.tensor(y_tr[idx])), 
                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = RULNet(input_size=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda')

    print(f"Training on {len(X_tr)} unrolled sequences...")
    train_losses, val_losses = [], []
    best_v, patience = 1e9, 0

    for epoch in range(EPOCHS):
        model.train()
        bl = []
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(bx), by)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            bl.append(loss.item())

        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            v_input = torch.tensor(X_val_all[:934]).to(device)
            v_target = torch.tensor(y_val_all[:934]).to(device)
            with torch.amp.autocast('cuda'):
                v_out = model(v_input)
                v_loss = loss_fn(v_out, v_target).item()
        
        train_losses.append(np.mean(bl))
        val_losses.append(v_loss)

        if v_loss < best_v:
            best_v, patience = v_loss, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_losses[-1]:.6f} | Val: {v_loss:.6f}")
        if patience >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("\nApplying Monotonicity Constraint & Saving Artifacts...")
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val_all[:934]).to(device)).cpu().numpy()
        actuals = y_val_all[:934]

    y_smooth = savgol_filter(preds, 51, 3)
    
    y_monotonic = np.minimum.accumulate(y_smooth)
    y_final = np.clip(y_monotonic, 0, 1)

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training History"); plt.legend(); plt.savefig("training_loss.png"); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(actuals, '--b', label='Actual RUL')
    plt.plot(preds, color='gray', alpha=0.3, label='Raw Prediction')
    plt.plot(y_final, 'r', linewidth=2, label='Monotonic Smoothed RUL')
    plt.title("Final Model Submission Tracking (Run 2 Failure)")
    plt.xlabel("Sample Index"); plt.ylabel("Normalized Health")
    plt.legend(); plt.savefig("rul_predictions.png"); plt.show()

    print("Done. All artifacts ready for the dashboard.")