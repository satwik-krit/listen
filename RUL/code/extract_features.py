import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
import os
import multiprocessing as mp
from functools import partial

# ── CONFIG ──────────────────────────────────────────
RUNS = [
    {"path": "nasa_data/1st_test",              "fs": 20000, "cols": 8},
    {"path": "nasa_data/2nd_test",              "fs": 20000, "cols": 4},
    {"path": "nasa_data/3rd_test/4th_test/txt", "fs": 20000, "cols": 4},
]
OUTPUT_X  = "X_features.npy"
OUTPUT_Y  = "y_rul.npy"

# Rexnord ZA-2115 bearing fault frequencies (Hz)
# Source: IMS dataset documentation
BPFO = 161.1   # Ball Pass Frequency Outer race
BPFI = 236.4   # Ball Pass Frequency Inner race
BAND_HZ = 15   # ±15 Hz around each fault frequency
FS      = 20000
N       = 20480
# ────────────────────────────────────────────────────

# Precompute FFT frequency bins once
FREQS     = fftfreq(N, d=1.0/FS)[:N//2]
BPFO_MASK = (FREQS >= BPFO - BAND_HZ) & (FREQS <= BPFO + BAND_HZ)
BPFI_MASK = (FREQS >= BPFI - BAND_HZ) & (FREQS <= BPFI + BAND_HZ)


def extract_features(signal):
    """
    Extract 14 features from one bearing channel:
    - 8 time-domain
    - 2 general frequency-domain
    - 2 fault-specific frequency-domain (BPFO, BPFI)
    - 2 sideband energy around fault frequencies
    """
    signal = signal.astype(np.float32)

    # ── Time domain ──────────────────────────────────
    mean      = np.mean(signal)
    std       = np.std(signal)
    rms       = np.sqrt(np.mean(signal ** 2))
    peak      = np.max(np.abs(signal))
    crest     = peak / (rms + 1e-10)
    kurtosis  = float(stats.kurtosis(signal))
    skewness  = float(stats.skew(signal))
    shape_fac = rms / (np.mean(np.abs(signal)) + 1e-10)

    # ── Frequency domain ─────────────────────────────
    fft_mag   = np.abs(fft(signal))[:N//2]
    fft_mean  = np.mean(fft_mag)
    fft_peak  = np.max(fft_mag)

    # ── Fault-specific band energy ───────────────────
    bpfo_energy = np.sum(fft_mag[BPFO_MASK] ** 2)
    bpfi_energy = np.sum(fft_mag[BPFI_MASK] ** 2)

    # Total spectrum energy (for normalization)
    total_energy = np.sum(fft_mag ** 2) + 1e-10

    bpfo_ratio = bpfo_energy / total_energy
    bpfi_ratio = bpfi_energy / total_energy

    return [mean, std, rms, peak, crest, kurtosis, skewness, shape_fac,
            fft_mean, fft_peak, bpfo_energy, bpfi_energy,
            bpfo_ratio, bpfi_ratio]


def process_file(args):
    """Process a single file — called by multiprocessing pool."""
    filepath, bearing_cols, file_idx, total_files = args

    try:
        df = pd.read_csv(filepath, sep="\t", header=None)
    except Exception:
        return None

    if df.shape[1] < max(bearing_cols) + 1:
        return None

    file_features = []
    for col in bearing_cols:
        signal = df[col].values
        file_features.extend(extract_features(signal))

    rul = total_files - file_idx - 1
    return file_features, rul


def process_run(run_path, declared_cols):
    all_files    = sorted(os.listdir(run_path))
    total        = len(all_files)

    # Detect bearing columns
    bearing_cols = [0, 2, 4, 6] if declared_cols >= 8 else [0, 1, 2, 3]
    print(f"  {run_path}: {total} files | "
          f"cols: {bearing_cols} | "
          f"features per file: {len(bearing_cols) * 14}")

    # Build argument list for pool
    args_list = [
        (os.path.join(run_path, f), bearing_cols, i, total)
        for i, f in enumerate(all_files)
    ]

    # Use all available CPU cores
    n_workers = mp.cpu_count()
    print(f"  Using {n_workers} CPU cores...")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_file, args_list)

    features_list = []
    rul_list      = []
    skipped       = 0

    for r in results:
        if r is None:
            skipped += 1
            continue
        feat, rul = r
        features_list.append(feat)
        rul_list.append(rul)

    if skipped > 0:
        print(f"  Skipped {skipped} files")

    return features_list, rul_list


if __name__ == "__main__":
    mp.freeze_support()   # required for Windows multiprocessing

    all_features = []
    all_rul      = []

    for run in RUNS:
        print(f"\nProcessing: {run['path']}")
        f, r = process_run(run["path"], run["cols"])
        all_features.extend(f)
        all_rul.extend(r)
        print(f"  Collected: {len(f)} samples")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_rul,      dtype=np.float32)

    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Features per sample: {X.shape[1]} "
          f"(14 per bearing × {X.shape[1]//14} bearings)")
    print(f"y range: {y.min():.0f} → {y.max():.0f}")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print(f"\nSaved {OUTPUT_X} and {OUTPUT_Y}")