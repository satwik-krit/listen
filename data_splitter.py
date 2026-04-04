
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\risha\Downloads\listen\listen\processed_features"
OUTPUT_ROOT = r"C:\Users\risha\Downloads\listen\listen\split_output"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
STRATIFY    = True

# i7-14650HX: 8P + 8E cores = 16 cores / 24 threads.
# IO-bound disk reads benefit from more threads than physical cores.
# 14 threads keeps 2 threads free for OS + UI.
NUM_WORKERS = 14
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Discover ──────────────────────────────────────────────────────────

def discover_samples(base_dir):
    """Walk folder tree, return list of sample descriptor dicts."""
    samples = []
    snr_machine_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    for snr_machine in snr_machine_dirs:
        sm_path = os.path.join(base_dir, snr_machine)
        id_dirs = sorted([
            d for d in os.listdir(sm_path)
            if os.path.isdir(os.path.join(sm_path, d)) and d.startswith("id_")
        ])
        for id_dir in id_dirs:
            id_path = os.path.join(sm_path, id_dir)
            for label_str, label_int in [("normal", 0), ("abnormal", 1)]:
                label_path = os.path.join(id_path, label_str)
                if not os.path.isdir(label_path):
                    continue
                audio_files = sorted([
                    f for f in os.listdir(label_path)
                    if f.endswith("_audio_features.npy")
                ])
                for af in audio_files:
                    stem     = af.replace("_audio_features.npy", "")
                    mel_path = os.path.join(label_path, stem + "_mel_data.npy")
                    if not os.path.exists(mel_path):
                        continue
                    samples.append({
                        "audio_path": os.path.join(label_path, af),
                        "mel_path"  : mel_path,
                        "label"     : label_int,
                        "meta"      : f"{snr_machine}|{id_dir}|{label_str}|{stem}",
                    })
    return samples


# ── Step 2: Load (threaded) ───────────────────────────────────────────────────

def load_one(sample):
    """Load a single (audio, mel) pair. Runs in a worker thread."""
    audio = np.load(sample["audio_path"]).flatten().astype(np.float32)
    mel   = np.load(sample["mel_path"]).astype(np.float32)

    if audio.shape[0] != 8:
        raise ValueError(
            f"Expected 8 audio features, got {audio.shape[0]}: {sample['audio_path']}"
        )

    # Normalise mel to (C, 128, T) — preserve however many channels exist.
    # Saved as (128, T)        → add channel dim → (1, 128, T)
    # Saved as (C, 128, T)    → already correct  (C can be 1 or 3)
    # Saved as (128, T, C)    → transpose        → (C, 128, T)
    if mel.ndim == 2:
        mel = mel[np.newaxis, :, :]                    # (128,T) -> (1,128,T)
    elif mel.ndim == 3:
        # Distinguish (C,128,T) from (128,T,C) by checking which axis is 128
        if mel.shape[1] == 128:
            pass                                       # already (C, 128, T)
        elif mel.shape[0] == 128:
            mel = np.transpose(mel, (2, 0, 1))         # (128, T, C) -> (C, 128, T)
        # any other layout falls through unchanged

    return audio, mel, sample["label"], sample["meta"]


def load_all_parallel(samples):
    """
    Load all samples in parallel using ThreadPoolExecutor.

    Returns audio/label/meta lists immediately, but for mel we only return
    the shapes so we can pre-allocate before touching the data again.
    mel arrays are stored temporarily in results[] and consumed one-by-one
    in build_arrays — they are never all live in RAM simultaneously beyond
    the brief window while futures are in-flight.
    """
    results = [None] * len(samples)
    skipped = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_idx = {pool.submit(load_one, s): i for i, s in enumerate(samples)}

        with tqdm(
            total      = len(samples),
            desc       = "  Loading",
            unit       = "file",
            ncols      = 72,
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tqdm.write(f"    [SKIP] {samples[idx]['audio_path']}: {e}")
                    skipped += 1
                pbar.update(1)

    # Separate into lightweight lists; keep mel inside results[] for now
    # so we don't hold two copies (results + mel_list) simultaneously.
    audio_list, labels, meta_list = [], [], []
    valid_indices = []                    # positions in results[] that succeeded
    for i, r in enumerate(results):
        if r is None:
            continue
        audio, _mel, label, meta = r
        audio_list.append(audio)
        labels.append(label)
        meta_list.append(meta)
        valid_indices.append(i)

    return audio_list, labels, meta_list, valid_indices, results, skipped


# ── Step 3: Pre-allocate + fill in-place (zero extra copies) ─────────────────

def build_arrays(audio_list, labels, valid_indices, results):
    """
    Build X_scalar and X_mel without ever holding more than one extra
    mel array in memory beyond the pre-allocated destination buffer.

    Strategy:
      1. First pass (cheap): inspect mel shapes from results[] without copying.
      2. Pre-allocate X_mel as a zeroed contiguous array.
      3. Second pass: write each mel directly into its row of X_mel, then
         immediately delete the source from results[] so Python can GC it.

    Peak overhead = X_mel (final size) + 1 single mel array. Nothing else.
    """
    import gc

    N = len(valid_indices)
    X_scalar = np.stack(audio_list, axis=0)           # (N, 8) — tiny
    y        = np.array(labels, dtype=np.int32)

    # ── Pass 1: determine T_max and channel count from actual data ───────────
    T_max = 0
    C     = None   # number of channels — detected from first valid sample
    for i in valid_indices:
        mel = results[i][1]
        # After load_one normalisation mel is always 3-D: (C, 128, T)
        c_here = mel.shape[0]
        T_here = mel.shape[2]
        if C is None:
            C = c_here
        elif c_here != C:
            raise ValueError(
                f"Inconsistent channel count: expected {C}, got {c_here} "
                f"in sample index {i}"
            )
        if T_here > T_max:
            T_max = T_here

    mel_bytes_gb = N * C * 128 * T_max * 4 / 1e9
    print(f"    Pre-allocating X_mel  ({N} × {C} × 128 × {T_max}) ≈ {mel_bytes_gb:.2f} GB ...")
    X_mel = np.zeros((N, C, 128, T_max), dtype=np.float32)   # single allocation

    # ── Pass 2: write in-place, free source immediately ───────────────────────
    with tqdm(
        total      = N,
        desc       = "  Building mel",
        unit       = "arr",
        ncols      = 72,
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}]",
    ) as pbar:
        for row_idx, res_idx in enumerate(valid_indices):
            mel = results[res_idx][1]                  # grab reference
            T   = mel.shape[2] if mel.ndim == 3 else mel.shape[1]
            X_mel[row_idx, :, :, :T] = mel            # write directly into slice
            results[res_idx] = None                    # drop reference → GC eligible
            pbar.update(1)

    gc.collect()                                       # flush anything GC held back
    return X_scalar, X_mel, y


# ── Step 4: Save ──────────────────────────────────────────────────────────────

def save_split(split_name, X_scalar, X_mel, y, meta_list, output_root):
    """Write one split into model_A/ and model_B/ sub-folders."""

    def _save(folder, arrays_dict, meta):
        os.makedirs(folder, exist_ok=True)
        for fname, arr in arrays_dict.items():
            np.save(os.path.join(folder, fname), arr)
        with open(os.path.join(folder, "meta.txt"), "w") as f:
            f.write("\n".join(meta))

    base = os.path.join(output_root, split_name)

    print(f"    Saving model_A/ ...")
    _save(
        os.path.join(base, "model_A"),
        {"X.npy": X_scalar, "y.npy": y},
        meta_list,
    )

    print(f"    Saving model_B/ ...")
    _save(
        os.path.join(base, "model_B"),
        {"X_scalar.npy": X_scalar, "X_mel.npy": X_mel, "y.npy": y},
        meta_list,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MIMII Dataset Splitter  (14650HX build)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Source  : {BASE_DIR}")
    print(f"  Output  : {OUTPUT_ROOT}")
    print(f"  Split   : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test  "
          f"|  Workers: {NUM_WORKERS}  |  Seed: {RANDOM_SEED}")
    print()

    # ── 1. Discover ──────────────────────────────────────────────────────────
    print("[ 1 / 4 ]  Discovering samples ...")
    samples = discover_samples(BASE_DIR)
    if not samples:
        print("  ERROR: No samples found. Check BASE_DIR.")
        sys.exit(1)
    print(f"           Found {len(samples)} sample pairs.\n")

    # ── 2. Load ──────────────────────────────────────────────────────────────
    print(f"[ 2 / 4 ]  Loading with {NUM_WORKERS} threads ...")
    audio_list, labels, meta_list, valid_indices, results, skipped = load_all_parallel(samples)
    N = len(audio_list)
    print(f"\n           Loaded {N} samples  ({skipped} skipped)\n")

    # ── 3. Stack (memory-safe: pre-allocate + write in-place) ────────────────
    print("[ 3 / 4 ]  Building arrays (zero-copy mel stacking) ...")
    X_scalar, X_mel, y = build_arrays(audio_list, labels, valid_indices, results)
    del audio_list, results  # results[] is already cleared inside build_arrays

    print(f"\n  X_scalar : {X_scalar.shape}   dtype={X_scalar.dtype}")
    print(f"  X_mel    : {X_mel.shape}   dtype={X_mel.dtype}")
    print(f"  Normal   : {(y==0).sum()}    Abnormal : {(y==1).sum()}\n")

    # ── 4. Split + save ──────────────────────────────────────────────────────
    print("[ 4 / 4 ]  Splitting and saving ...")
    indices = np.arange(N)
    idx_train, idx_test = train_test_split(
        indices,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y if STRATIFY else None,
    )

    meta_arr = np.array(meta_list, dtype=object)

    for split_name, idx in [("train", idx_train), ("test", idx_test)]:
        n0 = (y[idx] == 0).sum()
        n1 = (y[idx] == 1).sum()
        print(f"\n  ── {split_name}/  ({len(idx)} samples | normal={n0}, abnormal={n1})")
        save_split(
            split_name  = split_name,
            X_scalar    = X_scalar[idx],
            X_mel       = X_mel[idx],
            y           = y[idx],
            meta_list   = list(meta_arr[idx]),
            output_root = OUTPUT_ROOT,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n✓  Finished in {elapsed:.1f}s")
    print(f"\nOutput structure:")
    print(f"  {OUTPUT_ROOT}/")
    print(f"    train/")
    print(f"      model_A/  →  X.npy (N,8)  y.npy  meta.txt")
    print(f"      model_B/  →  X_scalar.npy (N,8)  X_mel.npy (N,1,128,T)  y.npy  meta.txt")
    print(f"    test/")
    print(f"      model_A/  →  X.npy  y.npy  meta.txt")
    print(f"      model_B/  →  X_scalar.npy  X_mel.npy  y.npy  meta.txt")
    print()
    print("  Load example:")
    print("    X_train   = np.load(r'...\\train\\model_A\\X.npy')")
    print("    y_train   = np.load(r'...\\train\\model_A\\y.npy')")
    print("    X_mel_tr  = np.load(r'...\\train\\model_B\\X_mel.npy')")
    print()


if __name__ == "__main__":
    main()