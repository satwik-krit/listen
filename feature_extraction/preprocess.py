from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
import time
import json
import xml

from tqdm import tqdm

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


INPUT_DIR = Path("raw_data/6_dB_valve/valve/id_00/").resolve()
OUTPUT_DIR = Path("processed_features/6_dB_valve/valve/id_00/").resolve()
SAMPLING_RATE = 16e3
N_MELS = 128
FIXED_WIDTH = 313  # 10 seconds @ 16k Hz with 512 hop size
HOP_LENGTH = 512
N_FFT = 2048

TOTAL_NORMAL = 0
TOTAL_ABNORMAL = 0

#     filter_banks = librosa.filters.mel(n_fft=2048, sr = 16e3, n_mels=90)
#
# # Mel Filter Banks
#     plt.figure(figsize=(8, 4))
#     plt.title("Mel Filter Banks")
#     librosa.display.specshow(filter_banks,
#                              sr=sr,
#                              x_axis="linear")
#     plt.colorbar(format="%+.2f")
#     plt.show()


def get_normal_baseline(normal_files):
    # Collect a small chunk of data from normal files to "prime" the scalers
    mel_data, d_data, d2_data = [], [], []
    for f in normal_files[:20]:  # 20 files is good enough for generating a baseline
        y, sr = librosa.load(f, sr=None)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
        mel_data.append(m.flatten())
        d_data.append(librosa.feature.delta(m).flatten())
        d2_data.append(librosa.feature.delta(m, order=2).flatten())

    scaler_mel = MinMaxScaler(feature_range=(0, 1))
    scaler_delta = MinMaxScaler(feature_range=(0, 1))
    scaler_delta2 = MinMaxScaler(feature_range=(0, 1))

    scaler_mel.fit(np.concatenate(mel_data).reshape(-1, 1))
    scaler_delta.fit(np.concatenate(d_data).reshape(-1, 1))
    scaler_delta2.fit(np.concatenate(d2_data).reshape(-1, 1))

    return scaler_mel, scaler_delta, scaler_delta2


def process_feature_vectored():
    # Feature vector example for one audio file:
    # feature_vector = [
    #     np.mean(rms),           # Amplitude Mean
    #     np.std(rms),            # Amplitude Variance (High for Valve, Low for Fan)
    #     np.mean(voiced_pitches),# Pitch Mean
    #     np.mean(zcr),           # Zero-Crossing Rate (High for Slider)
    #     np.mean(centroid)       # Spectral Centroid (High for Valve)
    #
    # ]
    pass


def process_log_mel_spectrogram(y, sr, scaler_mel=None):
    mel_spectro = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    log_mel_spectro = librosa.power_to_db(mel_spectro, ref=np.max)

    # Standardize Shape (313 frames)
    if log_mel_spectro.shape[1] > FIXED_WIDTH:
        log_mel_spectro = log_mel_spectro[:, :FIXED_WIDTH]
    else:
        log_mel_spectro = np.pad(
            log_mel_spectro, ((0, 0), (0, FIXED_WIDTH - log_mel_spectro.shape[1]))
        )

    if scaler_mel is None:
        scaler_mel = MinMaxScaler(feature_range=(0, 1))
        scaler_mel.fit(np.concatenate(mel_spectro).reshape(-1, 1))
    norm_mel = scaler_mel.transform(log_mel_spectro.flatten().reshape(-1, 1))
    norm_mel = norm_mel.reshape(log_mel_spectro.shape)
    return norm_mel


def process_delta(series, scaler_delta, order=1):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    d = librosa.feature.delta(series, order=order)
    d_scaled = scaler_delta.fit_transform(d.flatten().reshape(-1, 1))
    d_scaled = d_scaled.reshape(d.shape)
    return d_scaled


def process_file(file_path, scaler_mel=None, scaler_delta=None, scaler_delta2=None):
    y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    # if scaler_mel is None:
    #     scaler_mel = MinMaxScaler(feature_range=(0, 1))
    #     scaler_mel.fit(y.reshape(-1, 1))
    # if scaler_delta is None:
    #     scaler_delta = MinMaxScaler(feature_range=(0, 1))
    #     scaler_delta.fit(y.reshape(-1, 1))
    # if scaler_delta2 is None:
    #     scaler_delta2 = MinMaxScaler(feature_range=(0, 1))
    #     scaler_delta2.fit(y.reshape(-1, 1))
    norm_mel = process_log_mel_spectrogram(y, sr, scaler_mel)
    delta_spectrogram = process_delta(norm_mel, scaler_delta)
    delta2_spectrogram = process_delta(norm_mel, scaler_delta2, order=2)

    mel_image = np.dstack([norm_mel, delta_spectrogram, delta2_spectrogram])

    return mel_image


def process_file_and_save(file_path, scaler_mel, scaler_delta, scaler_delta2):
    """
    return
        0 -> NORMAL
        1 -> ABNORMAL
        -1 -> ERROR
    """
    try:
        data = process_file(file_path, scaler_mel, scaler_delta, scaler_delta2)
        relative_path = file_path.relative_to(INPUT_DIR)

        target_path = OUTPUT_DIR / relative_path.with_suffix(".npy")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(target_path, data.astype(np.float32))

        # Extract normal/abnormal word from file_path
        label = str(relative_path).split("\\")[0]

        if label == "abnormal":
            return 1
        return 0

    except Exception as e:
        print(f"error on {file_path.name}: {e}")
        return -1


def plot_mel_spectrogram(*file_paths):
    rows = len(file_paths)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4), constrained_layout=True)

    for i, file_path in enumerate(file_paths):
        title = file_path.relative_to(INPUT_DIR)
        print(f"Processing: {title}")
        y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
        norm_mel = process_log_mel_spectrogram(y, sr=SAMPLING_RATE)
        axes[i].set_title(f"Mel Spectrogram: {title}")
        img = librosa.display.specshow(
            norm_mel, x_axis="time", y_axis="mel", sr=SAMPLING_RATE, ax=axes[i]
        )

    plt.colorbar(img, format="+%.02f", ax=axes, location="right", shrink=0.8)

    plt.show()


def batch_process():
    print("STARTING...")
    start = time.perf_counter()

    all_files = list(INPUT_DIR.rglob("*.wav"))

    # We know the last 20 files are probably normal files, so we can pass
    # them without verifying it.
    s_mel, s_delta, s_delta2 = get_normal_baseline(all_files[-1:-20:-1])

    worker_func = partial(
        process_file_and_save,
        scaler_mel=s_mel,
        scaler_delta=s_delta,
        scaler_delta2=s_delta2,
    )

    print(f"Detected {len(all_files)} files for processing.")

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(worker_func, all_files, chunksize=20),
                total=len(all_files),
            )
        )

    end = time.perf_counter()
    print(f"FINISHED in {round(end - start, 2)} seconds.")
    t_normal = results.count(0)
    t_abnormal = results.count(1)
    t_error = results.count(-1)
    print("Processed")
    print(f"\t{t_normal} NORMAL files")
    print(f"\t{t_abnormal} ABNORMAL files")
    print(f"\t{t_error} ERRORS")


if __name__ == "__main__":
    # plot_mel_spectrogram(
    #     INPUT_DIR / "abnormal/00000002.wav", INPUT_DIR / "normal/00000000.wav"
    # )
    batch_process()
