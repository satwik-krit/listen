import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import config
from feature_extraction.features import (
    process_log_mel_spectrogram,
    remove_background_noise,
    process_delta,
)
import soundfile
import joblib
import pathlib
import os


def plot_difference_mel_spectrogram(file1, file2, input_dir, remove_noise=True):
    y1, _ = librosa.load(file1, sr=config.SAMPLING_RATE)
    y2, _ = librosa.load(file2, sr=config.SAMPLING_RATE)

    master_noise = joblib.load(config.SCALER_DIRS[0] / "master_noise.pkl")

    if remove_noise:
        y1 = remove_background_noise(y1, config.SAMPLING_RATE, master_noise)
        y2 = remove_background_noise(y2, config.SAMPLING_RATE, master_noise)

    # We don't necessarily need the strict scaler just to visualize the mel
    mel1 = process_log_mel_spectrogram(y1, sr=config.SAMPLING_RATE)
    mel2 = process_log_mel_spectrogram(y2, sr=config.SAMPLING_RATE)

    plt.title(
        f"Difference Mel Spectrogram: {file1.relative_to(input_dir)} - {file2.relative_to(input_dir)}"
    )
    librosa.display.specshow(
        mel1 - mel2, x_axis="time", y_axis="mel", sr=config.SAMPLING_RATE
    )
    plt.colorbar(format="%+2.1f", location="right", shrink=0.8)
    plt.show()


def plot_mel_spectrogram(*file_paths):
    rows = len(file_paths)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4), constrained_layout=True)

    # Handle single file case where axes might not be an array
    if rows == 1:
        axes = [axes]

    for i, file_path in enumerate(file_paths):
        print(f"Plotting: {file_path.name}")
        y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)

        # We don't necessarily need the strict scaler just to visualize the mel
        norm_mel = process_log_mel_spectrogram(y, sr=config.SAMPLING_RATE)

        axes[i].set_title(f"Mel Spectrogram: {file_path.name}")
        img = librosa.display.specshow(
            norm_mel, x_axis="time", y_axis="mel", sr=config.SAMPLING_RATE, ax=axes[i]
        )

    plt.colorbar(img, format="%+2.1f", ax=axes, location="right", shrink=0.8)
    plt.show()


def plot_delta_spectrogram(file_path, order=1):
    print(f"Plotting: {file_path.name}")
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)

    norm_mel = process_log_mel_spectrogram(y, sr=config.SAMPLING_RATE)
    delta = process_delta(norm_mel, order=order)

    plt.title(f"Delta Mel Spectrogram: {file_path.name}")
    img = librosa.display.specshow(
        delta, x_axis="time", y_axis="mel", sr=config.SAMPLING_RATE
    )

    plt.colorbar(img, format="%+2.1f", location="right", shrink=0.8)
    plt.show()

def plot_zcr(file_path):
    print("Plotting zero crossing rate...")
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)

    # TODO: Figure out the correct hop length and add it to config.py
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=config.HOP_LENGTH)

    plt.figure(figsize=(10, 4))
    plt.plot(zcr[0], color='crimson', label='ZCR')
    plt.title(f'Zero Crossing Rate over Time - {file_path}')
    plt.xlabel('Frames')
    plt.ylabel('Rate')
    plt.text(0.5, 0.95, 
             f'Average ZCR - {round(np.mean(zcr), 5)}',
             transform=plt.gca().transAxes,
             horizontalalignment='center',
             verticalalignment='top')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_zcrs(*signals, colors=('r', 'g', 'b', 'y', 'black', 'orange')):
    plt.figure(figsize=(15, 5))
    plt.title("Zero Crossing Rate")

    for i, y in enumerate(signals):
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=config.FRAME_LENGTH,
            hop_length=config.HOP_LENGTH
        )

        frames = range(zcr.shape[1])

        t = librosa.frames_to_time(
            frames,
            sr=sr,
            hop_length=config.HOP_LENGTH
        )

        plt.plot(
            t,
            zcr[0],
            color=colors[i % len(colors)],
            # label=file.relative_to(pathlib.Path(os.getcwd()) / "raw_data")
        )

        # print(f"{file.relative_to(pathlib.Path(os.getcwd()) / "raw_data")}: {np.mean(zcr)}")

    plt.xlabel("Time (s)")
    plt.ylabel("ZCR")
    plt.legend()
    plt.show()

def plot_mean_and_variance_amplitude(file_path):
    print("Plotting mean and variance of amplitude...")
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)

    # Split into frames
    frames = librosa.util.frame(
        y,
        frame_length=2048,
        hop_length=config.HOP_LENGTH
    )


    # Mean amplitude and variance per frame.
    # Taking the absolute value of frames - abs(frames) makes mean_amp ~ 0
    # and both graphs look like scaled versions of each other.
    mean_amp = np.mean(frames, axis=0)
    variance = np.var(frames, axis=0)

    # Time axis - convert from frames/samples to seconds
    times = librosa.frames_to_time(
        np.arange(len(mean_amp)),
        sr=sr,
        hop_length=config.HOP_LENGTH
    )

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(times, mean_amp)
    plt.title(f"Mean Amplitude\n{file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(times, variance)
    plt.title("Variance")
    plt.xlabel("Time (s)")
    plt.ylabel("Variance")

    plt.tight_layout()
    plt.show()

def plot_pitch_contour(file_path):
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)
    y = remove_background_noise(y, sr)

    f0, voiced_flag, voiced_probe = librosa.pyin(y,
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'),
                                                 sr=sr)

    times = librosa.times_like(f0, sr=sr)

    valid_f0 = f0[~np.isnan(f0)]
    med = np.median(valid_f0) if len(valid_f0) > 0 else 0.0

    plt.figure(figsize=(10, 4))
    # Plotting as points ('.') or a line layout helps see the 'garbage' scattered dots
    plt.plot(times, f0, label='Frame Pitch (f0)', color='blue', marker='.', linestyle='None')
    plt.axhline(y=med, color='r', linestyle='--', label=f'Median Baseline ({med:.1f} Hz)')
    
    plt.title(f"Pitch Tracking ($f_0$) Over Time\n{file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_machine_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract the 2D MFCC matrix
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfccs, 
        sr=sr, 
        x_axis='time', 
        cmap='coolwarm' # 'coolwarm' or 'viridis' work wonderfully for contrast
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"MFCC Fingerprint ({n_mfcc} Coefficients)\n{file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Index")
    plt.tight_layout()
    plt.show()

def convert_to_audio(norm_mel, output):
    s = librosa.db_to_power(norm_mel)

    y_re = librosa.feature.inverse.mel_to_audio(
        s, sr=config.SAMPLING_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )

    soundfile.write(output, y_re, int(config.SAMPLING_RATE))


if __name__ == "__main__":
    from pathlib import Path

    sample1 = pathlib.Path("c:/dev/listen/raw_data/-6_dB_valve/valve/id_00/normal/00000092.wav")
    sample2 = pathlib.Path("c:/dev/listen/raw_data/6_dB_slider/slider/id_00/abnormal/00000000.wav")
    sample3 = pathlib.Path("c:/dev/listen/raw_data/6_dB_slider/slider/id_00/abnormal/00000001.wav")

    master_noise = joblib.load("c:/dev/listen/scalers/-6_dB_valve/valve/id_00/master_noise.pkl")

    y,sr = librosa.load(sample1, sr=config.SAMPLING_RATE)
    y2 = remove_background_noise(y, master_noise=master_noise, sr=config.SAMPLING_RATE)
    mel = process_log_mel_spectrogram(y2, config.SAMPLING_RATE)
    convert_to_audio(mel, "output.wav")

    plot_zcrs(y, y2)

#     import os
#     import numpy as np
#     import pandas as pd
#     import librosa
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import noisereduce as nr  # Uncomment if you are using your noise reduction module
#
#     def extract_file_features(file_path):
#         """Loads an audio file, applies optional denoising, and calculates feature means."""
#         try:
#             # Load audio (MIMII default is 16000Hz)
#             y, sr = librosa.load(file_path, sr=16000)
#
#             # --- [OPTIONAL DENOISING STEP] ---
#             # If using a noise reduction library, apply it here:
#             y = nr.reduce_noise(y=y, sr=sr)
#
#             # Extract features (getting the time-series array)
#             zcr_series = librosa.feature.zero_crossing_rate(y=y)
#             centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr)
#             rms_series = librosa.feature.rms(y=y)
#
#             # Compress the time-series down to a single average scalar for the file
#             features = {
#                 'ZCR': np.mean(zcr_series),
#                 'Spectral_Centroid': np.mean(centroid_series),
#                 'RMS_Amplitude': np.mean(rms_series)
#             }
#             return features
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
#             return None
#
#     def build_dataset(normal_dir, abnormal_dir):
#         """Loops through folders to aggregate data records."""
#         records = []
#
#         # Process Normal Files
#         print("Processing normal files...")
#         for file_name in os.listdir(normal_dir)[:50]:
#             if file_name.endswith('.wav'):
#                 res = extract_file_features(os.path.join(normal_dir, file_name))
#                 if res:
#                     res['Status'] = 'Normal'
#                     records.append(res)
#
#         # Process Abnormal Files
#         print("Processing abnormal files...")
#         for file_name in os.listdir(abnormal_dir)[:50]:
#             if file_name.endswith('.wav'):
#                 res = extract_file_features(os.path.join(abnormal_dir, file_name))
#                 if res:
#                     res['Status'] = 'Abnormal'
#                     records.append(res)
#
#         return pd.DataFrame(records)
#
# # --- CONFIGURATION: Update these paths to match your local setup ---
# # Based on your image path style:
#     NORMAL_FOLDER = r"c:/dev/listen/raw_data\-6_dB_slider\slider\id_00\normal"
#     ABNORMAL_FOLDER = r"c:/dev/listen/raw_data/-6_dB_slider\slider\id_00\abnormal"
#
# # 1. Run the aggregation pipeline
#     df = build_dataset(NORMAL_FOLDER, ABNORMAL_FOLDER)
#
# # 2. Set up the plotting environment styling
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.suptitle('Slider Asset Statistical Feature Discrimination (-6dB SNR)', fontsize=16, fontweight='bold')
#
# # Plot 1: Zero Crossing Rate
#     sns.boxplot(ax=axes[0], x='Status', y='ZCR', data=df, palette=['#2ecc71', '#e74c3c'])
#     axes[0].set_title('Zero Crossing Rate')
#     axes[0].set_ylabel('Mean ZCR Value')
#
# # Plot 2: Spectral Centroid
#     sns.boxplot(ax=axes[1], x='Status', y='Spectral_Centroid', data=df, palette=['#2ecc71', '#e74c3c'])
#     axes[1].set_title('Spectral Centroid')
#     axes[1].set_ylabel('Frequency (Hz)')
#
# # Plot 3: RMS Amplitude
#     sns.boxplot(ax=axes[2], x='Status', y='RMS_Amplitude', data=df, palette=['#2ecc71', '#e74c3c'])
#     axes[2].set_title('RMS Amplitude')
#     axes[2].set_ylabel('Energy')
#
#     plt.tight_layout()
#     plt.show()
