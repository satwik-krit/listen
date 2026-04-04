import librosa
import librosa.display
import matplotlib.pyplot as plt
import config
from features import process_log_mel_spectrogram, remove_background_noise
import soundfile
import joblib


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
    plt.colorbar(format="+%.02f", location="right", shrink=0.8)
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

    plt.colorbar(img, format="+%.02f", ax=axes, location="right", shrink=0.8)
    plt.show()


def convert_to_audio(norm_mel, output):
    s = librosa.db_to_power(norm_mel)

    y_re = librosa.feature.inverse.mel_to_audio(
        s, sr=config.SAMPLING_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )

    soundfile.write(output, y_re, int(config.SAMPLING_RATE))


if __name__ == "__main__":
    from pathlib import Path

    sample_1 = config.INPUT_DIRS[0] / "abnormal/00000002.wav"
    sample_2 = config.INPUT_DIRS[0] / "normal/00000000.wav"

    plot_difference_mel_spectrogram(sample_2, sample_1, config.INPUT_DIRS[0])
