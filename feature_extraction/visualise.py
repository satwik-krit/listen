import librosa
import librosa.display
import matplotlib.pyplot as plt
import config
from features import process_log_mel_spectrogram


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


if __name__ == "__main__":
    from pathlib import Path

    sample_1 = config.INPUT_DIR / "abnormal/00000002.wav"
    sample_2 = config.INPUT_DIR / "normal/00000000.wav"

    if sample_1.exists() and sample_2.exists():
        plot_mel_spectrogram(sample_1, sample_2)
