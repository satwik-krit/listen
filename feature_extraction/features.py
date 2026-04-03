import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
import warnings

import config


def get_normal_baseline(normal_files):
    mel_data, d_data, d2_data = [], [], []
    for f in normal_files[:20]:
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


def process_log_mel_spectrogram(y, sr, scaler_mel=None):
    mel_spectro = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel_spectro = librosa.power_to_db(mel_spectro, ref=np.max)

    if log_mel_spectro.shape[1] > config.FIXED_WIDTH:
        log_mel_spectro = log_mel_spectro[:, : config.FIXED_WIDTH]
    else:
        log_mel_spectro = np.pad(
            log_mel_spectro,
            ((0, 0), (0, config.FIXED_WIDTH - log_mel_spectro.shape[1])),
        )

    if scaler_mel is None:
        scaler_mel = MinMaxScaler(feature_range=(0, 1))
        scaler_mel.fit(np.concatenate(mel_spectro).reshape(-1, 1))

    norm_mel = scaler_mel.transform(log_mel_spectro.flatten().reshape(-1, 1))
    return norm_mel.reshape(log_mel_spectro.shape)


def process_delta(series, scaler_delta, order=1):
    d = librosa.feature.delta(series, order=order)
    # Changed from fit_transform to transform so it uses your baseline!
    d_scaled = scaler_delta.transform(d.flatten().reshape(-1, 1))
    return d_scaled.reshape(d.shape)


def process_file(file_path, scaler_mel, scaler_delta, scaler_delta2):
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)
    norm_mel = process_log_mel_spectrogram(y, sr, scaler_mel)
    delta_spectrogram = process_delta(norm_mel, scaler_delta)
    delta2_spectrogram = process_delta(norm_mel, scaler_delta2, order=2)
    audio_features = extract_audio_features(y, sr)
    return (
        np.dstack([norm_mel, delta_spectrogram, delta2_spectrogram]),
        audio_features,
    )


def extract_audio_features(y, sr):
    """
    Extracts audio features and returns them as a 1D NumPy array.
    Index mapping:
    0: ZCR
    1: Mean Amplitude
    2: Amplitude Variance
    3: Median Pitch
    4: Spectral Centroid
    5: Spectral Rolloff
    6: MFCC 1
    7: MFCC 2
    """
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    rms = librosa.feature.rms(y=y)
    mean_amp = np.mean(rms)
    amp_var = np.var(rms)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        median_pitch = np.median(valid_f0) if len(valid_f0) > 0 else 0.0

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
    mfcc1 = np.mean(mfccs[0, :])
    mfcc2 = np.mean(mfccs[1, :])

    # Return as a 1D float32 NumPy array
    return np.array(
        [zcr, mean_amp, amp_var, median_pitch, centroid, rolloff, mfcc1, mfcc2],
        dtype=np.float32,
    )


# --- Example Usage ---
# y, sr = librosa.load("your_audio.wav", sr=16000)
# features = extract_audio_features(y, sr)
# print(features)
