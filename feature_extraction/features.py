import numpy as np
import noisereduce
import librosa
from sklearn.preprocessing import MinMaxScaler
import warnings

import config as config


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
    d_scaled = scaler_delta.transform(d.flatten().reshape(-1, 1))
    return d_scaled.reshape(d.shape)


def process_file(
    file_path, scaler_mel, scaler_delta, scaler_delta2, master_noise, no_mel=False
):
    y, sr = librosa.load(file_path, sr=config.SAMPLING_RATE)
    y = remove_background_noise(y, sr, master_noise)
    audio_features = extract_audio_features(y, sr)
    if not no_mel:
        norm_mel = process_log_mel_spectrogram(y, sr, scaler_mel)
        delta_spectrogram = process_delta(norm_mel, scaler_delta)
        delta2_spectrogram = process_delta(norm_mel, scaler_delta2, order=2)
        return (
            np.dstack([norm_mel, delta_spectrogram, delta2_spectrogram]),
            audio_features,
        )
    return audio_features


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


def remove_background_noise(y, sr, master_noise=None):

    reduced_noise_y = noisereduce.reduce_noise(
        y=y,
        sr=sr,
        y_noise=master_noise,
        prop_decrease=1.0,
    )

    return reduced_noise_y


def create_master_mask(normal_files):
    noise_chunks = list()

    for file in normal_files[:20]:
        y, _ = librosa.load(file, sr=config.SAMPLING_RATE, duration=0.5)
        noise_chunks.append(y)

    return np.concatenate(noise_chunks)


class Preprocessor1D:
    def __init__(self, sr=16000):
        self.sr = sr
        self.prev_kurtosis = 0.0
        self.prev_rms = 0.0
        self.prev_peak = 0.0
        self.warmup_counter = 0
        self.current_window_size = 2.0  # seconds

    def set_window_size(self, size):
        if size != self.current_window_size:
            self.current_window_size = size
            self.warmup_counter = 4  # Start the 4-frame warm-up mask

    def extract_frame_features(self, y):
        # 1. 8 static statistical features
        rms = np.sqrt(np.mean(y ** 2))
        mean_abs = np.mean(np.abs(y))
        peak = np.max(np.abs(y))
        
        # Mean & Variance
        mean_y = np.mean(y)
        var_y = np.var(y)
        std_y = np.std(y)
        
        # Kurtosis
        kurtosis = np.mean((y - mean_y) ** 4) / (std_y ** 4 + 1e-9)
        
        # Skewness
        skewness = np.mean((y - mean_y) ** 3) / (std_y ** 3 + 1e-9)
        
        # Factors
        crest_factor = peak / (rms + 1e-9)
        peak_factor = peak / (mean_abs + 1e-9)
        shape_factor = rms / (mean_abs + 1e-9)
        
        # 2. 3 temporal delta features
        if self.warmup_counter > 0:
            # Transition Buffer Discard: discard/zero-out the temporal delta features
            delta_kurtosis = 0.0
            delta_rms = 0.0
            delta_peak = 0.0
            self.warmup_counter -= 1
        else:
            delta_kurtosis = kurtosis - self.prev_kurtosis
            delta_rms = rms - self.prev_rms
            delta_peak = peak - self.prev_peak
            
        # Update prev values
        self.prev_kurtosis = kurtosis
        self.prev_rms = rms
        self.prev_peak = peak
        
        features = np.array([
            rms, kurtosis, peak_factor, crest_factor, mean_y, var_y, skewness, shape_factor,
            delta_kurtosis, delta_rms, delta_peak
        ], dtype=np.float32)
        
        return features

    def process_signal(self, y, mode="production"):
        """
        Processes a full 1D signal using windowing.
        Modes:
          - 'calibration': Force 2.0s window, stride 2.0s (0% overlap), extract 1,000 frames.
          - 'production': 2.0s window, stride 1.0s (50% overlap).
        """
        window_len = int(self.current_window_size * self.sr)
        if mode == "calibration":
            window_len = int(2.0 * self.sr)
            stride = window_len
            num_frames = 1000
        else:
            stride = int(self.current_window_size * self.sr / 2) if self.current_window_size == 2.0 else int(self.current_window_size * self.sr)
            # Calculate frames to fit full signal length
            num_frames = (len(y) - window_len) // stride + 1
            if num_frames <= 0:
                num_frames = 1

        features_list = []
        for i in range(num_frames):
            start_idx = i * stride
            end_idx = start_idx + window_len
            if end_idx > len(y):
                frame_y = y[start_idx:]
                if len(frame_y) < window_len:
                    frame_y = np.pad(frame_y, (0, window_len - len(frame_y)))
            else:
                frame_y = y[start_idx:end_idx]
            
            feat = self.extract_frame_features(frame_y)
            features_list.append(feat)

        return np.array(features_list, dtype=np.float32)
