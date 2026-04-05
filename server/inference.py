"""
feature_extraction/inference.py  — Per-machine scaler-aware inference wrapper.

The model_manager.py now calls feature extraction directly with librosa,
but this module remains the canonical entry-point for the classify-only path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import joblib

import config
from feature_extraction.features import process_file


# Cache loaded scalers so we don't reload from disk on every call
_scaler_cache: dict[str, dict] = {}


def _load_scalers(machine_type: str, machine_id: str) -> dict:
    """
    Load mel, delta, delta2 scalers and master_noise for the given machine.
    Falls back through several search paths:
      1. scalers/6_dB_{type}/id_{id}/
      2. scalers/6_dB_valve/id_00/  (global fallback)
    """
    key = f"{machine_type}{machine_id}"
    if key in _scaler_cache:
        return _scaler_cache[key]

    # Candidate directories in priority order
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "scalers" / f"6_dB_{machine_type}" / f"id_{machine_id}",
        root / "scalers" / f"6_dB_{machine_type}" / "id_00",
        root / "scalers" / "6_dB_valve"            / "id_00",
    ]
    # Also try paths from config.SCALER_DIRS
    candidates += list(config.SCALER_DIRS)

    for d in candidates:
        d = Path(d)
        mel_path  = d / "scaler_mel.pkl"
        if mel_path.exists():
            _scaler_cache[key] = {
                "scaler_mel":   joblib.load(mel_path),
                "scaler_delta": joblib.load(d / "scaler_delta.pkl"),
                "scaler_delta2":joblib.load(d / "scaler_delta2.pkl"),
                "master_noise": joblib.load(d / "master_noise.pkl")
                                if (d / "master_noise.pkl").exists() else None,
                "source":       str(d),
            }
            return _scaler_cache[key]

    raise FileNotFoundError(
        f"No scalers found for {machine_type}/id_{machine_id}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def process_incoming_audio(
    file_path: str | Path,
    machine_type: str = "valve",
    machine_id:   str = "00",
    no_mel:       bool = False,
) -> np.ndarray:
    """
    Full feature extraction pipeline for a single WAV file.

    Parameters
    ----------
    file_path    : path to the .wav file
    machine_type : used to select the correct scaler set
    machine_id   : used to select the correct scaler set
    no_mel       : if True, return only the 8 audio features (for classification)
                   if False, return the 3-channel mel spectrogram (for GPU AE)

    Returns
    -------
    no_mel=True  → np.ndarray of shape (8,)
    no_mel=False → np.ndarray of shape (N_MELS, FIXED_WIDTH, 3)
    """
    scalers = _load_scalers(machine_type, machine_id)

    result = process_file(
        file_path,
        scaler_mel    = scalers["scaler_mel"],
        scaler_delta  = scalers["scaler_delta"],
        scaler_delta2 = scalers["scaler_delta2"],
        master_noise  = scalers["master_noise"],
        no_mel        = no_mel,
    )
    return result
