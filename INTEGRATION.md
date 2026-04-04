# SonicSense AI — Integration Guide

## Plugging in your real models

The app has three clearly marked stub functions in `app.py`. Replace each one:

---

### 1. Feature extraction  →  `extract_features(audio_bytes)`

```python
import io, librosa, numpy as np

def extract_features(audio_bytes: bytes) -> dict:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    rms        = librosa.feature.rms(y=y)[0]
    zcr        = librosa.feature.zero_crossing_rate(y=y)[0]
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    median_p   = float(np.median(pitches[pitches > 0])) if (pitches > 0).any() else 0.0
    cent       = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)

    return {
        "RMS Mean":          float(np.mean(rms)),
        "ZCR":               float(np.mean(zcr)),
        "RMS Variance":      float(np.var(rms)),
        "Median Pitch":      median_p,
        "Spectral Centroid": float(np.mean(cent)),
        "Spectral Rolloff":  float(np.mean(rolloff)),
        "MFCC-1":            float(np.mean(mfccs[0])),
        "MFCC-2":            float(np.mean(mfccs[1])),
    }
```

---

### 2. Component classifier  →  `predict_component(features)`

```python
import joblib, numpy as np

rf_model = joblib.load("random_forest.pkl")   # load once at module level
COMPONENTS = ["Valve", "Fan", "Pump", "Slider"]

def predict_component(features: dict):
    X     = np.array(list(features.values())).reshape(1, -1)
    pred  = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0]
    label = COMPONENTS[pred]
    return label, dict(zip(COMPONENTS, proba.round(3)))
```

---

### 3. Anomaly detector  →  `predict_anomaly(features)`

```python
import torch, numpy as np

autoencoder = torch.load("autoencoder.pt")    # load once at module level
autoencoder.eval()
THRESHOLD = 0.5

def predict_anomaly(features: dict):
    x     = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        x_hat = autoencoder(x)
    score = float(torch.mean((x - x_hat) ** 2))
    return score > THRESHOLD, round(score, 4)
```

---

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File structure expected

```
project/
├── app.py
├── requirements.txt
├── random_forest.pkl     ← your RF model
├── autoencoder.pt        ← your AE model
└── upload_history.json   ← auto-created on first run
```
