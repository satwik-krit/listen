from pathlib import Path
import joblib
import config
from features import process_file


def process_incoming_audio(file_path):
    scaler_mel = joblib.load(config.SCALER_DIR / "scaler_mel.pkl")
    scaler_delta = joblib.load(config.SCALER_DIR / "scaler_delta.pkl")
    scaler_delta2 = joblib.load(config.SCALER_DIR / "scaler_delta2.pkl")

    feature_image = process_file(file_path, scaler_mel, scaler_delta, scaler_delta2)

    return feature_image


# if __name__ == "__main__":
#     incoming_file = Path("path/to/new/streamed_audio.wav")
#     if incoming_file.exists():
#         features = process_incoming_audio(incoming_file)
#         print(f"Extracted features shape: {features.shape}")
