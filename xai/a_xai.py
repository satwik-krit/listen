import time

import numpy as np
import onnxruntime as ort

from scoring_engine import scoringEngine

# INITIALIZATION (Run this once when the edge device boots)


session = ort.InferenceSession("edge_deployments/valve_id_00/edge_ae.onnx")
input_name = session.get_inputs()[0].name
print(input_name)

# Initialize the SPC Scoring Engine
scoring_engine = scoringEngine(
    critical_threshold=0.08, warning_threshold=0.05, window_size=10
)

# ---------------------------------------------------------
# 2. THE EDGE INFERENCE LOOP (Runs continuously)
# ---------------------------------------------------------


def run_edge_inference():
    print("Starting Edge Monitor...")

    while True:
        # Step A: Ingest data (Replace with your actual edge data source)
        # Assuming input shape [1, num_features]
        incoming_data = get_sensor_data()
        incoming_data = np.expand_dims(incoming_data, axis=0).astype(np.float32)

        # Step B: Run Project AXAI (The Autoencoder)
        reconstructed_data = session.run(None, {input_name: incoming_data})[0]

        # Step C: Calculate the Errors
        feature_errors = np.square(incoming_data - reconstructed_data)
        current_mse = np.mean(feature_errors)

        # Step D: Pass the raw MSE to the Scoring Engine
        state, smoothed_mse = scoring_engine.updates(current_mse)

        # Step E: Act on the System State (This is where XAI is triggered!)
        if state == 2:
            print(
                f"[CRITICAL] Smoothed MSE: {smoothed_mse:.4f} crossed critical threshold!"
            )
            trigger_xai_explanation(incoming_data, reconstructed_data)

        elif state == 1:
            print(f"[WARNING] Smoothed MSE: {smoothed_mse:.4f} is degrading.")

        else:
            # State 0: Nominal
            pass

        time.sleep(1)  # Or whatever your polling rate is


# ---------------------------------------------------------
# 3. YOUR XAI EXPLANATION HELPER
# ---------------------------------------------------------
def trigger_xai_explanation(incoming_data, reconstructed_data):
    """
    When the Scoring Engine alarms, this function explains WHY.
    """
    from xai.a_pferd import explain_reconstruction, print_explanation

    print_explanation(explain_reconstruction(incoming_data, reconstructed_data))
    # # Find which feature had the worst reconstruction error
    # worst_feature_idx = np.argmax(feature_errors)
    # max_error_val = feature_errors[0][worst_feature_idx]

    # print(
    #     f"--> XAI DIAGNOSIS: The anomaly is being driven primarily by Feature Index [{worst_feature_idx}] with an error variance of {max_error_val:.4f}"
    # )
    # Here you could send this data to an MQTT topic, a dashboard, or a log file.


# Dummy function for the example
def get_sensor_data():
    from feature_extraction.features import (
        extract_audio_features,
        remove_background_noise,
        process_log_mel_spectrogram,
    )
    import librosa
    import config

    y, sr = librosa.load(config.INPUT_DIRS[0] / "abnormal" / "00000000.wav")
    features = extract_audio_features(y, sr)
    return features


if __name__ == "__main__":
    run_edge_inference()
