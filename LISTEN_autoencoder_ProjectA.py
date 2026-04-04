"""
EdgeLinearAE — Final Hackathon Assembly (Bug-Free Release)
Stable ONNX -> TF Pipeline | Hardware-Linked C++ | INT8 Stabilized
PROJECT A - THE LOCAL DEPLOYED VERSION
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ─────────────────────────────────────────────────────────────
# STEP 1 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class EdgeLinearAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ─────────────────────────────────────────────────────────────
# STEP 2 — TRAINING & SAVING
# ─────────────────────────────────────────────────────────────
def train_autoencoder(raw_features, epochs=300, batch_size=64, lr=1e-3, save_path="edge_ae.pth", scaler_path="scaler.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler = StandardScaler()
    # Fit scaler on ALL healthy data (Assuming this is purely healthy training data)
    norm_data = scaler.fit_transform(raw_features).astype(np.float32)
    
    train_ds = TensorDataset(torch.from_numpy(norm_data))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = EdgeLinearAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval().cpu()
    
    # BUG FIX: Explicitly save artifacts for production reload
    torch.save(model.state_dict(), save_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"[Save] Model weights -> {save_path}")
    print(f"[Save] Scaler -> {scaler_path}")
    
    return model, scaler

# ─────────────────────────────────────────────────────────────
# STEP 3 — THRESHOLD COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_threshold(model, scaler, train_raw, percentile=99.0):
    model.eval()
    norm = scaler.transform(train_raw).astype(np.float32)
    x = torch.from_numpy(norm)
    with torch.no_grad():
        recon = model(x)
    per_frame_mse = ((recon - x) ** 2).mean(dim=1).numpy()
    return float(np.percentile(per_frame_mse, percentile))

# ─────────────────────────────────────────────────────────────
# STEP 4 & 5 — STABLE ONNX -> TFLITE QUANTIZATION
# ─────────────────────────────────────────────────────────────
def export_and_quantize(model, train_raw, scaler, onnx_path="edge_ae.onnx", tflite_path="edge_ae_int8.tflite"):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    import shutil

    # 1. Stable ONNX Export (Opset 13 avoids some INT8 reduce_prod bugs)
    dummy_input = torch.zeros(1, 8)
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=13, do_constant_folding=False)

    # 2. ONNX -> TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("tf_saved_model")

    # 3. Calibration Data Prep (BUG FIX: Injecting +/- 3-Sigma Edge Cases)
    np.random.shuffle(train_raw)
    base_samples = train_raw[:200]
    
    # Synthesize edge cases to prevent INT8 saturation clamping on anomalies
    feature_means = np.mean(train_raw, axis=0)
    feature_stds = np.std(train_raw, axis=0)
    high_edge = feature_means + (3 * feature_stds)
    low_edge = feature_means - (3 * feature_stds)
    
    mixed_samples = np.vstack([base_samples, high_edge, low_edge])
    rep_data = scaler.transform(mixed_samples).astype(np.float32)

    def representative_dataset():
        for frame in rep_data:
            yield [frame[np.newaxis, :]]

    # 4. TF -> TFLite Conversion
    converter = tf.lite.TFLiteConverter.from_saved_model("tf_saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    shutil.rmtree("tf_saved_model") # Clean up
    print(f"[Export] TFLite INT8 generated via ONNX bridge -> {tflite_path}")
    return tflite_path

# ─────────────────────────────────────────────────────────────
# STEP 6 — BULLETPROOF C++ GENERATION
# ─────────────────────────────────────────────────────────────
def generate_cpp_deployment(tflite_path, threshold, scaler, output_h="edge_ae_model.h", output_cpp="main.ino"):
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_scale, in_zp = interpreter.get_input_details()[0]['quantization']
    out_scale, out_zp = interpreter.get_output_details()[0]['quantization']

    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    hex_values = [f"0x{b:02x}" for b in model_bytes]
    hex_block = ",\n".join(["  " + ", ".join(hex_values[i:i+12]) for i in range(0, len(hex_values), 12)])

    # BUG FIX: Pure Header Declarations
    h_content = f"""#pragma once
extern const unsigned char g_model_data[];
extern const unsigned int  g_model_data_len;
extern const float ALARM_THRESHOLD;
extern const float FEATURE_MEAN[8];
extern const float FEATURE_SCALE[8];
"""
    with open(output_h, "w") as f:
        f.write(h_content)

    # BUG FIX: Linker Definitions, PROGMEM, and exact MSE Math
    cpp_content = f"""#include <Arduino.h>
#include <algorithm>
#include "{output_h}"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- MEMORY MACROS ---
#if defined(ESP32)
  constexpr int kTensorArenaSize = 16 * 1024; 
  #define BAUD_RATE 115200
#elif defined(ARDUINO_ARCH_RP2040)
  constexpr int kTensorArenaSize = 12 * 1024; 
  #define BAUD_RATE 115200
#elif defined(ARDUINO)
  constexpr int kTensorArenaSize = 6 * 1024;  
  #define BAUD_RATE 9600
#else
  constexpr int kTensorArenaSize = 12 * 1024;  
  #define BAUD_RATE 115200
#endif

uint8_t tensor_arena[kTensorArenaSize];

// --- QUANTIZATION CONSTANTS ---
const float INPUT_SCALE = {in_scale}f;
const int32_t INPUT_ZERO_POINT = {in_zp};
const float OUTPUT_SCALE = {out_scale}f;
const int32_t OUTPUT_ZERO_POINT = {out_zp};

// --- ARRAYS & LINKAGES (BUG FIX: alignas + PROGMEM) ---
#if defined(__AVR__)
  #include <avr/pgmspace.h>
#else
  #ifndef PROGMEM
    #define PROGMEM
  #endif
#endif

alignas(8) const unsigned char g_model_data[] PROGMEM = {{
{hex_block}
}};
const unsigned int g_model_data_len = {len(model_bytes)};
const float ALARM_THRESHOLD = {threshold:.8f}f;

const float FEATURE_MEAN[8]  = {{{', '.join(map(str, scaler.mean_))}}};
const float FEATURE_SCALE[8] = {{{', '.join(map(str, scaler.scale_))}}};

tflite::MicroMutableOpResolver<3> resolver;
const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

void normalize_and_quantize(float* raw, int8_t* out) {{
    for (int i = 0; i < 8; i++) {{
        float norm = (raw[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
        int32_t quantized = static_cast<int32_t>(round(norm / INPUT_SCALE)) + INPUT_ZERO_POINT;
        out[i] = static_cast<int8_t>(std::max(-128, std::min(127, (int)quantized)));
    }}
}}

void setup() {{
    Serial.begin(BAUD_RATE);
    while (!Serial);
    
    model_ptr = tflite::GetModel(g_model_data);
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {{
        Serial.println("Critical Error: TFLite schema mismatch!");
        while(1); 
    }}

    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddQuantize();     
    
    interpreter = new tflite::MicroInterpreter(model_ptr, resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {{
        Serial.println("Critical Error: AllocateTensors() failed! Increase kTensorArenaSize.");
        while(1);
    }}
    Serial.println("System Boot: Edge AI Pipeline Online.");
}}

float run_inference(float raw_features[8]) {{
    TfLiteTensor* input  = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    normalize_and_quantize(raw_features, input->data.int8);
    
    if (interpreter->Invoke() != kTfLiteOk) {{
        Serial.println("Error: Inference Invocation Failed.");
        return -1.0f;
    }}

    float mse = 0.0f;
    for (int i = 0; i < 8; i++) {{
        // BUG FIX: Compare Dequantized Output to Dequantized Input (Not Raw Float)
        float recon = (output->data.int8[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
        float deq_in = (input->data.int8[i] - INPUT_ZERO_POINT) * INPUT_SCALE;
        
        float diff = recon - deq_in;
        mse += diff * diff;
    }}
    return mse / 8.0f;
}}

void loop() {{
    // Simulated live DSP feed
    float live_data[8] = {{0.05, 0.001, 0.12, 220.0, 3500.0, 6000.0, -5.0, 3.0}}; 
    float mse = run_inference(live_data);
    
    if (mse >= 0.0f) {{
        Serial.print("Current MSE: "); Serial.println(mse, 6);
        if (mse > ALARM_THRESHOLD) {{
            Serial.println("CRITICAL ALARM: Anomaly Detected");
        }}
    }}
    delay(500);
}}
"""
    with open(output_cpp, "w") as f:
        f.write(cpp_content)
    print(f"[Hardware] Auto-generated C++ code deployed to {output_cpp}.")

# ─────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 2000
    synthetic_data = np.column_stack([
        np.random.normal(0.05, 0.01, N), np.random.normal(0.001, 0.0002, N),
        np.random.normal(0.12, 0.02, N), np.random.normal(220.0, 30.0, N),
        np.random.normal(3500.0, 400.0, N), np.random.normal(6000.0, 600.0, N),
        np.random.normal(-5.0, 2.0, N), np.random.normal(3.0, 1.5, N)
    ])

    model, scaler = train_autoencoder(synthetic_data)
    threshold = compute_threshold(model, scaler, synthetic_data)
    tflite_path = export_and_quantize(model, synthetic_data, scaler)
    generate_cpp_deployment(tflite_path, threshold, scaler)
