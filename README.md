# LISTEN: Latent Inference of Sequential Temporal Energy Networks

LISTEN is an industrial-grade, physics-informed deep learning pipeline for acoustic anomaly detection and Remaining Useful Life (RUL) estimation. Moving beyond "one-size-fits-all" audio classification, this repository implements a Kinematic-Aware Routing Pipeline. It routes acoustic data based on the physical realities of the hardware, deploying lightweight 1D temporal tracking for rhythmic machinery and high-fidelity 2D Convolutional Autoencoders for complex, transient-load machines.

Coupled with game-theoretic Explainable AI (SHAP), this system bridges the gap between black-box neural networks and actionable mechanical engineering.

-----

## Datasets Utilized

This architecture has been tested, validated, and optimized against two primary industry-standard datasets:

  * **NASA IMS Bearing Dataset:** Used for the 1D Temporal RUL forecasting (LISTEN-Edge). Access the NASA dataset here: https://data.nasa.gov/dataset/ims-bearings
  * **MIMII Dataset:** (Malfunctioning Industrial Machine Investigation and Inspection) Used for the 2D Acoustic Vision Engine (LISTEN-Spatial) testing on complex valves, pumps, and sliders. Access the MIMII dataset here: https://zenodo.org/records/3384388

-----

## Table of Contents

1.  [The Architecture: Kinematic-Aware Routing](https://www.google.com/search?q=%23the-architecture-kinematic-aware-routing)
2.  [LISTEN-Edge: Temporal Tracker & RUL](https://www.google.com/search?q=%23listen-edge-temporal-tracker--rul)
3.  [LISTEN-Spatial: Acoustic Vision Engine](https://www.google.com/search?q=%23listen-spatial-acoustic-vision-engine)
4.  [Labeling Strategy: Piecewise Linear RUL](https://www.google.com/search?q=%23labeling-strategy-piecewise-linear-rul)
5.  [Explainable AI (SHAP) & Post-Processing](https://www.google.com/search?q=%23explainable-ai-shap--post-processing)
6.  [Installation & Dependencies](https://www.google.com/search?q=%23installation--dependencies)
7.  [References](https://www.google.com/search?q=%23references)

-----

## The Architecture: Kinematic-Aware Routing

Standard models fail because they attempt to evaluate a clicking valve and a humming exhaust fan with the exact same mathematical ruler. LISTEN routes acoustic data based on physical kinematics:

  * **LISTEN-Edge (1D Temporal Branch):** Designed for rhythmic, continuous-cycle machines. Prioritizes extreme efficiency and edge-deployment (compiled to .onnx) by tracking 1D statistical features.
  * **LISTEN-Spatial (2D Spatial Branch):** Designed for transient or variable-load machines. Treats audio as a high-fidelity image, utilizing strict structural bottlenecks to identify complex acoustic anomalies.

-----

## LISTEN-Edge: Temporal Tracker & RUL

### What is Remaining Useful Life (RUL)?

RUL is the precise time remaining between the current observation and the threshold of mechanical failure:
$$RUL(t) = T_{failure} - t$$

### Feature Engineering (NASA IMS)

Feeding raw vibration samples directly into an LSTM is computationally inefficient. We compress 20,480-sample snapshots into a 16-dimensional feature vector (8 base features + 8 temporal slope features).

**1. Time Domain Features**

  * **Root Mean Square (RMS):** The average energy content. Rises monotonically as damage dissipates energy as vibration.
  * **Kurtosis:** Measures signal "impulsiveness." A healthy bearing sits near 3.0. A bearing with a developing micro-crack will spike dramatically. It is the earliest indicator of failure.
  * **Crest Factor & Peak:** Captures the ratio of peak amplitude to RMS. Sensitive to early impulsive faults before total energy rises.

**2. Frequency Domain Features (FFT)**
We explicitly isolate the exact kinematic frequencies of the bearing. For example, the Ball Pass Frequency Outer race (BPFO):
$$BPFO = \frac{N}{2} \times \left(1 - \frac{d}{D} \cos(\alpha)\right) \times \frac{RPM}{60}$$
For the Rexnord ZA-2115 bearing at 2000 RPM, BPFO is 161.1 Hz. We extract the **BPFO Energy Ratio** (energy in a 15 Hz band around 161.1 Hz divided by total energy).

### 1D Model Architecture (CNN-BiLSTM-Attention)

1.  **1D Convolution:** Acts as a local pattern extractor across sequential timesteps.
2.  **Bidirectional LSTM (BiLSTM):** Processes sequences forward and backward, utilizing forget gates to retain long-range degradation trends while ignoring short-term noise.
3.  **Multi-Head Self-Attention:** Dynamically weighs which timesteps are most critical to the final prediction.
4.  **Regression Head:** Bounded by a Sigmoid function to enforce normalized bounds for RUL predictions.

-----

## LISTEN-Spatial: Acoustic Vision Engine

For machines in the MIMII dataset, temporal tracking is insufficient. LISTEN-Spatial utilizes a deterministic Convolutional Autoencoder (CAE).

### Data Preprocessing & Input State

  * **3-Channel Input:** Audio is pre-computed into a tensor containing the Mel-spectrogram, Delta (velocity), and Delta-Delta (acceleration) arrays.
  * **Z-Score Normalization:** Global min-max scaling is highly vulnerable to transient audio artifacts. We enforce strictly isolated, per-machine ID Z-score standardization.

### Architecture & Thresholding

  * **Deterministic Bottleneck:** We strictly avoid Variational Autoencoders (VAEs) to prevent the network from "hallucinating" or smoothing over anomalies.
  * **Loss Function:** We use Mean Squared Error (MSE) to exponentially penalize sharp, unexpected energy spikes typical of transient mechanical failures.
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$
  * **Statistical Thresholding:** Instead of arbitrary percentiles, the anomaly boundary is mathematically locked at $\mu + 3\sigma$ of the reconstruction errors on a healthy validation set, ensuring a 99.7% confidence interval against false positives.

-----

## Labeling Strategy: Piecewise Linear RUL

A healthy bearing shows zero measurable degradation for most of its life. Naive linear RUL forces a model to differentiate between identical healthy signals. We utilize a Piecewise Linear approach for model training:

1.  **Useful Life Phase:** RUL is capped at 1.0 (100%). The model is not penalized for predicting a healthy state early in the run.
2.  **Degradation Phase:** Set via domain knowledge at 75% of the run duration. At this point, microscopic defects initiate, and the RUL decays linearly to 0.0.
3.  **Bearing Unrolling:** Healthy bearings in a run receive constant RUL labels, ensuring the model learns non-failure patterns alongside failure patterns.

-----

## Explainable AI (SHAP) & Post-Processing

### Monotonicity Constraint (LISTEN-Edge)

Neural networks oscillate, but physical bearings do not self-heal. We apply Savitzky-Golay smoothing followed by a strict running minimum accumulation:

```python
y_monotone = np.minimum.accumulate(y_smooth)
```

This guarantees a physically interpretable, non-increasing degradation curve.

### DeepSHAP Pixel-Level Diagnostics (LISTEN-Spatial)

Using `shap.DeepExplainer`, we map the anomaly scores back to the physical world. The XAI layer pushes the output loss backward, assigning a contribution score to every pixel across the 3-channel spectrogram. The resulting UI generates an immediate acoustic heatmap, explicitly showing technicians which frequency band (e.g., an 8000 Hz bearing scrape) and which channel triggered the alarm.

-----

## Installation & Dependencies

To deploy this pipeline locally or edge-compile the ONNX files, clone the repository and install the dependencies. A virtual environment is highly recommended.

```bash
git clone https://github.com/your-username/LISTEN.git
cd LISTEN
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### requirements.txt

```text
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.2
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.2
tqdm>=4.64.0
onnx>=1.14.0
onnxruntime>=1.15.0
streamlit>=1.25.0
```

-----

## References

If utilizing this architecture or methodology, please refer to the foundational concepts outlined in the following literature:

1.  Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2.  Saxena, A., et al. (2008). Metrics for evaluating performance of prognostic techniques. PHM.
3.  Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
4.  Qiu, H., et al. (2006). Wavelet filter-based weak signature detection. Journal of Sound and Vibration.
5.  Purohit, H., et al. (2019). MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. arXiv.
6.  Li, X., et al. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. Reliability Engineering & System Safety.
