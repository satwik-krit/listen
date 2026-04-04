# Project A: PER FEATURE ERROR RECONSTRUCTION DICTIONARY

import numpy as np

# List must be in the same order extract_audio_features() returns them.
FEATURE_NAMES = [
    "ZCR",
    "RMS Mean",
    "RMS Variance",
    "Median Pitch",
    "Spectral Centroid",
    "Rolloff",
    "MFCC-1",
    "MFCC-2",
]

FAULT_DICTIONARY = {
    "ZCR": "Bearing friction or surface roughness detected",
    "Median Pitch": "Shaft speed deviation — possible rotational imbalance",
    "RMS Mean": "Amplitude surge — mechanical looseness or impact",
    "RMS Variance": "Unstable vibration — intermittent fault developing",
    "Spectral Centroid": "Frequency content shifted — wear toward higher bands",
    "Rolloff": "High-frequency spike — metal contact or cracking",
    "MFCC-1": "Tonal character changed — resonance frequency shifted",
    "MFCC-2": "Spectral shape distorted — multiple fault signatures",
}


def explain_reconstruction(input_features, reconstructed_features, top_n=3):
    """
    input_features:        numpy array of 8 numbers from live audio
    reconstructed_features: numpy array of 8 numbers the autoencoder guessed
    top_n:                 how many to report (default: top 3)
    """

    errors = np.abs(input_features - reconstructed_features)

    paired = list(zip(FEATURE_NAMES, errors, input_features, reconstructed_features))

    ranked = sorted(paired, key=lambda x: x[1], reverse=True)

    results = []
    for name, error, actual, expected in ranked[:top_n]:
        results.append(
            {
                "rank": len(results) + 1,
                "feature": name,
                "error": round(float(error), 4),
                "actual": round(float(actual), 4),
                "expected": round(float(expected), 4),
                "diagnosis": FAULT_DICTIONARY[name],
            }
        )

    return results


def print_explanation(results):
    print("\n=== FAULT DIAGNOSIS ===")
    for item in results:
        print(f"\n#{item['rank']} — {item['feature']}")
        print(f"  Actual:   {item['actual']}")
        print(f"  Expected: {item['expected']}")
        print(f"  Error:    {item['error']}")
        print(f"  Verdict:  {item['diagnosis']}")
