# Project A: Statistical Process Control Monitor

from collections import deque
import numpy as np


class SPCMonitor:
    def __init__(self, window_size=90):
        # window_size=90 means we remember the last 90 frames
        # At ~30 frames/second = 3 seconds of memory
        self.buffer = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
        self.is_calibrated = False

    def calibrate(self, normal_mse_scores):
        """
        Call ONCE before deployment.
        normal_mse_scores: a list of MSE values from purely normal audio
        (MSE - Mean Squared Error)
        """
        self.baseline_mean = np.mean(normal_mse_scores)
        self.baseline_std = np.std(normal_mse_scores)
        self.is_calibrated = True

        print(f"Calibrated!")
        print(f"  Normal average MSE: {self.baseline_mean:.4f}")
        print(f"  Normal spread (σ):  {self.baseline_std:.4f}")
        print(f"  Warning threshold:  {self.baseline_mean + 2*self.baseline_std:.4f}")
        print(f"  Critical threshold: {self.baseline_mean + 3*self.baseline_std:.4f}")

    def update(self, new_mse_score):
        """
        Call this for every new audio frame processed.
        Returns a string describing the current machine health.
        """
        if not self.is_calibrated:
            return "ERROR: Call calibrate() first"

        # Add the new score to our rolling memory
        self.buffer.append(new_mse_score)

        # Calculate how many standard deviations this score is from normal
        # If sigma=1 → slightly unusual
        # If sigma=2 → concerning
        # If sigma=3 → definitely wrong
        sigma = (new_mse_score - self.baseline_mean) / (self.baseline_std + 1e-8)
        # (the 1e-8 prevents division by zero if std is somehow 0)

        # Western Electric Rule: 7 rising points
        # Even if no threshold is crossed, a steady upward trend is a red flag
        if len(self.buffer) >= 7:
            last_7 = list(self.buffer)[-7:]

            # Check if each one is higher than the one before it
            all_rising = all(last_7[i] > last_7[i - 1] for i in range(1, 7))

            if all_rising and sigma < 2.0:
                # Trend detected but hasn't crossed warning threshold yet
                return f"TREND ALERT (σ={sigma:.2f}): Score rising for 7 frames — schedule inspection"

        # Standard threshold checks
        if sigma > 3.0:
            return f"CRITICAL (σ={sigma:.2f}): Immediate inspection required"
        elif sigma > 2.0:
            return f"WARNING (σ={sigma:.2f}): Anomaly developing — monitor closely"
        elif sigma > 1.0:
            return f"ADVISORY (σ={sigma:.2f}): Slightly elevated — keep watching"
        else:
            return f"NOMINAL (σ={sigma:.2f}): Machine operating normally"

    def current_sigma(self, score):
        """Just returns the sigma number if you need it for a dashboard gauge"""
        if not self.is_calibrated:
            return 0.0
        return (score - self.baseline_mean) / (self.baseline_std + 1e-8)
