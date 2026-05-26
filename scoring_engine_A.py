import numpy as np
from collections import deque

class scoringEngine:
    def __init__(self, critical_threshold, warning_threshold=None, window_size=5):
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.warning_threshold = (
            warning_threshold if warning_threshold else (critical_threshold * 0.75)
        )
        self.current_average = 0.0

    def updates(self, new_mse):
        self.buffer.append(new_mse)
        if not self.buffer:
            return 0, 0.0

        self.current_average = sum(self.buffer) / len(self.buffer)

        if self.current_average >= self.critical_threshold:
            return 2, self.current_average  # State 2: Critical Failure

        elif self.current_average >= self.warning_threshold:
            return 1, self.current_average  # State 1: Warning / Degradation

        else:
            return 0, self.current_average  # State 0: System Nominal


class EdgeHysteresisStateMachine:
    def __init__(self, M=20):
        self.M = M
        self.mu_calib = 0.0
        self.sigma_calib = 0.0
        self.var_calib = 0.0
        self.threshold = 0.0
        self.safety_limit = 0.0
        
        # State: 2.0s (Nominal/Calibration/Monitoring) vs 0.25s (Triggered)
        self.current_window_size = 2.0  
        self.state = 0  # 0: Nominal, 1: Triggered (0.25s window)
        
        self.error_history = deque(maxlen=M)
        self.is_calibrated = False

    def calibrate(self, calibration_errors):
        """
        Expects 1,000 non-overlapping frame errors collected during calibration.
        """
        errors = np.array(calibration_errors, dtype=np.float32)
        self.mu_calib = float(np.mean(errors))
        self.sigma_calib = float(np.std(errors))
        self.var_calib = float(np.var(errors))
        
        # Lock threshold: mu_calibration + 3 * sigma_calibration
        self.threshold = self.mu_calib + 3.0 * self.sigma_calib
        # Lock exit safety limit: epsilon = 0.5 * var_calibration
        self.safety_limit = 0.5 * self.var_calib
        self.is_calibrated = True
        
        print(f"[Calibration Engine] Locked mu={self.mu_calib:.6f}, sigma={self.sigma_calib:.6f}, var={self.var_calib:.6f}")
        print(f"[Calibration Engine] Threshold locked at {self.threshold:.6f}")
        print(f"[Calibration Engine] Hysteresis safety limit locked at {self.safety_limit:.6f}")

    def update(self, frame_error):
        """
        Updates the state machine with the new reconstruction frame error.
        Returns: (state, current_window_size, threshold)
        """
        if not self.is_calibrated:
            # If not calibrated, stay nominal and collect stats
            return 0, 2.0, 0.0

        self.error_history.append(frame_error)

        if self.state == 0:
            # Nominal state (2.0s window)
            # Enter Trigger State: Transition from 2.0s to 0.25s when error breaches Threshold
            if frame_error > self.threshold:
                self.state = 1
                self.current_window_size = 0.25
                print(f"[Hysteresis SM] BREACH! Error {frame_error:.6f} > Threshold {self.threshold:.6f}. Switching to 0.25s window.")
        else:
            # Triggered state (0.25s window)
            # Exit Trigger State: Transition back to 2.0s only when rolling variance of reconstruction error
            # over the trailing M=20 frames drops below the grounded safety limit
            if len(self.error_history) >= self.M:
                rolling_var = np.var(list(self.error_history))
                if rolling_var < self.safety_limit:
                    self.state = 0
                    self.current_window_size = 2.0
                    self.error_history.clear()  # Clear history after transition
                    print(f"[Hysteresis SM] RECOVERY! Rolling Var {rolling_var:.6f} < Safety Limit {self.safety_limit:.6f}. Transition back to 2.0s window.")

        return self.state, self.current_window_size, self.threshold
