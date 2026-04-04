from collections import deque

class scoringEngine:
    def __innit__(self,critical_threshold,warning_threshold = None, window_size = 5):
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.buffer = deque(maxlen = window_size)
        self.warning_threshold = warning_threshold if warning_threshold else (critical_threshold*0.75)
        self.current_average = 0.0
    
    def updates(self, new_mse):
        self.buffer.append(new_mse)
        if not self.buffer:
            return 0,0.0

        self.current_average = sum(self.buffer) / len(self.buffer)
        
        if self.current_average >= self.critical_threshold:
            return 2, self.current_average  # State 2: Critical Failure
            
        elif self.current_average >= self.warning_threshold:
            return 1, self.current_average  # State 1: Warning / Degradation
            
        else:
            return 0, self.current_average  # State 0: System Nominal


