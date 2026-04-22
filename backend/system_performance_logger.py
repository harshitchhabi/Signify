import csv
from pathlib import Path
import time
import os

class SystemPerformanceLogger:
    def __init__(self):
        self.log_path = Path(__file__).resolve().parent / "reports" / "system_runtime_log.csv"
        
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV headers if file is empty
        file_exists = self.log_path.exists() and os.path.getsize(self.log_path) > 0
        
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", 
                    "fps",
                    "raw_pred", 
                    "raw_conf", 
                    "calibrated_pred", 
                    "calibrated_conf",
                    "smoothed_pred", 
                    "smoothed_conf", 
                    "corrected_pred",
                    "final_committed_word"
                ])

    def log_frame(self, fps, raw_pred, raw_conf, calibrated_pred, calibrated_conf, 
                  smoothed_pred, smoothed_conf, corrected_pred, final_committed_word):
        """
        Appends a snapshot of the pipeline's state to the CSV log.
        """
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                fps,
                raw_pred,
                raw_conf,
                calibrated_pred,
                calibrated_conf,
                smoothed_pred,
                smoothed_conf,
                corrected_pred,
                final_committed_word
            ])
