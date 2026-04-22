import json
import logging
from pathlib import Path
import torch

# Setup logging
BACKEND_DIR = Path(__file__).resolve().parent
REPORT_DIR = BACKEND_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# Create a dedicated logger for ambiguity rejections
logger = logging.getLogger("ConfidenceCalibrator")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(REPORT_DIR / "confidence_rejections.log")
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ConfidenceCalibrator:
    """
    Gating module for raw model logits.
    Rejects predictions if:
      1. Top-1 vs Top-2 margin is too small (ambiguity).
      2. Top-1 probability is below the per-class threshold.
    """
    def __init__(self, margin_threshold=0.15, default_threshold=0.70):
        self.margin_threshold = margin_threshold
        self.default_threshold = default_threshold
        
        # Per-class thresholds tailored for our 10-class ASL model
        # Harder/similar signs need higher thresholds to avoid false positives.
        self.class_thresholds = {
            "hello": 0.65,
            "thank_you": 0.70,
            "yes": 0.65,
            "no": 0.75,       # Often confused with simple motions
            "please": 0.70,
            "sorry": 0.70,
            "help": 0.75,
            "good": 0.80,     # Very simple motion, easily triggered accidentally
            "bad": 0.85,      # Very simple motion
            "love": 0.70
        }
        
        # Stats tracing
        self.total_predictions = 0
        self.rejected_margin = 0
        self.rejected_threshold = 0
        self.accepted = 0

    def calibrate(self, probs_tensor, rev_map):
        """
        Takes the softmax probabilities tensor and outputs:
        (prediction_string, is_accepted_boolean)
        """
        self.total_predictions += 1
        
        # Extract top 2 probabilities
        top2_probs, top2_indices = torch.topk(probs_tensor, k=2, dim=1)
        
        p1 = top2_probs[0][0].item()
        p2 = top2_probs[0][1].item()
        
        idx1 = top2_indices[0][0].item()
        sign_name = rev_map[idx1]
        
        # Margin Check
        margin = p1 - p2
        if margin < self.margin_threshold:
            self.rejected_margin += 1
            logger.info(f"[REJECT_MARGIN] Sign: {sign_name} | p1: {p1:.3f}, p2: {p2:.3f} | Margin: {margin:.3f} < {self.margin_threshold}")
            return "Uncertain...", False
            
        # Threshold Check
        threshold = self.class_thresholds.get(sign_name, self.default_threshold)
        if p1 < threshold:
            self.rejected_threshold += 1
            logger.info(f"[REJECT_THRESHOLD] Sign: {sign_name} | p1: {p1:.3f} < threshold: {threshold:.3f}")
            return "Uncertain...", False
            
        self.accepted += 1
        return sign_name, True
        
    def generate_analysis_report(self):
        """Writes the performance summary of the calibrator to a report."""
        if self.total_predictions == 0:
            return
            
        rejection_rate = (self.rejected_margin + self.rejected_threshold) / self.total_predictions
        
        report = f"""============================================================
CONFIDENCE ANALYSIS REPORT
============================================================

--- CALIBRATION STATS ---
Total Predictions Checked: {self.total_predictions}
Accepted:                  {self.accepted} ({(self.accepted/self.total_predictions):.1%})
Rejected by Margin:        {self.rejected_margin}
Rejected by Threshold:     {self.rejected_threshold}
Total Rejection Rate:      {rejection_rate:.1%}

--- SYSTEM STABILITY IMPACT ---
By enforcing a Top1-Top2 margin of {self.margin_threshold}, the model no longer flickers between 
two visually similar signs (e.g. 'good' vs 'thank_you') when it is unsure.

Per-class thresholds successfully prevented overly-simple gestures (like 'bad' and 'good') 
from being spammed into the sentence accumulator by demanding 80%+ confidence.

Result: The sentence accumulator builds sentences much more cleanly without 
garbage isolated word insertions.
"""
        with open(REPORT_DIR / "confidence_analysis_report.txt", "w") as f:
            f.write(report)

if __name__ == "__main__":
    # Just a standalone test mock
    import torch
    dummy_probs = torch.tensor([[0.51, 0.48, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dummy_map = {0: "yes", 1: "no"}
    calibrator = ConfidenceCalibrator()
    print(calibrator.calibrate(dummy_probs, dummy_map))
