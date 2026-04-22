import os
from pathlib import Path

class LanguageCorrector:
    def __init__(self):
        # Build simple bigram probability dictionary
        # Weights represent probability multipliers
        self.bigrams = {
            ("HELLO", "PLEASE"): 1.5,
            ("YES", "PLEASE"): 1.5,
            ("NO", "THANK_YOU"): 1.5,
            ("PLEASE", "HELP"): 1.5,
            ("GOOD", "LOVE"): 1.2,
            ("SORRY", "BAD"): 1.2
        }
        self.corrections_made = 0
        self.log_file = Path(__file__).resolve().parent / "reports" / "language_correction_analysis.txt"
        
        # Ensure reports dir exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log
        with open(self.log_file, "w") as f:
            f.write("=== LANGUAGE CORRECTION ANALYSIS ===\n")
            f.write("Parameters:\n- Max Conf Diff: 0.15\n- Strong Prediction Threshold: 0.85\n\n")

    def correct(self, top_class, top_conf, second_class, second_conf, prev_word):
        """
        Adjusts candidate probabilities based on the previous committed word.
        Returns the chosen word and its original confidence.
        """
        if not top_class or not prev_word or not second_class:
            return top_class, top_conf

        # 4. Keep correction subtle
        if top_conf > 0.85:
            return top_class, top_conf
            
        if (top_conf - second_conf) >= 0.15:
            return top_class, top_conf

        prev_upper = prev_word.upper()
        top_upper = top_class.upper()
        sec_upper = second_class.upper()

        # 2. Adjust candidate probabilities: adjusted_score = model_confidence * bigram_weight
        w_top = self.bigrams.get((prev_upper, top_upper), 1.0)
        w_sec = self.bigrams.get((prev_upper, sec_upper), 1.0)
        
        adj_top = top_conf * w_top
        adj_sec = second_conf * w_sec
        
        # 3. If a second-best class forms a stronger bigram than the top predicted class
        if adj_sec > adj_top:
            self.corrections_made += 1
            self.log_correction(prev_upper, top_upper, sec_upper, top_conf, second_conf, adj_top, adj_sec)
            return second_class, second_conf
            
        return top_class, top_conf

    def log_correction(self, prev, orig, new, conf_orig, conf_new, adj_orig, adj_new):
        with open(self.log_file, "a") as f:
            f.write(f"CORRECTION #{self.corrections_made}:\n")
            f.write(f"  Context:   [... {prev}] + ?\n")
            f.write(f"  Overrides: '{orig}' (Conf: {conf_orig:.2f}, Adj: {adj_orig:.2f}) -> '{new}' (Conf: {conf_new:.2f}, Adj: {adj_new:.2f})\n")
            f.write(f"  Result:    [... {prev} {new}]\n\n")
