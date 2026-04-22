import time
from language_corrector import LanguageCorrector

class SentenceAccumulator:
    """
    State machine that handles accumulating stable word predictions into a sentence.
    Manages freezing, duplicate prevention, and timeouts.
    """
    def __init__(self, freeze_duration=1.0, word_timeout=1.0, auto_finalize_timeout=3.0):
        self.sentence = []
        self.freeze_duration = freeze_duration
        self.word_timeout = word_timeout
        self.auto_finalize_timeout = auto_finalize_timeout
        
        self.last_commit_time = 0.0
        self.last_hand_detected_time = time.time()
        self.last_committed_word = None
        
        self.corrector = LanguageCorrector()
        
        # We also want to ensure a word is "stable" for a few frames before committing,
        # but the TemporalSmoother already does majority voting over 10 frames.
        # We just need to make sure we don't rapid-fire commit.

    def process(self, predicted_word, current_confidence, second_best_word, second_best_confidence, hand_detected):
        """
        Called every frame.
        
        Args:
            predicted_word: The word from the TemporalSmoother (or None)
            current_confidence: The smoothed probability for the top word
            second_best_word: The raw top-2 word to act as a linguistic alternative
            second_best_confidence: The probability for the second alternative
            hand_detected: Boolean indicating if a hand is currently visible
            
        Returns:
            should_clear_buffer: Boolean indicating if the frontend should clear the frame buffer
            corrected_word: The final word selected after language correction
        """
        current_time = time.time()
        should_clear_buffer = False
        corrected_word = None
        
        if hand_detected:
            self.last_hand_detected_time = current_time
        else:
            time_since_hand = current_time - self.last_hand_detected_time
            
            # Word boundary timeout: If no hand seen for > word_timeout,
            # we reset the last_committed_word so the same word can be signed again
            if time_since_hand > self.word_timeout:
                self.last_committed_word = None
                
            # Auto-finalize timeout: For now, we'll just leave the sentence as is,
            # or maybe add a period? We'll just reset the history of last word.
            if time_since_hand > self.auto_finalize_timeout:
                pass  # Optional: logic to finalize/export the sentence here
            
            return False, None # No valid prediction to process if no hand

        # If we are in the freeze period, ignore new predictions
        if current_time - self.last_commit_time < self.freeze_duration:
            return False, None

        # If we have a valid prediction
        if predicted_word is not None and predicted_word != "Waiting...":
            # Duplicate prevention: don't commit the exact same word back-to-back 
            # unless a word_timeout has elapsed (handled above)
            if predicted_word != self.last_committed_word:
                
                # LIGHTWEIGHT LANGUAGE CORRECTION
                corrected_word, _ = self.corrector.correct(
                    predicted_word, current_confidence,
                    second_best_word, second_best_confidence,
                    self.last_committed_word
                )
                
                if corrected_word != self.last_committed_word:
                    # COMMIT WORD
                    self.sentence.append(corrected_word.upper())
                    self.last_committed_word = corrected_word
                    self.last_commit_time = current_time
                    should_clear_buffer = True

        return should_clear_buffer, corrected_word

    def get_sentence_string(self):
        """Returns the accumulated sentence as a single string."""
        return " ".join(self.sentence)
        
    def clear(self):
        """Resets the sentence."""
        self.sentence = []
        self.last_committed_word = None
        self.last_commit_time = 0.0

    def is_frozen(self):
        """Returns True if the accumulator is currently in a freeze cooldown."""
        return (time.time() - self.last_commit_time) < self.freeze_duration
