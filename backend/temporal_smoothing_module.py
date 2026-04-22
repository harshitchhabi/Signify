import collections
import numpy as np

class FrameBuffer:
    """
    Maintains a sliding window of hand landmarks.
    """
    def __init__(self, window_size=30, feature_size=63):
        self.window_size = window_size
        self.feature_size = feature_size
        self.buffer = collections.deque(maxlen=window_size)
        
    def add_frame(self, landmarks):
        """
        Adds normalized landmarks (array of size 63) to the buffer.
        """
        if len(landmarks) != self.feature_size:
            raise ValueError(f"Expected landmarks of size {self.feature_size}, got {len(landmarks)}")
        self.buffer.append(landmarks)
        
    def is_full(self):
        return len(self.buffer) == self.window_size
        
    def get_flattened_buffer(self):
        """
        Returns a flattened numpy array of shape (window_size * feature_size,)
        Used for feeding the MLP classifier.
        """
        if not self.is_full():
            return None
        return np.array(self.buffer).flatten()
        
    def clear(self):
        self.buffer.clear()

class TemporalSmoother:
    """
    Smooths predictions over time using majority voting and confidence thresholding.
    """
    def __init__(self, history_size=10, confidence_threshold=0.7):
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        
        # Store tuples of (prediction_string, probability)
        self.history = collections.deque(maxlen=history_size)
        
    def add_prediction(self, sign_name, probability):
        """
        Records the latest prediction and probability.
        """
        self.history.append((sign_name, probability))
        
    def get_smoothed_prediction(self):
        """
        Returns the (sign_name, average_probability) if the most frequent
        prediction passes the confidence threshold. Otherwise returns (None, 0.0).
        """
        if len(self.history) == 0:
            return None, 0.0
            
        # Extract just the signs for voting
        signs = [pred[0] for pred in self.history]
        
        # Count frequencies
        counter = collections.Counter(signs)
        most_common_sign, count = counter.most_common(1)[0]
        
        # Calculate average confidence for the most common sign
        confidences = [pred[1] for pred in self.history if pred[0] == most_common_sign]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Only return the prediction if it's consistently predicted AND high confidence
        if avg_confidence >= self.confidence_threshold and count >= (self.history_size // 2):
            return most_common_sign, avg_confidence
        
        return None, 0.0
