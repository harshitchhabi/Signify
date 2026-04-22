"""
Landmark Extractor: Uses MediaPipe Hands to extract normalized 21-point
hand landmarks from video frames. Converts raw images into compact 63-dim
feature vectors (21 landmarks × 3 coords) that are position/scale invariant.
"""
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False

import cv2


class HandLandmarkExtractor:
    def __init__(self, static_mode=False, max_hands=1, min_confidence=0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not available. Install with: pip install mediapipe==0.10.14")
        
        self.hands = mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
    
    def extract_landmarks(self, frame_rgb):
        """
        Extract normalized hand landmarks from a single RGB frame.
        
        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format
            
        Returns:
            numpy array of shape (63,) if hand detected, None otherwise
        """
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Take the first detected hand
        hand = results.multi_hand_landmarks[0]
        
        # Extract raw coordinates
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])  # (21, 3)
        
        # Normalize: center on wrist (landmark 0)
        wrist = coords[0].copy()
        coords = coords - wrist  # translate so wrist is at origin
        
        # Scale: normalize by the max distance from wrist
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist
        
        return coords.flatten()  # (63,)
    
    def extract_from_video(self, video_path, num_frames=30):
        """
        Extract landmark sequences from a video file.
        
        Args:
            video_path: path to .mp4 file
            num_frames: number of frames to sample
            
        Returns:
            numpy array of shape (num_frames, 63), or None if no hand detected in any frame
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total <= 0:
            cap.release()
            return None
        
        # Sample frame indices uniformly
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        
        all_landmarks = []
        last_valid = None
        
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lm = self.extract_landmarks(rgb)
                if lm is not None:
                    all_landmarks.append(lm)
                    last_valid = lm
                else:
                    # If no hand detected in this frame, use last valid or zeros
                    all_landmarks.append(last_valid if last_valid is not None else np.zeros(63))
        
        cap.release()
        
        if len(all_landmarks) == 0:
            return None
        
        # Pad or trim to exactly num_frames
        while len(all_landmarks) < num_frames:
            all_landmarks.append(all_landmarks[-1] if all_landmarks else np.zeros(63))
        all_landmarks = all_landmarks[:num_frames]
        
        return np.array(all_landmarks, dtype=np.float32)  # (num_frames, 63)
    
    def close(self):
        self.hands.close()
