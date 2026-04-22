import torch
import cv2
import numpy as np
import json
import tempfile
import os
from landmark_model import LandmarkASLModel
from landmark_extractor import HandLandmarkExtractor

class SimpleASLClassifier:
    def __init__(self, model_path, label_map_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Label Map
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load Model
        self.model = LandmarkASLModel(num_classes=len(self.label_map))
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.extractor = HandLandmarkExtractor(static_mode=False, max_hands=1)
        self.num_frames = 16

    def predict_video(self, video_path):
        features = self.extractor.extract_from_video(video_path, num_frames=self.num_frames)
        if features is None:
            features = np.zeros((self.num_frames, 63), dtype=np.float32)
            
        tensor = torch.tensor(features, dtype=torch.float32)
        return self._predict_tensor(tensor)

    def predict_frame_buffer(self, frames_list):
        """
        Args:
            frames_list: List of numpy arrays (RGB)
        """
        if len(frames_list) == 0:
            return {"text": "", "confidence": 0.0}
            
        indices = np.linspace(0, len(frames_list) - 1, self.num_frames, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]
        
        all_landmarks = []
        last_valid = None
        
        for frame in sampled_frames:
            lm = self.extractor.extract_landmarks(frame)
            if lm is not None:
                all_landmarks.append(lm)
                last_valid = lm
            else:
                all_landmarks.append(last_valid if last_valid is not None else np.zeros((63,), dtype=np.float32))
                
        if all([x is None for x in all_landmarks]):
             features = np.zeros((self.num_frames, 63), dtype=np.float32)
        else:
             for i in range(len(all_landmarks)):
                 if all_landmarks[i] is None:
                     all_landmarks[i] = np.zeros((63,), dtype=np.float32)
             features = np.array(all_landmarks, dtype=np.float32)
             
        tensor = torch.tensor(features, dtype=torch.float32)
        return self._predict_tensor(tensor)

    def _predict_tensor(self, seq_tensor):
        # [T, 63] -> [1, T, 63]
        input_tensor = seq_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
        return {
            "text": self.idx_to_label[pred_idx.item()],
            "confidence": float(conf.item())
        }

