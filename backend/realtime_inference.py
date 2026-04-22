import sys
import collections
import cv2
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add 'training' directory to sys.path to fix import 'config' in model.py
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(BACKEND_DIR / "training"))

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from model import SignLanguageMLP
from temporal_smoothing_module import FrameBuffer, TemporalSmoother
from sentence_accumulator import SentenceAccumulator
from confidence_calibrator import ConfidenceCalibrator
from system_performance_logger import SystemPerformanceLogger

# Backend config Paths
LABEL_MAP_JSON = BACKEND_DIR / "data" / "label_map.json"
CHECKPOINT_PATH = BACKEND_DIR / "checkpoints" / "best_model.pth"

# Hyperparams
NUM_FRAMES = 30
FEATURE_SIZE = 63  # 21 landmarks * 3 coords
SMOOTHING_HISTORY = 10
CONFIDENCE_THRESHOLD = 0.7

def load_label_map():
    with open(LABEL_MAP_JSON, "r") as f:
        return json.load(f)

def normalize_landmarks(landmarks):
    """
    Normalizes a single frame consisting of 21 (x, y, z) landmarks relative to the wrist.
    Input: numpy array of shape (21, 3)
    Output: flattened numpy array of shape (63,)
    """
    wrist = landmarks[0] # Wrist is index 0
    normalized = landmarks - wrist
    return normalized.flatten()

def main():
    print("Initializing Real-Time Inference System...")
    
    # Load Label Map
    label_map = load_label_map()
    rev_map = {v: k for k, v in label_map.items()}
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SignLanguageMLP()
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(device)
        model.eval()
        print("MLP Model loaded successfully.")
    except Exception as e:
        print(f"FAILED TO LOAD MODEL: {e}")
        return

    # Initialize Buffers and Smoothing
    buffer = FrameBuffer(window_size=NUM_FRAMES, feature_size=FEATURE_SIZE)
    smoother = TemporalSmoother(history_size=SMOOTHING_HISTORY, confidence_threshold=CONFIDENCE_THRESHOLD)
    accumulator = SentenceAccumulator(freeze_duration=1.0, word_timeout=1.0, auto_finalize_timeout=3.0)
    calibrator = ConfidenceCalibrator(margin_threshold=0.15, default_threshold=0.70)
    logger = SystemPerformanceLogger()

    # Initialize MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path=str(BACKEND_DIR / 'data' / 'hand_landmarker.task'))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("System verified. Press 'q' to quit. Press 'c' to clear sentence.")

    prev_frame_time = 0
    current_predicted_sign = "Waiting..."
    current_confidence = 0.0
    current_raw_second_sign = None
    current_raw_second_conf = 0.0

    # For monotonic increasing timestamp tracking
    stream_start_ms = int(time.time() * 1000)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip the frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            # Process frame with MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.time() * 1000) - stream_start_ms
            if timestamp_ms <= 0:
                 timestamp_ms = 1

            result = detector.detect_for_video(mp_image, timestamp_ms)
            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                for hand_landmarks in result.hand_landmarks:
                    # Extract landmarks
                    lm_array = np.zeros((21, 3))
                    for i, lm in enumerate(hand_landmarks):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        lm_array[i] = [lm.x, lm.y, lm.z]
                    
                    # Normalize and Add to buffer
                    normalized_features = normalize_landmarks(lm_array)
                    buffer.add_frame(normalized_features)
                    
                    # Inference Action
                    if buffer.is_full() and not accumulator.is_frozen():
                        flat_features = buffer.get_flattened_buffer()
                        
                        # Convert to tensor shape (1, 1890)
                        input_tensor = torch.FloatTensor(flat_features).unsqueeze(0).to(device)
                        
                        raw_pred = None
                        raw_conf = 0.0
                        calibrated_pred = None
                        calibrated_conf = 0.0
                        
                        with torch.no_grad():
                            logits = model(input_tensor)
                            probs = F.softmax(logits, dim=1)
                            
                            # Calibrate Confidence
                            sign_name, is_accepted = calibrator.calibrate(probs, rev_map)
                            
                            # Grab raw top 2 for language correction
                            top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
                            raw_pred = rev_map[top2_indices[0][0].item()]
                            raw_conf = top2_probs[0][0].item()
                            current_raw_second_sign = rev_map[top2_indices[0][1].item()]
                            current_raw_second_conf = top2_probs[0][1].item()
                            
                            if is_accepted:
                                calibrated_pred = sign_name
                                calibrated_conf = raw_conf
                                
                                # Add to smoother
                                smoother.add_prediction(sign_name, raw_conf)
                                smoothed_sign, smoothed_prob = smoother.get_smoothed_prediction()
                                
                                if smoothed_sign is not None:
                                    current_predicted_sign = smoothed_sign
                                    current_confidence = smoothed_prob
                            else:
                                calibrated_pred = "Uncertain..."
                                calibrated_conf = 0.0
                                # Rejected by calibrator (ambiguous or low confidence)
                                current_predicted_sign = "Uncertain..."
                                current_confidence = 0.0
                                # Also flush smoother history to prevent stale accumulation jumping over the uncertainty
                                smoother.history.clear()
            else:
                # Clear buffer if hand is lost to prevent stale predictions
                buffer.clear()
                current_predicted_sign = "No Hand Detected"
                current_confidence = 0.0
                current_raw_second_sign = None
                current_raw_second_conf = 0.0

            # Feed the accumulator
            should_clear, corrected_word = accumulator.process(
                current_predicted_sign if current_confidence >= CONFIDENCE_THRESHOLD and current_predicted_sign != "Waiting..." else None,
                current_confidence,
                current_raw_second_sign,
                current_raw_second_conf,
                hand_detected
            )
            
            # Log all logic layers
            if hand_detected and buffer.is_full() and not accumulator.is_frozen():
                logger.log_frame(
                    fps=fps,
                    raw_pred=raw_pred,
                    raw_conf=raw_conf,
                    calibrated_pred=calibrated_pred,
                    calibrated_conf=calibrated_conf,
                    smoothed_pred=current_predicted_sign,
                    smoothed_conf=current_confidence,
                    corrected_pred=corrected_word if corrected_word else current_predicted_sign,
                    final_committed_word=accumulator.last_committed_word if should_clear else None
                )

            if should_clear:
                buffer.clear()
                smoother.history.clear()
                current_predicted_sign = "Waiting..."
                current_confidence = 0.0
                current_raw_second_sign = None
                current_raw_second_conf = 0.0

            # UI Overlay
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display Status/Prediction (Freeze indicator)
            if accumulator.is_frozen():
                status_color = (0, 255, 255) # Yellow for freeze
                display_text = "COMMITTED"
            else:
                status_color = (0, 255, 0) if "No" not in current_predicted_sign and "Waiting" not in current_predicted_sign else (0, 0, 255)
                display_text = current_predicted_sign.upper()
                
            cv2.putText(frame, f"Sign: {display_text}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
                        
            if current_confidence > 0 and not accumulator.is_frozen():
                cv2.putText(frame, f"Conf: {current_confidence:.2f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the built sentence at the bottom
            sentence = accumulator.get_sentence_string()
            if sentence:
                cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, sentence, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow("Signify ASL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                accumulator.clear()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        # Generate the analysis report when the app closes
        calibrator.generate_analysis_report()

if __name__ == "__main__":
    main()
