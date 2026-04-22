"""
Fast Landmark Preparation: Extracts hand landmarks from training videos
using MediaPipe, with aggressive optimizations for speed.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import mediapipe with fallback
try:
    import mediapipe as mp
    mp_hands_module = mp.solutions.hands
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands_module


NUM_FRAMES = 20  # frames to sample per video


def extract_landmarks_from_video(video_path, hands_detector, num_frames=NUM_FRAMES):
    """Extract normalized hand landmarks from a video quickly."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        cap.release()
        return None
    
    # Sample fewer frames for speed
    sample_indices = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int))
    
    all_landmarks = []
    last_valid = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in sample_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for speed - MediaPipe doesn't need full resolution
            small = cv2.resize(rgb, (320, 240))
            
            results = hands_detector.process(small)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
                
                # Normalize: center on wrist
                wrist = coords[0].copy()
                coords -= wrist
                
                # Scale normalize
                max_dist = np.max(np.linalg.norm(coords, axis=1))
                if max_dist > 0:
                    coords /= max_dist
                
                flat = coords.flatten()  # (63,)
                all_landmarks.append(flat)
                last_valid = flat
            else:
                all_landmarks.append(last_valid if last_valid is not None else np.zeros(63, dtype=np.float32))
        
        frame_idx += 1
    
    cap.release()
    
    if not all_landmarks or last_valid is None:
        return None
    
    # Pad/trim to num_frames
    while len(all_landmarks) < num_frames:
        all_landmarks.append(all_landmarks[-1])
    all_landmarks = all_landmarks[:num_frames]
    
    return np.array(all_landmarks, dtype=np.float32)


def augment(landmarks):
    """Create 4 augmented copies."""
    augmented = []
    T, D = landmarks.shape
    
    # 1. Mirror (flip x)
    lm = landmarks.copy().reshape(T, 21, 3)
    lm[:, :, 0] *= -1
    augmented.append(lm.reshape(T, D))
    
    # 2. Gaussian noise
    lm = landmarks.copy() + np.random.normal(0, 0.02, landmarks.shape).astype(np.float32)
    augmented.append(lm)
    
    # 3. Scale
    lm = landmarks.copy() * np.random.uniform(0.85, 1.15)
    augmented.append(lm.astype(np.float32))
    
    # 4. Temporal shift
    lm = np.roll(landmarks.copy(), np.random.randint(1, max(2, T // 4)), axis=0)
    augmented.append(lm)
    
    return augmented


def main():
    raw_dir = "data/raw_videos"
    output_dir = "data/landmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    with open("data/label_map.json") as f:
        label_map = json.load(f)
    
    # Create ONE detector and reuse it
    hands = mp_hands_module.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )
    
    records = []
    processed = 0
    failed = 0
    
    for sign, idx in label_map.items():
        sign_dir = os.path.join(raw_dir, sign)
        if not os.path.isdir(sign_dir):
            continue
        
        videos = sorted([f for f in os.listdir(sign_dir) if f.endswith('.mp4')])
        
        for vf in videos:
            vpath = os.path.join(sign_dir, vf)
            lm = extract_landmarks_from_video(vpath, hands)
            
            if lm is None:
                failed += 1
                continue
            
            base = os.path.splitext(vf)[0]
            
            # Save original
            p = os.path.join(output_dir, f"{sign}_{base}.npy")
            np.save(p, lm)
            records.append({"path": p, "label": sign, "label_idx": idx})
            
            # Save augmented
            for ai, alm in enumerate(augment(lm)):
                ap = os.path.join(output_dir, f"{sign}_{base}_aug{ai}.npy")
                np.save(ap, alm)
                records.append({"path": ap, "label": sign, "label_idx": ai})
                records[-1]["label_idx"] = idx  # fix
            
            processed += 1
        
        # Write progress to file so we can monitor
        with open("_progress.txt", "w") as pf:
            pf.write(f"Done: {sign} | processed={processed} failed={failed}\n")
    
    hands.close()
    
    # Build CSV
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df["split"] = "val"
    df.loc[:split_idx, "split"] = "train"
    
    df.to_csv("data/landmark_index.csv", index=False)
    
    with open("_progress.txt", "w") as pf:
        pf.write(f"COMPLETE | processed={processed} failed={failed} total_samples={len(df)}\n")


if __name__ == "__main__":
    main()
