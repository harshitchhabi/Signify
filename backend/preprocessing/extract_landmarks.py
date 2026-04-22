"""
Signify STEP 1 — Extract Hand Landmarks
=========================================
Reads raw ASL sign videos and extracts hand landmark data using MediaPipe.

HOW IT WORKS:
  For each video file in data/raw_videos/:
    1. Opens the video with OpenCV (reads it frame by frame)
    2. Samples exactly 30 frames evenly across the video
    3. Runs MediaPipe HandLandmarker on each frame to detect 21 hand landmarks
    4. Saves the landmarks as a NumPy array: shape (30, 21, 3)
    5. Generates a labels.csv file mapping everything together

OUTPUT:
  - data/landmarks/{sign_name}/{video_name}.npy   → one per video
  - data/labels.csv                                → master mapping file
  - data/label_map.json                            → word → numeric ID

USAGE:
  cd backend
  python preprocessing/extract_landmarks.py
"""

import json
import random
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

# Import our config
from config import (
    BACKEND_DIR,
    RAW_VIDEO_DIR,
    LANDMARK_DIR,
    LABELS_CSV,
    LABEL_MAP_JSON,
    TARGET_SIGNS,
    NUM_FRAMES,
    NUM_LANDMARKS,
    NUM_COORDS,
    MAX_ZERO_FRAME_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
)

# ──────────────────────────────────────────────
# MEDIAPIPE MODEL SETUP
# ──────────────────────────────────────────────

# Path where the MediaPipe hand landmark model file will be stored.
# This is a small (~5MB) pre-trained model that MediaPipe needs to detect hands.
HAND_LANDMARKER_MODEL_PATH = BACKEND_DIR / "data" / "hand_landmarker.task"

# URL to download the model from Google's servers
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_downloaded():
    """
    Download the MediaPipe HandLandmarker model if it doesn't exist yet.

    WHY?
    The new MediaPipe Tasks API requires a separate model file (.task).
    We download it once and reuse it for all videos.
    """
    if HAND_LANDMARKER_MODEL_PATH.exists():
        print(f"✅ Model already exists: {HAND_LANDMARKER_MODEL_PATH.name}")
        return

    print(f"📥 Downloading HandLandmarker model...")
    print(f"   From: {HAND_LANDMARKER_MODEL_URL}")
    print(f"   To:   {HAND_LANDMARKER_MODEL_PATH}")

    try:
        urllib.request.urlretrieve(
            HAND_LANDMARKER_MODEL_URL,
            str(HAND_LANDMARKER_MODEL_PATH),
        )
        print(f"✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("   You can manually download it from:")
        print(f"   {HAND_LANDMARKER_MODEL_URL}")
        print(f"   and place it at: {HAND_LANDMARKER_MODEL_PATH}")
        sys.exit(1)


def create_hand_landmarker():
    """
    Create a MediaPipe HandLandmarker instance using the Tasks API.

    This replaces the old mp.solutions.hands API that was removed in
    newer versions of MediaPipe.

    Returns:
        A HandLandmarker object ready to process images
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,  # We're processing video frames
        num_hands=1,                           # Detect only one hand (STEP 1)
        min_hand_detection_confidence=0.5,     # How confident to be about detection
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────
# VIDEO READING
# ──────────────────────────────────────────────

def read_video_frames(video_path: str) -> tuple:
    """
    Read ALL frames from a video file and get its FPS.

    HOW IT WORKS:
    - Opens the video file using OpenCV's VideoCapture
    - Reads frames one by one in a loop
    - Each frame is an image (a NumPy array of pixels)
    - Returns all frames as a list of images, plus the video FPS

    Args:
        video_path: Path to the .mp4 video file

    Returns:
        Tuple of (frames_list, fps)
        - frames_list: List of frames, each a NumPy array of shape [H, W, 3]
        - fps: Frames per second of the video
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    ⚠️  Could not open video: {video_path}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback default

    frames = []
    while True:
        ret, frame = cap.read()  # ret = True/False, frame = the image
        if not ret:
            break  # No more frames
        frames.append(frame)

    cap.release()  # Always close the video file when done
    return frames, fps


def sample_frame_indices(total_frames: int, target_count: int) -> list:
    """
    Compute which frame indices to sample, evenly spaced across the video.

    WHY DO WE DO THIS?
    Videos have different lengths (some 2 seconds, some 5 seconds).
    ML models need all inputs to be the SAME size.
    So we pick exactly 30 frames, evenly spaced.

    Args:
        total_frames: How many frames the video has
        target_count: How many frames we want (default: 30)

    Returns:
        List of frame indices to use
    """
    if total_frames <= 0:
        return []

    if total_frames >= target_count:
        # Pick evenly spaced frames
        return np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()
    else:
        # Video is too short — use all frames, then repeat the last one
        indices = list(range(total_frames))
        while len(indices) < target_count:
            indices.append(total_frames - 1)  # Repeat last frame index
        return indices


# ──────────────────────────────────────────────
# LANDMARK EXTRACTION
# ──────────────────────────────────────────────

def extract_landmarks_from_frames(
    frames: list,
    frame_indices: list,
    fps: float,
    landmarker,
) -> np.ndarray:
    """
    Run MediaPipe HandLandmarker on sampled frames and extract 21 landmark points.

    WHAT ARE LANDMARKS?
    MediaPipe detects 21 key points on a hand:
      - Point 0: Wrist
      - Points 1-4: Thumb
      - Points 5-8: Index finger
      - Points 9-12: Middle finger
      - Points 13-16: Ring finger
      - Points 17-20: Pinky finger

    Each point has (x, y, z) coordinates:
      - x: Horizontal position (0.0 = left edge, 1.0 = right edge)
      - y: Vertical position   (0.0 = top edge,  1.0 = bottom edge)
      - z: Depth (relative to wrist)

    Args:
        frames:        All video frames
        frame_indices: Which frame indices to process (exactly 30)
        fps:           Video frames per second (needed for timestamps)
        landmarker:    MediaPipe HandLandmarker object

    Returns:
        NumPy array of shape (30, 21, 3)
    """
    all_landmarks = []

    for i, idx in enumerate(frame_indices):
        frame = frames[idx]

        # MediaPipe Tasks API needs an mp.Image object
        # Also convert BGR (OpenCV default) → RGB (MediaPipe expects)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds for this frame.
        # Use loop counter 'i' instead of 'idx' because 'idx' can have duplicates
        # at the end of short videos, avoiding "timestamp must be monotonically increasing" errors.
        timestamp_ms = int((i / fps) * 1000)

        # Run hand detection on this frame
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            # Hand detected! Extract the 21 landmark points.
            hand = result.hand_landmarks[0]  # Use first (dominant) hand

            # Convert landmarks to a simple list of [x, y, z]
            points = []
            for landmark in hand:
                points.append([landmark.x, landmark.y, landmark.z])

            all_landmarks.append(points)
        else:
            # No hand detected — fill with zeros to keep shape consistent.
            all_landmarks.append([[0.0, 0.0, 0.0]] * NUM_LANDMARKS)

    # Convert to NumPy array: shape (30, 21, 3)
    return np.array(all_landmarks, dtype=np.float32)


def check_quality(landmarks: np.ndarray) -> bool:
    """
    Check if the extracted landmarks are good enough to keep.

    We reject a video if too many frames have zero landmarks (meaning
    MediaPipe couldn't find a hand).

    Args:
        landmarks: NumPy array of shape (30, 21, 3)

    Returns:
        True if the video passes quality check, False if it should be skipped
    """
    zero_frames = 0
    for frame_idx in range(landmarks.shape[0]):
        frame_data = landmarks[frame_idx]  # shape (21, 3)
        if np.all(frame_data == 0):
            zero_frames += 1

    zero_ratio = zero_frames / landmarks.shape[0]

    if zero_ratio > MAX_ZERO_FRAME_RATIO:
        return False  # Too many failures

    return True


# ──────────────────────────────────────────────
# SAVING DATA
# ──────────────────────────────────────────────

def save_landmarks(landmarks: np.ndarray, output_path: Path):
    """
    Save landmark data as a .npy file.

    Args:
        landmarks:   NumPy array of shape (30, 21, 3)
        output_path: Where to save (e.g., data/landmarks/hello/hello_01.npy)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, landmarks)


def create_label_map() -> dict:
    """
    Create a mapping from sign names to numeric IDs.

    ML models work with numbers, not text.
    So "hello" → 0, "thank_you" → 1, etc.
    """
    label_map = {}
    for idx, sign in enumerate(TARGET_SIGNS):
        safe_name = sign.replace(" ", "_")
        label_map[safe_name] = idx
    return label_map


def save_labels_csv(records: list):
    """Save the master labels file."""
    df = pd.DataFrame(records)
    df.to_csv(LABELS_CSV, index=False)
    print(f"  Saved labels to: {LABELS_CSV}")


def assign_splits(records: list) -> list:
    """
    Randomly assign each record to train, val, or test split.

    - Train (70%): Data the model learns from
    - Val   (15%): Data used to check progress during training
    - Test  (15%): Data used ONLY at the end to measure accuracy
    """
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n = len(records)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    for i, record in enumerate(records):
        if i < train_end:
            record["split"] = "train"
        elif i < val_end:
            record["split"] = "val"
        else:
            record["split"] = "test"

    return records


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    """
    Main pipeline: reads all raw videos → extracts landmarks → saves results.
    """
    print("=" * 60)
    print("SIGNIFY — STEP 1: Extract Hand Landmarks")
    print("=" * 60)
    print()

    # ── Check that raw videos exist ──
    if not RAW_VIDEO_DIR.exists():
        print(f"ERROR: Raw video directory not found: {RAW_VIDEO_DIR}")
        print("  Run download_wlasl.py first, or manually place videos there.")
        sys.exit(1)

    # ── Create output directories ──
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    # ── Create label map ──
    label_map = create_label_map()
    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map to: {LABEL_MAP_JSON}")
    print(f"  Signs: {label_map}")
    print()

    # ── Download model if needed ──
    ensure_model_downloaded()
    print()

    # ── Initialize MediaPipe HandLandmarker ──
    print("Initializing MediaPipe HandLandmarker...")
    landmarker = create_hand_landmarker()
    print("✅ HandLandmarker ready")
    print()

    # ── Process all videos ──
    all_records = []
    total_processed = 0
    total_skipped = 0

    for sign_word in TARGET_SIGNS:
        safe_name = sign_word.replace(" ", "_")
        sign_folder = RAW_VIDEO_DIR / safe_name

        if not sign_folder.exists():
            print(f"[{safe_name.upper()}] ⚠️  Folder not found, skipping")
            continue

        # Find all video files in this sign's folder
        video_files = sorted(
            list(sign_folder.glob("*.mp4"))
            + list(sign_folder.glob("*.mov"))
            + list(sign_folder.glob("*.avi"))
            + list(sign_folder.glob("*.webm"))
        )

        if not video_files:
            print(f"[{safe_name.upper()}] ⚠️  No video files found")
            continue

        print(f"[{safe_name.upper()}] Processing {len(video_files)} videos...")

        sign_processed = 0
        sign_skipped = 0

        # Each sign needs its own landmarker instance because the VIDEO
        # running mode tracks state across frames. We recreate per-sign
        # to reset that tracking between different sign folders.
        landmarker.close()
        landmarker = create_hand_landmarker()

        for video_path in video_files:
            # We need a fresh landmarker per video because VIDEO mode
            # requires strictly increasing timestamps
            landmarker.close()
            landmarker = create_hand_landmarker()

            # Step 1: Read all frames from the video
            frames, fps = read_video_frames(str(video_path))

            if len(frames) == 0:
                print(f"    ⚠️  Empty/unreadable: {video_path.name}")
                sign_skipped += 1
                continue

            # Step 2: Determine which frames to sample (exactly 30)
            frame_indices = sample_frame_indices(len(frames), NUM_FRAMES)

            # Step 3: Extract landmarks using MediaPipe
            landmarks = extract_landmarks_from_frames(
                frames, frame_indices, fps, landmarker
            )

            # Step 4: Quality check
            if not check_quality(landmarks):
                print(f"    ⚠️  Too many failed detections: {video_path.name}")
                sign_skipped += 1
                continue

            # Step 5: Save the landmarks as .npy
            npy_filename = video_path.stem + ".npy"
            npy_path = LANDMARK_DIR / safe_name / npy_filename
            save_landmarks(landmarks, npy_path)

            # Step 6: Record this video's info for labels.csv
            all_records.append({
                "video_file": video_path.name,
                "sign": safe_name,
                "label_id": label_map[safe_name],
                "landmark_file": f"{safe_name}/{npy_filename}",
                "split": "",  # Assigned later
            })

            sign_processed += 1

        total_processed += sign_processed
        total_skipped += sign_skipped
        print(f"  → {sign_processed} saved, {sign_skipped} skipped")
        print()

    # ── Assign train/val/test splits and save labels ──
    if all_records:
        all_records = assign_splits(all_records)
        save_labels_csv(all_records)
    else:
        print("No videos were processed! Check your raw_videos directory.")
        landmarker.close()
        sys.exit(1)

    # ── Cleanup ──
    landmarker.close()

    # ── Print final summary ──
    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total processed: {total_processed}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Landmark shape:  ({NUM_FRAMES}, {NUM_LANDMARKS}, {NUM_COORDS})")
    print(f"  Output folder:   {LANDMARK_DIR}")
    print(f"  Labels file:     {LABELS_CSV}")
    print()

    # Count per split
    splits = {"train": 0, "val": 0, "test": 0}
    for r in all_records:
        splits[r["split"]] += 1
    print(f"  Train: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
    print()
    print("✅ STEP 1 complete! Landmark data is ready for model training.")


if __name__ == "__main__":
    main()
