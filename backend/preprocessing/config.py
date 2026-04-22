"""
Signify STEP 1 — Configuration
===============================
All constants and paths used by the preprocessing pipeline.
Change these values to adjust the pipeline behavior.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PATHS (relative to backend/ directory)
# ──────────────────────────────────────────────

# Root of the backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Where raw videos are downloaded (organized by sign name)
RAW_VIDEO_DIR = BACKEND_DIR / "data" / "raw_videos"

# Where extracted landmark .npy files are saved
LANDMARK_DIR = BACKEND_DIR / "data" / "landmarks"

# Master CSV mapping videos → labels → landmarks → splits
LABELS_CSV = BACKEND_DIR / "data" / "labels.csv"

# JSON file mapping sign names to numeric IDs
LABEL_MAP_JSON = BACKEND_DIR / "data" / "label_map.json"

# WLASL dataset JSON file (download from WLASL GitHub repo)
WLASL_JSON = BACKEND_DIR / "data" / "WLASL_v0.3.json"

# ──────────────────────────────────────────────
# TARGET SIGNS (10 common ASL words)
# ──────────────────────────────────────────────
# These were chosen because they are:
#   1. Common everyday words
#   2. Visually distinct from each other
#   3. Well-represented in the WLASL dataset
#   4. Useful for a greeting/conversation scenario

TARGET_SIGNS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "good",
    "bad",
    "love",
]

# ──────────────────────────────────────────────
# LANDMARK EXTRACTION SETTINGS
# ──────────────────────────────────────────────

# Number of frames to sample from each video.
# WHY 30: Most ASL signs take 1-3 seconds. 30 frames captures
# enough motion while keeping file sizes small. All videos are
# padded/trimmed to this exact count so ML models can batch them.
NUM_FRAMES = 30

# Number of hand landmark points detected by MediaPipe.
# MediaPipe Hands always returns exactly 21 points per hand.
NUM_LANDMARKS = 21

# Number of coordinates per landmark point (x, y, z).
# x,y are normalized to [0,1] based on image dimensions.
# z represents depth relative to the wrist.
NUM_COORDS = 3

# Maximum percentage of frames allowed to have zero (failed) detections.
# Videos exceeding this threshold are skipped as unreliable.
# 0.5 = 50%, meaning if more than half the frames have no hand detected,
# the video is considered too noisy.
MAX_ZERO_FRAME_RATIO = 0.5

# ──────────────────────────────────────────────
# DOWNLOAD SETTINGS
# ──────────────────────────────────────────────

# Maximum number of videos to download per sign.
# WLASL may have 50+ videos per word; we only need 15-25 for STEP 1.
MAX_VIDEOS_PER_SIGN = 40

# ──────────────────────────────────────────────
# DATA SPLIT RATIOS
# ──────────────────────────────────────────────

# How to split the data for training, validation, and testing.
# Train = 70% → used to teach the model
# Val   = 15% → used to tune the model during training
# Test  = 15% → used to evaluate the final model (never seen during training)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
