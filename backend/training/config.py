"""
Signify STEP 2 — Training Configuration
=========================================
All hyperparameters and paths for training the MLP classifier.

WHAT ARE HYPERPARAMETERS?
They are settings that control HOW the model learns.
Think of them like the knobs on an oven — they control the
temperature and time, but YOU set them (the oven doesn't learn them).
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Import STEP 1 config for data paths
# ──────────────────────────────────────────────

# We load preprocessing/config.py by its file path to avoid name
# conflicts (both files are called "config.py").
import importlib.util

BACKEND_DIR = Path(__file__).resolve().parent.parent
_preprocess_config_path = str(BACKEND_DIR / "preprocessing" / "config.py")
_spec = importlib.util.spec_from_file_location("preprocessing_config", _preprocess_config_path)
_preprocess_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preprocess_config)

LANDMARK_DIR   = _preprocess_config.LANDMARK_DIR
LABELS_CSV     = _preprocess_config.LABELS_CSV
LABEL_MAP_JSON = _preprocess_config.LABEL_MAP_JSON
NUM_FRAMES     = _preprocess_config.NUM_FRAMES
NUM_LANDMARKS  = _preprocess_config.NUM_LANDMARKS
NUM_COORDS     = _preprocess_config.NUM_COORDS
TARGET_SIGNS   = _preprocess_config.TARGET_SIGNS

# ──────────────────────────────────────────────
# INPUT DIMENSIONS
# ──────────────────────────────────────────────

# Total numbers fed into the model per video.
# We flatten (30 frames × 21 landmarks × 3 coords) into one long vector.
INPUT_SIZE = NUM_FRAMES * NUM_LANDMARKS * NUM_COORDS  # = 1890

# Number of output classes (one per sign)
NUM_CLASSES = len(TARGET_SIGNS)  # = 10

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────

# Sizes of the three hidden layers.
# We go from large → small to progressively compress the information.
# 512 → 256 → 128 is a common "funnel" pattern in MLPs.
HIDDEN_SIZES = [512, 256, 128]

# Dropout rate: what fraction of neurons to randomly turn off during training.
# 0.3 means 30% of neurons are turned off each training step.
# WHY: Prevents the model from memorizing the training data (overfitting).
DROPOUT_RATE = 0.3

# ──────────────────────────────────────────────
# TRAINING SETTINGS
# ──────────────────────────────────────────────

# Learning rate: how big each learning step is.
# Too high = model jumps around and never converges
# Too low = model learns too slowly
# 0.001 is the default for Adam optimizer and works well for most cases.
LEARNING_RATE = 0.001

# Batch size: how many samples the model sees at once before updating.
# 32 is a safe default — small enough for laptop memory, large enough
# for stable learning.
BATCH_SIZE = 32

# Maximum number of training epochs (full passes through training data).
# We set a high limit and rely on early stopping to end sooner.
MAX_EPOCHS = 100

# Early stopping patience: how many epochs to wait for improvement
# before stopping training. If validation accuracy doesn't improve
# for this many epochs, we assume the model is done learning.
EARLY_STOPPING_PATIENCE = 15

# Learning rate scheduler: cuts the learning rate when progress stalls.
# factor=0.5 means halve the learning rate.
# patience=7 means wait 7 epochs without improvement before cutting.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 7

# Random seed for reproducibility (same seed = same results)
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# OUTPUT PATHS
# ──────────────────────────────────────────────

# Where to save the trained model
CHECKPOINT_DIR = BACKEND_DIR / "checkpoints"

# Best model file (saved whenever validation accuracy improves)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# Training history (loss/accuracy per epoch, for plotting later)
TRAINING_HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
