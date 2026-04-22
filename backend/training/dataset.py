"""
Signify STEP 2 — Dataset Loader
=================================
Loads .npy landmark files, normalizes them, and prepares them for training.

THE DATA PIPELINE:
  1. Read labels.csv to find all landmark files and their labels
  2. Load each .npy file → shape (30, 21, 3)
  3. Normalize: subtract wrist position from all landmarks (per frame)
  4. Flatten: (30, 21, 3) → (1890,)
  5. Return as PyTorch tensors ready for the model
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import (
    LANDMARK_DIR,
    LABELS_CSV,
    LABEL_MAP_JSON,
    NUM_FRAMES,
    NUM_LANDMARKS,
    NUM_COORDS,
    INPUT_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)


class LandmarkDataset(Dataset):
    """
    PyTorch Dataset that loads hand landmark .npy files.

    WHAT IS A DATASET?
    In PyTorch, a Dataset is an object that knows:
      - How many samples you have  → __len__()
      - How to load one sample     → __getitem__(index)
    PyTorch then handles batching and shuffling automatically.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            split: Which data split to load — "train", "val", or "test"
        """
        # Load the master CSV that maps files → labels → splits
        if not LABELS_CSV.exists():
            raise FileNotFoundError(
                f"Labels file not found: {LABELS_CSV}\n"
                f"Run STEP 1 (extract_landmarks.py) first to generate it."
            )

        df = pd.read_csv(LABELS_CSV)

        # Filter to only the requested split
        self.data = df[df["split"] == split].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"No samples found for split '{split}'.\n"
                f"Available splits: {df['split'].unique().tolist()}"
            )

        print(f"  Loaded {len(self.data)} samples for '{split}' split")

    def __len__(self) -> int:
        """How many samples in this split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load and preprocess one sample.

        Returns:
            (features, label) tuple where:
              - features: torch.FloatTensor of shape (1890,)
              - label: torch.LongTensor scalar (0-9)
        """
        row = self.data.iloc[idx]

        # ── Step 1: Load the .npy file ──
        npy_path = LANDMARK_DIR / row["landmark_file"]
        landmarks = np.load(npy_path)  # shape: (30, 21, 3)

        # ── Step 2: Normalize (wrist-relative) ──
        landmarks = normalize_landmarks(landmarks)

        # ── Step 3: Flatten to 1D vector ──
        flat = landmarks.flatten()  # shape: (1890,)

        # ── Step 4: Convert to PyTorch tensors ──
        features = torch.FloatTensor(flat)
        label = torch.LongTensor([row["label_id"]])[0]  # scalar tensor

        return features, label


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks by making them relative to the wrist (landmark 0).

    WHY DO THIS?
    If someone signs "hello" on the left side of the camera vs the right side,
    the raw MediaPipe coordinates would be totally different numbers — but
    the actual hand shape and motion are identical.

    By subtracting the wrist position from every other landmark, we make the
    coordinates "wrist-relative." Now the hand shape is described the same way
    regardless of where in the frame the hand appears.

    BEFORE:  Wrist=(0.7, 0.5), Index=(0.8, 0.3)  →  depends on position
    AFTER:   Wrist=(0.0, 0.0), Index=(0.1, -0.2)  →  same regardless of position

    Args:
        landmarks: shape (30, 21, 3) — raw MediaPipe coordinates

    Returns:
        Normalized landmarks: shape (30, 21, 3) — wrist-relative coordinates
    """
    normalized = landmarks.copy()

    for frame_idx in range(landmarks.shape[0]):
        # Get the wrist position for this frame (landmark 0)
        wrist = landmarks[frame_idx, 0, :]  # shape: (3,)

        # Subtract wrist from ALL 21 landmarks in this frame
        # After this: wrist = (0, 0, 0), all others are relative offsets
        normalized[frame_idx] = landmarks[frame_idx] - wrist

    return normalized


def create_data_loaders() -> tuple:
    """
    Create PyTorch DataLoader objects for train, val, and test splits.

    WHAT IS A DATALOADER?
    A DataLoader takes a Dataset and:
      - Splits it into batches (groups of 32 samples)
      - Shuffles the training data each epoch (so the model doesn't
        see samples in the same order every time)
      - Loads data efficiently in the background

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    print("Loading datasets...")

    train_dataset = LandmarkDataset(split="train")
    val_dataset = LandmarkDataset(split="val")
    test_dataset = LandmarkDataset(split="test")

    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,       # ← Randomize order each epoch (important for training!)
        generator=generator,
        drop_last=False,     # Keep the last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,       # ← No shuffling needed for validation/test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def load_label_map() -> dict:
    """
    Load the label map (sign name → numeric ID).

    Returns:
        Dictionary like {"hello": 0, "thank_you": 1, ...}
    """
    with open(LABEL_MAP_JSON, "r") as f:
        return json.load(f)


def get_reverse_label_map() -> dict:
    """
    Get the reverse label map (numeric ID → sign name).
    Useful for converting model predictions back to readable words.

    Returns:
        Dictionary like {0: "hello", 1: "thank_you", ...}
    """
    label_map = load_label_map()
    return {v: k for k, v in label_map.items()}
