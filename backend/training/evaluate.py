"""
Signify STEP 2 — Evaluation Script
=====================================
Evaluates the trained MLP model on the test set.

WHAT THIS DOES:
  1. Loads the best saved model from checkpoints/
  2. Runs it on the TEST set (data the model has never seen)
  3. Prints accuracy, per-class metrics, and a confusion matrix

WHY TEST ON UNSEEN DATA?
  During training, the model saw the training data many times and the
  validation data was used to pick the best model. The TEST set is data
  the model has NEVER seen — it gives us an honest measure of how well
  the model will work on new, real-world data.

USAGE:
  cd backend
  python training/evaluate.py
"""

import json
import sys

import numpy as np
import torch

from config import (
    BEST_MODEL_PATH,
    LABEL_MAP_JSON,
    NUM_CLASSES,
)
from dataset import create_data_loaders, get_reverse_label_map
from model import SignLanguageMLP


def load_trained_model(device) -> SignLanguageMLP:
    """
    Load the best model from the checkpoint file.

    Returns:
        The trained model, ready for evaluation
    """
    if not BEST_MODEL_PATH.exists():
        print(f"ERROR: No trained model found at: {BEST_MODEL_PATH}")
        print("  Run train.py first to train a model.")
        sys.exit(1)

    # Load the checkpoint
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

    # Create a fresh model and load the saved weights
    model = SignLanguageMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Validation accuracy at save: {checkpoint['val_accuracy']:.1%}")

    return model


def evaluate_model(model, test_loader, device) -> tuple:
    """
    Run the model on all test samples and collect predictions.

    Returns:
        (all_true_labels, all_predicted_labels) as numpy arrays
    """
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)

            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=1)

            all_true.extend(batch_labels.numpy())
            all_pred.extend(predictions.cpu().numpy())

    return np.array(all_true), np.array(all_pred)


def print_confusion_matrix(true_labels, pred_labels, label_names):
    """
    Print a text-based confusion matrix.

    WHAT IS A CONFUSION MATRIX?
    It's a grid that shows:
      - Rows = what the TRUE label is
      - Columns = what the model PREDICTED
      - Each cell = how many times that combination occurred

    A perfect model has all numbers on the diagonal (true = predicted).
    Off-diagonal numbers show which signs get confused with each other.

    Example:
                    Predicted
                    hello  yes  no
    True  hello  [  15    1    0  ]  ← 15 correct, 1 confused with "yes"
          yes    [   0   14    2  ]
          no     [   1    0   13  ]
    """
    num_classes = len(label_names)

    # Build the confusion matrix manually (no sklearn needed)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        matrix[true][pred] += 1

    # Find the widest label name for formatting
    max_name_len = max(len(name) for name in label_names)
    cell_width = max(5, max_name_len)

    # Print header
    print()
    print("CONFUSION MATRIX")
    print("  (Rows = True label, Columns = Predicted label)")
    print()

    # Column headers
    header = " " * (max_name_len + 2)
    for name in label_names:
        header += f"{name:>{cell_width}} "
    print(header)
    print(" " * (max_name_len + 2) + "-" * ((cell_width + 1) * num_classes))

    # Rows
    for i, name in enumerate(label_names):
        row = f"{name:>{max_name_len}} |"
        for j in range(num_classes):
            val = matrix[i][j]
            if i == j:
                # Highlight diagonal (correct predictions)
                row += f"{val:>{cell_width}} "
            else:
                row += f"{val:>{cell_width}} "
        # Also show row accuracy
        row_total = matrix[i].sum()
        row_acc = matrix[i][i] / row_total if row_total > 0 else 0
        row += f" | {row_acc:.0%}"
        print(row)

    print()
    return matrix


def print_per_class_metrics(true_labels, pred_labels, label_names):
    """
    Print accuracy for each individual sign class.

    WHY PER-CLASS METRICS?
    Overall accuracy can be misleading. If 90% of your data is "hello"
    and the model always predicts "hello," you'd get 90% accuracy but
    the model is useless for other signs.

    Per-class metrics show how well each sign is recognized individually.
    """
    print("PER-CLASS ACCURACY")
    print("-" * 40)

    for class_id, name in enumerate(label_names):
        # How many times this class appeared in the test set
        total = (true_labels == class_id).sum()

        if total == 0:
            print(f"  {name:>15}: no test samples")
            continue

        # How many were correctly predicted
        correct = ((true_labels == class_id) & (pred_labels == class_id)).sum()
        accuracy = correct / total

        # Visual bar
        bar_len = int(accuracy * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"  {name:>15}: {bar} {accuracy:.0%} ({correct}/{total})")

    print()


def print_debugging_advice(true_labels, pred_labels, label_names, overall_accuracy):
    """
    Print helpful advice based on the evaluation results.
    """
    print("=" * 60)
    print("DIAGNOSTIC ADVICE")
    print("=" * 60)

    if overall_accuracy >= 0.9:
        print("🎉 Excellent! >90% accuracy. The model is working very well.")
        print("   Next steps: Try adding more signs, or connect to the frontend.")
    elif overall_accuracy >= 0.8:
        print("✅ Good! >80% accuracy. The model is working.")
        print("   To improve: Add more training videos, or try data augmentation.")
    elif overall_accuracy >= 0.5:
        print("⚠️  Moderate accuracy. The model is learning, but struggling.")
        print("   Possible fixes:")
        print("   - Check if some signs have very few training videos")
        print("   - Look at the confusion matrix — which signs get confused?")
        print("   - Try reducing dropout rate (currently 0.3, try 0.2)")
        print("   - Try training for more epochs (increase MAX_EPOCHS)")
    else:
        print("❌ Low accuracy. The model isn't learning effectively.")
        print("   Possible fixes:")
        print("   - Check your .npy files — are they valid? (shape should be 30, 21, 3)")
        print("   - Check labels.csv — are labels correct?")
        print("   - Try a lower learning rate (0.0005 or 0.0001)")
        print("   - Make sure you have at least 10 samples per sign")

    # Check for class imbalance
    unique, counts = np.unique(true_labels, return_counts=True)
    if len(counts) > 1:
        ratio = counts.max() / counts.min()
        if ratio > 3:
            print()
            print("⚠️  CLASS IMBALANCE DETECTED")
            print(f"   Largest class has {ratio:.1f}x more samples than smallest.")
            print("   This can cause the model to favor the larger class.")
            print("   Fix: Record more videos for under-represented signs.")

    print()


def main():
    """
    Complete evaluation pipeline.
    """
    print("=" * 60)
    print("SIGNIFY — STEP 2: Evaluate Trained Model")
    print("=" * 60)
    print()

    # ── Setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = load_trained_model(device)
    print()

    # ── Load test data ──
    print("Loading test data...")
    _, _, test_loader = create_data_loaders()
    print()

    # ── Get label names ──
    reverse_map = get_reverse_label_map()
    label_names = [reverse_map[i] for i in range(len(reverse_map))]

    # ── Run evaluation ──
    print("Running evaluation on test set...")
    true_labels, pred_labels = evaluate_model(model, test_loader, device)

    # ── Overall accuracy ──
    overall_accuracy = (true_labels == pred_labels).sum() / len(true_labels)
    print()
    print(f"OVERALL TEST ACCURACY: {overall_accuracy:.1%} "
          f"({(true_labels == pred_labels).sum()}/{len(true_labels)} correct)")
    print()

    # ── Detailed metrics ──
    print_per_class_metrics(true_labels, pred_labels, label_names)
    print_confusion_matrix(true_labels, pred_labels, label_names)
    print_debugging_advice(true_labels, pred_labels, label_names, overall_accuracy)


if __name__ == "__main__":
    main()
