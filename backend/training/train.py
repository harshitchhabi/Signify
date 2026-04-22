"""
Signify STEP 2 — Training Script
===================================
Trains the MLP model on hand landmark data with early stopping.

WHAT HAPPENS DURING TRAINING?
  1. Load training and validation data
  2. Create the MLP model
  3. For each epoch (pass through all training data):
     a. Feed training samples through the model
     b. Measure how wrong the predictions are (loss)
     c. Update the model weights to reduce the loss
     d. Check accuracy on validation data
     e. Save the model if it's the best so far
  4. Stop early if the model stops improving

USAGE:
  cd backend
  python training/train.py
"""

import json
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    TRAINING_HISTORY_PATH,
)
from dataset import create_data_loaders
from model import create_model


def train_one_epoch(model, train_loader, loss_fn, optimizer, device) -> tuple:
    """
    Train the model for one complete pass through the training data.

    HOW DOES LEARNING WORK?
    For each batch of samples:
      1. Forward pass: give samples to model, get predictions
      2. Compute loss: measure how wrong predictions are
      3. Backward pass: compute gradients (direction to adjust weights)
      4. Update weights: nudge weights to reduce the loss

    Args:
        model:        The MLP model
        train_loader: DataLoader with training samples
        loss_fn:      Loss function (CrossEntropyLoss)
        optimizer:    Optimizer (Adam)
        device:       "cpu" or "cuda" (GPU)

    Returns:
        (average_loss, accuracy) for this epoch
    """
    model.train()  # Enable training mode (activates dropout + batch norm)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_features, batch_labels in train_loader:
        # Move data to the right device (CPU or GPU)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # ── Step 1: Forward pass ──
        # Feed inputs through the model to get predictions
        outputs = model(batch_features)  # shape: (batch_size, 10)

        # ── Step 2: Compute loss ──
        # CrossEntropyLoss measures how far predictions are from true labels.
        # Lower loss = better predictions.
        loss = loss_fn(outputs, batch_labels)

        # ── Step 3: Backward pass ──
        # First, clear any leftover gradients from the previous batch
        optimizer.zero_grad()
        # Compute gradients: "which direction should each weight move?"
        loss.backward()

        # ── Step 4: Update weights ──
        # Nudge each weight in the direction that reduces the loss
        optimizer.step()

        # ── Track metrics ──
        total_loss += loss.item() * batch_features.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, loss_fn, device) -> tuple:
    """
    Evaluate the model on validation data (no learning happens here).

    WHY VALIDATE?
    During training, we only see how well the model does on training data.
    Validation data is data the model has never trained on, so it tells us
    how well the model generalizes to new, unseen samples.

    Args:
        model:      The MLP model
        val_loader: DataLoader with validation samples
        loss_fn:    Loss function
        device:     "cpu" or "cuda"

    Returns:
        (average_loss, accuracy) on validation data
    """
    model.eval()  # Disable dropout + batch norm (use learned values)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Don't compute gradients (saves memory + speed)
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            loss = loss_fn(outputs, batch_labels)

            total_loss += loss.item() * batch_features.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train():
    """
    Complete training pipeline with early stopping and model saving.
    """
    print("=" * 60)
    print("SIGNIFY — STEP 2: Train MLP Classifier")
    print("=" * 60)
    print()

    # ── Set random seeds for reproducibility ──
    torch.manual_seed(RANDOM_SEED)

    # ── Choose device (CPU or GPU) ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Load data ──
    train_loader, val_loader, test_loader = create_data_loaders()
    print()

    # ── Create model ──
    model = create_model()
    model = model.to(device)

    # ── Setup training components ──

    # Loss function: CrossEntropyLoss
    # Measures how wrong the model's predictions are.
    # Perfect prediction = loss of 0. Random guessing (10 classes) ≈ loss of 2.3.
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer: Adam
    # The algorithm that updates model weights to reduce loss.
    # Adam is the most popular optimizer — it adapts the learning rate
    # for each weight individually, which usually works better than
    # a fixed learning rate.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler: ReduceLROnPlateau
    # If validation loss stops improving for 7 epochs, it cuts the
    # learning rate in half. This lets the model take smaller steps
    # to fine-tune when it's close to a good solution.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",           # Monitor validation loss (lower = better)
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True,
    )

    # ── Early stopping setup ──
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # ── Training history (for plotting later) ──
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    # ── Create checkpoint directory ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print("Starting training...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10} | {'Status'}")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train for one epoch ──
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )

        # ── Validate ──
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # ── Update learning rate scheduler ──
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # ── Record history ──
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        # ── Check for improvement ──
        status = ""
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            status = "⭐ Best!"

            # Save the best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                status = "🛑 Early stop"

        # ── Print progress ──
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
            f"{val_loss:>10.4f} | {val_acc:>8.1%} | {current_lr:>10.6f} | {status}"
        )

        # ── Early stopping ──
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print()
            print(f"Early stopping triggered after {epoch} epochs.")
            print(f"No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break

    # ── Training complete ──
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total epochs:        {epoch}")
    print(f"  Training time:       {elapsed:.1f} seconds")
    print(f"  Best val accuracy:   {best_val_accuracy:.1%}")
    print(f"  Model saved to:      {BEST_MODEL_PATH}")

    # ── Save training history ──
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to:    {TRAINING_HISTORY_PATH}")
    print()
    print("Next step: Run evaluate.py to test on the held-out test set.")


if __name__ == "__main__":
    train()
