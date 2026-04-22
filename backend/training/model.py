"""
Signify STEP 2 — MLP Model
============================
A simple Multi-Layer Perceptron (MLP) for ASL sign classification.

WHAT IS AN MLP?
An MLP is the simplest type of neural network. It's a series of
"layers," each made of artificial neurons. Data flows through
the layers from input to output, getting transformed at each step.

Think of it like a series of filters:
  Raw data → Filter 1 → Filter 2 → Filter 3 → Answer

ARCHITECTURE:
  Input (1890) → Hidden1 (512) → Hidden2 (256) → Hidden3 (128) → Output (10)

Each hidden layer has:
  - Linear: multiplies input by learned weights (the actual "learning")
  - BatchNorm: normalizes values (makes training stable)
  - ReLU: keeps positive values, sets negatives to 0 (adds non-linearity)
  - Dropout: randomly turns off neurons (prevents memorization)
"""

import torch
import torch.nn as nn

from config import (
    INPUT_SIZE,
    NUM_CLASSES,
    HIDDEN_SIZES,
    DROPOUT_RATE,
)


class SignLanguageMLP(nn.Module):
    """
    Multi-Layer Perceptron for ASL sign classification.

    Takes a flattened landmark vector (1890 numbers) and outputs
    a score for each of the 10 signs. The highest score = the prediction.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
        hidden_sizes: list = None,
        dropout_rate: float = DROPOUT_RATE,
    ):
        """
        Args:
            input_size:    Number of input features (default: 1890)
            num_classes:   Number of output classes (default: 10)
            hidden_sizes:  List of hidden layer sizes (default: [512, 256, 128])
            dropout_rate:  Probability of dropout (default: 0.3)
        """
        super(SignLanguageMLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = HIDDEN_SIZES

        # ── Build the network layer by layer ──
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer: prev_size inputs → hidden_size outputs
            # This is where the actual learning happens — the model
            # learns the best weights to multiply the inputs by.
            layers.append(nn.Linear(prev_size, hidden_size))

            # BatchNorm: normalizes the outputs of the linear layer
            # WHY: Makes training faster and more stable by ensuring
            # values don't explode or vanish as they flow through layers.
            layers.append(nn.BatchNorm1d(hidden_size))

            # ReLU activation: f(x) = max(0, x)
            # WHY: Without activation functions, stacking multiple linear
            # layers would be equivalent to a single linear layer (because
            # multiplying matrices is still linear). ReLU adds non-linearity,
            # allowing the network to learn complex patterns like hand shapes.
            layers.append(nn.ReLU())

            # Dropout: randomly sets some neurons to 0 during training
            # WHY: Forces the network to spread learning across many neurons
            # instead of relying on a few. This prevents overfitting
            # (memorizing training data instead of learning general patterns).
            layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Final output layer: hidden_size → num_classes
        # No activation here — CrossEntropyLoss expects raw scores ("logits")
        layers.append(nn.Linear(prev_size, num_classes))

        # nn.Sequential chains all layers into one pipeline
        # When we call model(input), data flows through all layers in order
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process input through all layers.

        Args:
            x: Input tensor of shape (batch_size, 1890)

        Returns:
            Output tensor of shape (batch_size, 10)
            Each value is a raw score (logit) for one sign class.
            Higher score = model is more confident about that class.
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction (get the class with highest score).

        Args:
            x: Input tensor of shape (batch_size, 1890)

        Returns:
            Predicted class indices of shape (batch_size,)
            Each value is 0-9, corresponding to a sign in label_map.json
        """
        self.eval()  # Switch to evaluation mode (disables dropout)
        with torch.no_grad():  # Don't compute gradients (faster)
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def count_parameters(self) -> int:
        """
        Count the total number of learnable parameters (weights).
        Useful for understanding model complexity.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model() -> SignLanguageMLP:
    """
    Create a new MLP model and print its summary.

    Returns:
        A fresh SignLanguageMLP ready for training
    """
    model = SignLanguageMLP()

    print("Model Architecture:")
    print(f"  Input size:  {INPUT_SIZE}")
    print(f"  Hidden layers: {HIDDEN_SIZES}")
    print(f"  Output size: {NUM_CLASSES}")
    print(f"  Dropout:     {DROPOUT_RATE}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print()

    return model
