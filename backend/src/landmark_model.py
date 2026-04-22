"""
Landmark-based ASL Classifier: A lightweight 1D-CNN + BiLSTM model.

Takes sequences of normalized hand landmarks (T, 63) and classifies them
into ASL signs. Total ~50K parameters — trains in seconds, not hours.
"""
import torch
import torch.nn as nn


class LandmarkASLModel(nn.Module):
    def __init__(self, num_classes=10, input_dim=63, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LandmarkASLModel, self).__init__()
        
        # 1D Convolutional layers to extract local temporal patterns
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM to capture full gesture trajectory
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 63) — landmark sequences
        Returns:
            logits: (batch, num_classes)
        """
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x.permute(0, 2, 1)
        
        x_conv = self.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = self.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.dropout(x_conv)
        
        # Back to (batch, seq_len, features)
        x_seq = x_conv.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x_seq)
        
        # Use the last timestep output
        final = lstm_out[:, -1, :]
        
        return self.fc(final)
