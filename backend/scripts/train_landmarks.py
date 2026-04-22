"""
Train Landmark Model: Trains the lightweight 1D-CNN + BiLSTM classifier
on pre-extracted hand landmark sequences.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import copy
import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from landmark_model import LandmarkASLModel


class LandmarkDataset(Dataset):
    def __init__(self, index_path, split="train"):
        df = pd.read_csv(index_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        landmarks = np.load(row["path"])  # (T, 63)
        label = int(row["label_idx"])
        
        # Add small random noise during training for extra regularization
        if self.split == "train" and np.random.random() < 0.3:
            landmarks = landmarks + np.random.normal(0, 0.01, landmarks.shape).astype(np.float32)
        
        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load label map
    with open("data/label_map.json", "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Classes: {num_classes} -> {list(label_map.keys())}")
    
    # Datasets
    train_ds = LandmarkDataset("data/landmark_index.csv", split="train")
    val_ds = LandmarkDataset("data/landmark_index.csv", split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Model
    model = LandmarkASLModel(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_acc = 0.0
    best_model_wts = None
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = "checkpoints/landmark_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved best model (val_acc={val_acc:.4f})")
    
    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.4f}")
    print(f"Model saved to: checkpoints/landmark_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    Path("checkpoints").mkdir(exist_ok=True)
    train(args)
