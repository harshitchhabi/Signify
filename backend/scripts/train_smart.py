"""
Smart Frame Trainer: Instead of MediaPipe landmarks, we use a smarter
training approach with the existing ResNet18:

1. Heavy data augmentation (color jitter, random crop, flip, rotation)
2. Freeze the ResNet backbone (only train the LSTM + classifier head)
3. Use a cosine annealing learning rate schedule
4. Train for 100 epochs (fast because backbone is frozen)

This dramatically reduces overfitting since we only train ~500K params
instead of 11M, and augmentation effectively 10x our dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import cv2
import numpy as np
import pandas as pd
import json
import copy
from pathlib import Path
from PIL import Image


class AugmentedVideoDataset(Dataset):
    """Video dataset with heavy augmentation for small datasets."""
    
    def __init__(self, index_path, split="train", num_frames=16):
        df = pd.read_csv(index_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.num_frames = num_frames
        self.split = split
        
        # Heavy augmentation for training
        if split == "train":
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.2),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        # During training, virtually multiply dataset by returning more samples
        if self.split == "train":
            return len(self.data) * 5  # each video seen 5 times with different augmentation
        return len(self.data)
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        row = self.data.iloc[real_idx]
        video_path = row["video_path"]
        label = int(row["label_idx"])
        
        frames = self._load_video(video_path)
        
        processed = []
        for frame in frames:
            img = Image.fromarray(frame)
            processed.append(self.transform(img))
        
        return torch.stack(processed), torch.tensor(label, dtype=torch.long)
    
    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total <= 0:
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        
        indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int))
        
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.num_frames]


def train_smart(epochs=100, batch_size=4, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("data/label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    # Import the model
    import sys
    sys.path.insert(0, "src")
    from model import ASLResNet
    
    model = ASLResNet(num_classes=num_classes, pretrained=True)
    
    # FREEZE the ResNet backbone - only train LSTM + FC head
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    model = model.to(device)
    
    # Datasets
    train_ds = AugmentedVideoDataset("data/index.csv", split="train", num_frames=16)
    val_ds = AugmentedVideoDataset("data/index.csv", split="val", num_frames=16)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    log_lines = []
    
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Classes: {num_classes} -> {list(label_map.keys())}")
    log_lines.append(f"Trainable params: {trainable:,} / {total:,}")
    log_lines.append(f"Train samples (virtual): {len(train_ds)}, Val: {len(val_ds)}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
            train_total += inputs.size(0)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
        
        scheduler.step()
        
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        
        line = f"Epoch {epoch+1:3d}/{epochs} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}"
        log_lines.append(line)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            log_lines.append(f"  >> SAVED best model (val_acc={val_acc:.4f})")
        
        # Write progress file every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with open("_training_log.txt", "w") as f:
                f.write("\n".join(log_lines))
    
    log_lines.append(f"\nBest Val Accuracy: {best_acc:.4f}")
    with open("_training_log.txt", "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    train_smart(epochs=100, batch_size=4, lr=1e-3)
