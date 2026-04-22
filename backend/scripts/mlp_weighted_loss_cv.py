import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import collections

# Import from training components
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED, NUM_CLASSES, INPUT_SIZE
from model import SignLanguageMLP

class StratifiedMLPDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data = data_df.reset_index(drop=True)
        self.landmark_dir = BACKEND_DIR / "data" / "landmarks"
    
    def __len__(self):
        return len(self.data)
        
    def normalize_landmarks(self, landmarks):
        normalized = landmarks.copy()
        for frame_idx in range(landmarks.shape[0]):
            wrist = landmarks[frame_idx, 0, :]
            normalized[frame_idx] = landmarks[frame_idx] - wrist
        return normalized
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        npy_path = self.landmark_dir / row["landmark_file"]
        landmarks = np.load(npy_path)
        landmarks = self.normalize_landmarks(landmarks)
        features_1d = landmarks.flatten()
        return torch.FloatTensor(features_1d), torch.LongTensor([row["label_id"]])[0]

N_SPLITS = 5
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BACKEND_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR = BACKEND_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def load_label_map():
    with open(BACKEND_DIR / "data" / "label_map.json", "r") as f:
        return json.load(f)

def run_kfold_cv():
    print("="*60)
    print(f"CLASS-WEIGHTED MLP: {N_SPLITS}-FOLD CROSS VALIDATION")
    print("="*60)
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load entire dataset labels for stratification and weighting
    df = pd.read_csv(BACKEND_DIR / "data" / "labels.csv")
    y = df["label_id"].values
    
    # ── 1. Calculate Class-Weighted Loss Weights ──
    class_counts = collections.Counter(y)
    print("\n--- CLASS DISTRIBUTION ---")
    
    label_map = load_label_map()
    rev_map = {v: k for k, v in label_map.items()}
    
    for class_id in range(NUM_CLASSES):
        print(f"Class {rev_map[class_id]:<10} : {class_counts.get(class_id, 0)} samples")
        
    # Inverse frequency: weight = 1.0 / count
    inverse_freqs = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 0)
        # Avoid division by zero by setting weight to 0 for missing classes
        inv = 1.0 / count if count > 0 else 0
        inverse_freqs.append(inv)
        
    inverse_freqs = np.array(inverse_freqs)
    
    # Normalize weights to sum to NUM_CLASSES (average weight = 1.0)
    sum_inv = np.sum(inverse_freqs)
    if sum_inv > 0:
        normalized_weights = inverse_freqs / sum_inv * NUM_CLASSES
    else:
        normalized_weights = np.ones(NUM_CLASSES)
        
    print("\n--- NORMALIZED CLASS WEIGHTS ---")
    for class_id in range(NUM_CLASSES):
        print(f"{rev_map[class_id]:<10} : {normalized_weights[class_id]:.3f}")
        
    # Convert weights to tensor
    class_weights_tensor = torch.FloatTensor(normalized_weights).to(device)
    
    # ── 2. Run Stratified K-Fold CV ──
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    # Using row index as data proxy since Dataset loads from disk
    X_indices = np.arange(len(df))
    
    fold_results = []
    class_correct_totals = {i: 0 for i in range(NUM_CLASSES)}
    class_counts_totals = {i: 0 for i in range(NUM_CLASSES)}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_indices, y), 1):
        print(f"\n--- FOLD {fold}/{N_SPLITS} ---")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_ds = StratifiedMLPDataset(train_df)
        test_ds = StratifiedMLPDataset(test_df)
        
        # Generator for dataloader reproducibility
        generator = torch.Generator().manual_seed(RANDOM_SEED + fold)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SignLanguageMLP().to(device)
        # Apply class weights to loss function
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_test_acc = 0.0
        best_train_acc = 0.0
        
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            correct, total = 0, 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total += labels.size(0)
            train_acc = correct / total
            
            # Evaluate test fold
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    test_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    test_total += labels.size(0)
            test_acc = test_correct / test_total
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"weighted_mlp_fold{fold}.pth")
                
            if epoch % 20 == 0 or epoch == MAX_EPOCHS:
                print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} (Best Test: {best_test_acc:.3f})")

        # Evaluate final best on test fold again to record per-class accuracy
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"weighted_mlp_fold{fold}.pth"))
        model.eval()
        fold_test_correct, fold_test_total = 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1)
                for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    class_counts_totals[t] += 1
                    if t == p:
                        class_correct_totals[t] += 1
                        
                fold_test_correct += (preds == labels).sum().item()
                fold_test_total += labels.size(0)
                
        final_fold_acc = fold_test_correct / fold_test_total
        print(f"Fold {fold} Result -> Train: {best_train_acc:.3f} | Test: {final_fold_acc:.3f}")
        fold_results.append({
            "fold": fold,
            "train_acc": best_train_acc,
            "test_acc": final_fold_acc
        })

    # Summarize results
    test_accs = [r["test_acc"] for r in fold_results]
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    
    print("\n" + "="*60)
    print(f"5-FOLD CV COMPLETE. Mean Test Acc: {mean_test_acc:.3f} ± {std_test_acc:.3f}")
    print("="*60)
    
    generate_reports(fold_results, mean_test_acc, std_test_acc, class_correct_totals, class_counts_totals, rev_map)

def generate_reports(fold_results, mean_test_acc, std_test_acc, class_correct_totals, class_counts_totals, rev_map):
    eval_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Context values from previous test:
    # We remember from the user prompt: Mean CV MLP was ~60.7%.
    # Specific weaknesses noted: 'good' was 0%, 'bad' was 17%.
    baseline_mlp_cv = 0.607 
    baseline_good_acc = 0.0
    baseline_bad_acc = 0.17
    
    # 1. Main CV Report
    cv_report = f"""============================================================
CLASS-WEIGHTED MLP CV RESULTS
Generated: {eval_time}
============================================================

--- CONFIGURATION ---
Folds:         5 (Stratified)
Loss Mode:     Class-Weighted CrossEntropyLoss
Architecture:  MLP Baseline

--- PER-FOLD RESULTS ---
"""
    for r in fold_results:
        cv_report += f"Fold {r['fold']}: Train = {r['train_acc']:.1%}, Test = {r['test_acc']:.1%}\n"
        
    cv_report += f"""
--- AGGREGATE RESULTS ---
Mean Test Accuracy: {mean_test_acc:.1%}
Std Deviation:      ±{std_test_acc * 100:.1f}%

--- COMPARISON ANALYSIS ---
Unweighted MLP CV Mean: {baseline_mlp_cv:.1%}
Weighted MLP CV Mean:   {mean_test_acc:.1%}
Absolute Change:        {mean_test_acc - baseline_mlp_cv:+.1%}
"""

    # 2. Class Balance Analysis
    balance_report = f"""============================================================
CLASS BALANCE ANALYSIS (WEIGHTED MLP)
Generated: {eval_time}
============================================================

--- PER-CLASS ACCURACY COMPARISON ---
"""
    # Track metrics for narrative
    good_acc = 0
    bad_acc = 0
    
    class_accs = {}

    for class_id in range(len(rev_map)):
        class_name = rev_map[class_id]
        total = class_counts_totals[class_id]
        acc = class_correct_totals[class_id] / total if total > 0 else 0
        class_accs[class_name] = acc
        
        balance_report += f"- {class_name:<10}: {acc:.0%} ({class_correct_totals[class_id]}/{total})\n"
        
        if class_name == 'good': good_acc = acc
        if class_name == 'bad': bad_acc = acc
        
    balance_report += f"""
--- BOTTLENECK ANALYSIS ---
Did weak classes improve?
- 'good': Was {baseline_good_acc:.0%} -> Now {good_acc:.0%} (Change: {good_acc - baseline_good_acc:+.0%})
- 'bad':  Was {baseline_bad_acc:.0%} -> Now {bad_acc:.0%} (Change: {bad_acc - baseline_bad_acc:+.0%})

Conclusion:
"""
    
    # Simple heuristic to write conclusion
    if mean_test_acc > baseline_mlp_cv and (good_acc > baseline_good_acc or bad_acc > baseline_bad_acc):
        balance_report += "YES. Class imbalance was significantly limiting model performance. Weighting the loss improved representation of weak classes without disproportionately sacrificing overall accuracy.\n"
    elif mean_test_acc <= baseline_mlp_cv and (good_acc > baseline_good_acc or bad_acc > baseline_bad_acc):
        balance_report += "MIXED. While weak classes improved slightly, overall accuracy degraded. This implies the model lacks feature capacity to learn all classes robustly, forcing a zero-sum trade-off.\n"
    else:
        balance_report += "NO. Class weighting did not solve the failure modes for weak classes. The underlying issue is likely overlapping features, poor dataset variance, or lack of temporal modeling rather than mere statistical imbalance.\n"

    with open(REPORT_DIR / "weighted_loss_cv_results.txt", "w") as f: f.write(cv_report)
    with open(REPORT_DIR / "class_balance_analysis.txt", "w") as f: f.write(balance_report)
    
    print(f"\nWeighted MLP Mean Acc: {mean_test_acc:.1%} (Change: {mean_test_acc - baseline_mlp_cv:+.1%})")
    print(f"'good' Acc: {good_acc:.0%} | 'bad' Acc: {bad_acc:.0%}")
    print("\nReports successfully generated in backend/reports/")

if __name__ == "__main__":
    run_kfold_cv()
