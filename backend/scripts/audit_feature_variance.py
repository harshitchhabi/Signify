import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Use the same exact setup
from step3_lstm import LSTMDataset, RANDOM_SEED

BATCH_SIZE = 32
REPORT_PATH = Path("reports/feature_variance_audit_report.txt")
REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)

# Also load the micro-overfit data using its same exact logic to be accurate
def load_micro_data():
    from step3d_micro_overfit import load_data
    X, y, _ = load_data()
    return X

def analyze_dataset(tensor_data, name):
    """
    Analyzes a tensor of shape (N, 30, 63)
    """
    N, frames, features = tensor_data.shape
    
    overall_mean = tensor_data.mean().item()
    overall_std = tensor_data.std().item()
    overall_min = tensor_data.min().item()
    overall_max = tensor_data.max().item()
    
    # Calculate frame-to-frame variance 
    # (Difference between consecutive frames, then variance/std of those differences)
    frame_diffs = tensor_data[:, 1:, :] - tensor_data[:, :-1, :]
    motion_std = frame_diffs.std().item()
    
    return {
        "name": name,
        "N": N,
        "mean": overall_mean,
        "std": overall_std,
        "min": overall_min,
        "max": overall_max,
        "motion_std": motion_std,
        "data": tensor_data
    }

def format_analysis(stats):
    content = f"--- {stats['name'].upper()} STATISTICS ({stats['N']} samples) ---\n"
    content += f"Mean: {stats['mean']:.6f}\n"
    content += f"Std:  {stats['std']:.6f}\n"
    content += f"Min:  {stats['min']:.6f}\n"
    content += f"Max:  {stats['max']:.6f}\n"
    content += f"Motion Std (Frame-to-Frame variance): {stats['motion_std']:.6f}\n"
    return content

def run_feature_audit():
    print("Starting Feature Variance Audit...")
    report = []
    
    def log(msg):
        print(msg)
        report.append(msg)

    log("============================================================")
    log("COMPLETE FEATURE VARIANCE AUDIT")
    log("============================================================\n")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load micro data
    micro_tensor = load_micro_data()
    micro_stats = analyze_dataset(micro_tensor, "Micro-Overfit Dataset")
    
    # 2. Load full train data
    train_ds = LSTMDataset("train")
    # Collect all train data into a single tensor
    all_train_features = []
    for i in range(len(train_ds)):
        feat, _ = train_ds[i]
        all_train_features.append(feat)
    train_tensor = torch.stack(all_train_features)
    train_stats = analyze_dataset(train_tensor, "Full Training Dataset")
    
    # 3. Random 10 samples from train
    indices = torch.randperm(len(train_tensor))[:10]
    sample_tensor = train_tensor[indices]
    
    # Log overall stats
    log("--- OVERALL DATASET STATISTICS ---")
    log(f"Extracted {len(train_ds)} samples from full training set.")
    log(format_analysis(train_stats))
    log(format_analysis(micro_stats))
    
    # Log 10 random samples
    log("--- 10 RANDOM TRAINING SAMPLES ---")
    for i, idx in enumerate(indices):
        sample = train_tensor[idx].unsqueeze(0)
        s_stats = analyze_dataset(sample, f"Sample {i+1} (Index {idx})")
        log(f"Sample {i+1}: Mean={s_stats['mean']:.4f}, Std={s_stats['std']:.4f}, Min={s_stats['min']:.4f}, Max={s_stats['max']:.4f}")
        
    # 5. Verify healthy variance
    log("\n--- VARIANCE HEALTH CHECK ---")
    is_healthy = True
    
    if train_stats['std'] < 1e-4:
        log("FAIL: Overall standard deviation is extremely small (< 1e-4).")
        is_healthy = False
    else:
        log("PASS: Overall standard deviation is healthy.")
        
    if train_stats['motion_std'] < 1e-4:
        log("FAIL: Motion variance (frame-to-frame difference) is virtually zero. Models cannot learn motion.")
        is_healthy = False
    else:
        log("PASS: Motion variance exists (hands are moving).")
        
    if abs(train_stats['mean']) > 1.0 or train_stats['max'] > 10.0 or train_stats['min'] < -10.0:
        log("WARNING: Features are reasonably large. Neural networks prefer inputs mean 0, std 1.")
        # We don't necessarily fail here, but it's a good observation

    # Compare micro vs train
    std_diff = abs(train_stats['std'] - micro_stats['std'])
    if std_diff > 0.1:
        log(f"WARNING: Large difference in Std between Micro ({micro_stats['std']:.4f}) and Full Train ({train_stats['std']:.4f}).")

    log("\n============================================================")
    if is_healthy:
        log("DECISION: PASS")
        log("The input features have sufficient numerical variance.")
        log("They are not completely collapsed or zeroed out.")
        log("Next step: Investigate optimizer, regularization, learning rate, or model capacity.")
    else:
        log("DECISION: FAIL")
        log("Normalization or extraction logic has collapsed the feature space into noise/zeros.")
    log("============================================================")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))
        
    print(f"\nAudit complete. View {REPORT_PATH} for details.")

if __name__ == "__main__":
    run_feature_audit()
