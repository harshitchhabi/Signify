import pandas as pd
from collections import Counter
from pathlib import Path

# Config
BACKEND_DIR = Path(__file__).resolve().parent.parent
LABELS_CSV = BACKEND_DIR / "data" / "labels.csv"
REPORT_PATH = BACKEND_DIR / "reports" / "wlasl_expansion_report.txt"

def generate_report():
    print("Generating WLASL Expansion Report...")
    
    if not LABELS_CSV.exists():
        print("ERROR: labels.csv not found!")
        return

    df = pd.read_csv(LABELS_CSV)
    
    total_samples = len(df)
    
    # Calculate samples per sign
    sign_counts = df["sign"].value_counts().to_dict()
    
    # Calculate split counts
    split_counts = df["split"].value_counts().to_dict()
    train_count = split_counts.get("train", 0)
    val_count = split_counts.get("val", 0)
    test_count = split_counts.get("test", 0)
    
    # Hardcoded stats from extraction script
    total_processed = 84
    total_skipped = 25
    total_attempted = total_processed + total_skipped
    
    extraction_success_rate = (total_processed / total_attempted) if total_attempted > 0 else 0
    
    report_content = f"""============================================================
WLASL DATASET EXPANSION REPORT
============================================================

--- SAMPLES PER SIGN (AFTER EXPANSION) ---
"""
    for sign, count in sorted(sign_counts.items()):
        report_content += f"- {sign:<15}: {count} samples\n"
        
    report_content += f"""
--- OVERALL STATISTICS ---
Total Dataset Size: {total_samples} samples
Extraction Success Rate: {extraction_success_rate:.1%} ({total_processed}/{total_attempted} videos)

--- DATA SPLITS ---
Train: {train_count} samples
Validation: {val_count} samples
Test: {test_count} samples
============================================================
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)
        
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
