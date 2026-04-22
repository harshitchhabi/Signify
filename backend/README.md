# Signify ML Backend

This directory contains the machine learning pipeline for the Signify ASL Translator.

---

## Setup

1.  **Create & activate virtual environment** (one time):
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## STEP 1: Data Collection & Landmark Extraction

Before training any model, we need clean hand landmark data. This step downloads ASL sign videos and extracts hand positions using MediaPipe.

### 1a. Download WLASL Videos

First, get the WLASL dataset index:
1. Go to [WLASL GitHub](https://github.com/dxli94/WLASL)
2. Download `WLASL_v0.3.json`
3. Place it at `backend/data/WLASL_v0.3.json`

Then run the download script:
```bash
cd backend
python preprocessing/download_wlasl.py
```

This downloads ~150-250 short ASL videos for 10 common signs into `data/raw_videos/`.

> **Alternatively:** You can record your own videos! Place `.mp4` files into
> `data/raw_videos/{sign_name}/` folders (e.g., `data/raw_videos/hello/hello_01.mp4`).

### 1b. Extract Hand Landmarks

Once you have videos, extract MediaPipe hand landmarks:
```bash
cd backend
python preprocessing/extract_landmarks.py
```

This produces:
- `data/landmarks/{sign}/*.npy` — 21-point hand landmarks per video, shape `(30, 21, 3)`
- `data/labels.csv` — master file mapping videos → labels → splits
- `data/label_map.json` — sign name → numeric ID mapping

### Data Structure After STEP 1

```
backend/data/
├── raw_videos/           # Downloaded/recorded sign videos
│   ├── hello/
│   │   ├── hello_01.mp4
│   │   └── ...
│   ├── thank_you/
│   └── ... (10 sign folders)
├── landmarks/            # Extracted .npy landmark files
│   ├── hello/
│   │   ├── hello_01.npy  # shape: (30, 21, 3)
│   │   └── ...
│   ├── thank_you/
│   └── ... (10 sign folders)
├── labels.csv            # Master mapping with train/val/test splits
├── label_map.json        # {"hello": 0, "thank_you": 1, ...}
└── WLASL_v0.3.json       # WLASL dataset index (you download this)
```

### Configuration

All pipeline settings are in `preprocessing/config.py`:
- `TARGET_SIGNS` — which 10 signs to use
- `NUM_FRAMES` — frames per video (default: 30)
- `MAX_VIDEOS_PER_SIGN` — download limit (default: 25)
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` — data split ratios

---

## STEP 2: Train MLP Classifier

Train a simple MLP (neural network) to classify ASL signs from the landmark data.

### 2a. Train the Model

```bash
cd backend
python training/train.py
```

This will:
- Load `.npy` landmark files and normalize them (wrist-relative positioning)
- Train a 3-layer MLP: Input(1890) → 512 → 256 → 128 → Output(10)
- Print epoch-by-epoch progress with train/val accuracy
- Save the best model to `checkpoints/best_model.pth`
- Stop early if the model stops improving

### 2b. Evaluate on Test Set

```bash
cd backend
python training/evaluate.py
```

This prints:
- Overall test accuracy
- Per-class accuracy for each sign
- Confusion matrix showing which signs get confused
- Debugging advice based on your results

### Training Configuration

All training settings are in `training/config.py`:
- `LEARNING_RATE` — default: 0.001
- `BATCH_SIZE` — default: 32
- `MAX_EPOCHS` — default: 100 (early stopping will end sooner)
- `HIDDEN_SIZES` — default: [512, 256, 128]
- `DROPOUT_RATE` — default: 0.3

---

## API Server

Start the FastAPI server to serve predictions:
```bash
python src/main.py
```
The API will be available at `http://localhost:8000`.
