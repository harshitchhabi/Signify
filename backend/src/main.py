from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os
import tempfile
from datetime import datetime
import cv2
import numpy as np
import time
from inference import SimpleASLClassifier
from collections import deque
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    model_path = "checkpoints/landmark_model.pt"
    label_map_path = "data/label_map.json"
    
    if os.path.exists(model_path) and os.path.exists(label_map_path):
        print("Loading model...")
        classifier = SimpleASLClassifier(model_path, label_map_path)
        print("Model loaded!")
    else:
        print("Warning: Model not found. Please train the model first.")
    
    yield


app = FastAPI(title="Signify ML Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
classifier = None
LANDMARK_BUFFER = deque(maxlen=16)

# Prediction control
LAST_PREDICTED_WORD = None
LAST_PREDICTION_TIME = 0.0
COOLDOWN_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.40

# Voting window: collect recent predictions and pick majority
VOTE_WINDOW = deque(maxlen=5)


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": classifier is not None}


class VideoResponse(BaseModel):
    text: str
    confidence: float
    duration: float
    createdAt: str


@app.post("/api/translate/video")
async def translate_video(video: UploadFile = File(...)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name
        
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        result = classifier.predict_video(tmp_path)
        
        return {
            "text": result["text"],
            "confidence": result["confidence"],
            "duration": duration,
            "createdAt": datetime.now().isoformat()
        }
    finally:
        os.unlink(tmp_path)


@app.post("/api/translate/frame")
async def translate_frame(frame: UploadFile = File(...)):
    global LAST_PREDICTED_WORD, LAST_PREDICTION_TIME
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Decode image
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"text": "", "confidence": 0.0, "partial": True, "timestamp": datetime.now().isoformat()}
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract landmark for just THIS frame
    lm = classifier.extractor.extract_landmarks(img_rgb)
    if lm is not None:
        LANDMARK_BUFFER.append(lm)
    else:
        # Predict zero or copy last frame if no hand is found
        if len(LANDMARK_BUFFER) > 0:
            LANDMARK_BUFFER.append(LANDMARK_BUFFER[-1])
        else:
            LANDMARK_BUFFER.append(np.zeros((63,), dtype=np.float32))
    
    # Need full buffer
    if len(LANDMARK_BUFFER) < 16:
        return {"text": "", "confidence": 0.0, "partial": True, "timestamp": datetime.now().isoformat()}
    
    # Run model directly on the buffered landmarks
    import torch
    tensor = torch.tensor(np.array(list(LANDMARK_BUFFER)), dtype=torch.float32)
    result = classifier._predict_tensor(tensor)
    
    pred_word = result["text"]
    conf = result["confidence"]
    
    current_time = time.time()
    
    # Add to voting window
    VOTE_WINDOW.append((pred_word, conf))
    
    # Majority vote from last 5 predictions
    from collections import Counter
    votes = Counter([w for w, c in VOTE_WINDOW])
    voted_word, vote_count = votes.most_common(1)[0]
    avg_conf = np.mean([c for w, c in VOTE_WINDOW if w == voted_word])
    
    print(f"[Signify] Raw: {pred_word} ({conf*100:.1f}%) | Vote: {voted_word} x{vote_count} ({avg_conf*100:.1f}%)")
    
    # Need at least 3/5 agreement for stability
    if vote_count < 3:
        return {"text": "", "confidence": avg_conf, "partial": True, "timestamp": datetime.now().isoformat()}
    
    # Confidence gate
    if avg_conf < CONFIDENCE_THRESHOLD:
        return {"text": "", "confidence": avg_conf, "partial": True, "timestamp": datetime.now().isoformat()}
    
    # Debounce: don't repeat same word within cooldown
    if voted_word == LAST_PREDICTED_WORD and (current_time - LAST_PREDICTION_TIME) < COOLDOWN_SECONDS:
        return {"text": "", "confidence": avg_conf, "partial": True, "timestamp": datetime.now().isoformat()}
    
    # Emit the word!
    LAST_PREDICTED_WORD = voted_word
    LAST_PREDICTION_TIME = current_time
    VOTE_WINDOW.clear()
    
    return {
        "text": voted_word,
        "confidence": float(avg_conf),
        "partial": False,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
