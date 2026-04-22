<div align="center">
  <h1>🤟 Signify</h1>
  <p><strong>Real-Time American Sign Language (ASL) to English Translator</strong></p>
  
  <p>
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#model-training">Model Training</a>
  </p>
</div>

---

## 🌟 Overview

**Signify** is a cutting-edge web application that bridges the communication gap by translating American Sign Language (ASL) into English in real-time. Built with a modern, responsive UI and a robust PyTorch-based machine learning backend, Signify offers both live camera translation and video upload capabilities.

## ✨ Features

- 📹 **Live Translation**: Real-time camera feed analysis with sub-second latency.
- 📤 **Video Upload**: Upload pre-recorded ASL videos for offline batch translation.
- ⏱️ **Temporal Smoothing**: Employs advanced sliding-window algorithms and majority voting to ensure stable and confident predictions, minimizing visual noise.
- 🎨 **Modern UI**: Sleek, fully responsive interface built with Next.js, Tailwind CSS, and Radix UI. Dark mode supported!
- 🔒 **Privacy-First**: Video feeds are processed locally on the client or sent securely to the backend for inference, maintaining strict user privacy.

## 🏗️ Architecture

Signify is divided into two primary services:

1. **Frontend (Next.js)**: A React-based web app that handles video capture, UI rendering, and communicates with the backend via REST APIs.
2. **Backend (FastAPI & PyTorch)**: A Python service that extracts hand landmarks using MediaPipe and runs a custom trained neural network (MLP/LSTM) to classify ASL signs.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Node.js (v18+)
- Python (3.9+)
- npm or yarn

### 1. Backend Setup

Navigate to the `backend` directory, set up your Python environment, and install dependencies:

```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Start the FastAPI inference server:

```bash
python src/main.py
```
*The API will be available at `http://localhost:8000`.*

### 2. Frontend Setup

In a new terminal window, navigate back to the root directory and install the Node modules:

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

---

## 🧠 Model Pipeline

Signify comes pre-configured with a robust machine learning pipeline. If you wish to retrain the model or add new signs, follow these steps in the `backend` directory:

1. **Data Collection**: Use `preprocessing/download_wlasl.py` to fetch videos from the WLASL dataset or add your own `.mp4` files to `data/raw_videos/`.
2. **Landmark Extraction**: Run `preprocessing/extract_landmarks.py` to extract 3D hand coordinates via MediaPipe.
3. **Training**: Execute `training/train.py` to train the deep learning classifier on the extracted landmarks.
4. **Evaluation**: Use `training/evaluate.py` to generate confusion matrices and test accuracy.

*Refer to the `backend/README.md` for an in-depth guide on the ML pipeline.*

---

## 🛠️ Tech Stack

- **Frontend**: [Next.js](https://nextjs.org/), [React](https://reactjs.org/), [Tailwind CSS](https://tailwindcss.com/), [Framer Motion](https://www.framer.com/motion/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), [MediaPipe](https://google.github.io/mediapipe/), OpenCV
