# Signify – ASL Translator

This project is a Sign Language Recognition system built to translate ASL gestures into text.

---

## 🧩 Data Pipeline Module (Contribution by Saarthi Bishnoi)

I worked on the data preprocessing and feature engineering pipeline for this project.

### 🔧 Responsibilities:
- Filtered WLASL dataset for relevant classes  
- Extracted video URLs from dataset metadata  
- Downloaded and cleaned video data  
- Converted videos into frames using OpenCV  
- Extracted hand landmarks using MediaPipe  
- Converted frames into structured feature vectors  
- Prepared dataset for machine learning models  

---

## 📊 Dataset Details

- Each frame → 63 features (21 landmarks × x,y,z)  
- Each video → 30 frames  
- Final input → 1890 features per sample  

---

## ⚙️ Tech Used

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Scikit-learn  

---

## 👥 Team Contribution

This project was developed collaboratively, with different modules including frontend, backend, and data pipeline.
