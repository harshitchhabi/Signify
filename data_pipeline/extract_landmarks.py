import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

frames_folder = "frames"

output_data = []
labels = []

label_map = {}
label_index = 0

# Define your classes
target_words = ["hello","yes","no","please","sorry","help","good","bad","love"]

for video_folder in os.listdir(frames_folder):
    video_path = os.path.join(frames_folder, video_folder)

    # detect label from folder name (we assume dataset order consistency)
    label_name = None
    for word in target_words:
        if word in video_folder.lower():
            label_name = word
            break

    if label_name is None:
        continue

    # assign numeric label
    if label_name not in label_map:
        label_map[label_name] = label_index
        label_index += 1

    frame_files = sorted(os.listdir(video_path))[:30]

    video_features = []

    for frame_file in frame_files:
        frame_path = os.path.join(video_path, frame_file)

        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            frame_features = []
            for lm in landmarks.landmark:
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features = [0] * 63

        video_features.extend(frame_features)

    if len(video_features) == 30 * 63:
        output_data.append(video_features)
        labels.append(label_map[label_name])

# convert to numpy
X = np.array(output_data)
y = np.array(labels)

np.save("dataset.npy", X)
np.save("labels.npy", y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Label mapping:", label_map)