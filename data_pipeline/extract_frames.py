import cv2
import os

video_folder = "videos"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

frame_count = 0

for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    
    cap = cv2.VideoCapture(video_path)
    
    video_name = os.path.splitext(video_file)[0]
    video_frame_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_frame_folder, exist_ok=True)
    
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(video_frame_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        
        count += 1
        frame_count += 1
    
    cap.release()

print("Total frames extracted:", frame_count)