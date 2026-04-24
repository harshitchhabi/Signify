import os

video_folder = "videos"

valid_videos = []
removed = 0

for file in os.listdir(video_folder):
    path = os.path.join(video_folder, file)
    
    # Check file size (remove very small files < 50KB)
    if os.path.getsize(path) < 50 * 1024:
        os.remove(path)
        removed += 1
    else:
        valid_videos.append(file)

print("Valid videos:", len(valid_videos))
print("Removed videos:", removed)