import json

# Load filtered dataset
with open("filtered_wlasl.json", "r") as f:
    data = json.load(f)

video_urls = []

LIMIT = 20  # number of videos per class

for item in data:
    count = 0
    for instance in item["instances"]:
        if count >= LIMIT:
            break
        url = instance["url"]
        video_urls.append(url)
        count += 1

# Save URLs to file
with open("video_urls.txt", "w") as f:
    for url in video_urls:
        f.write(url + "\n")

print("URLs extracted successfully!")
print("Total URLs:", len(video_urls))