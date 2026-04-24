import json

# Load JSON file
with open("WLASL_v0.3.json", "r") as f:
    data = json.load(f)

# Your target classes
target_words = [
    "hello", "thank_you", "yes", "no", "please",
    "sorry", "help", "good", "bad", "love"
]

filtered_data = []

# Filter data
for item in data:
    if item["gloss"] in target_words:
        filtered_data.append(item)

# Save filtered dataset
with open("filtered_wlasl.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print("Filtered dataset created successfully!")
print("Total classes found:", len(filtered_data))