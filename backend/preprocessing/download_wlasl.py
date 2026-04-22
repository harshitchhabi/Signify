"""
Signify STEP 1 — Download WLASL Videos
=======================================
Downloads ASL sign videos from the WLASL dataset for our target signs.

HOW IT WORKS:
  1. Reads the WLASL JSON file (which maps words → YouTube video URLs)
  2. For each of our 10 target signs, finds matching video entries
  3. Downloads each video using yt-dlp (a YouTube downloader)
  4. Saves videos into: data/raw_videos/{sign_name}/{sign_name}_01.mp4

BEFORE RUNNING:
  - Download WLASL_v0.3.json from https://github.com/dxli94/WLASL
    and place it at: backend/data/WLASL_v0.3.json
  - Install yt-dlp: pip install yt-dlp

USAGE:
  cd backend
  python preprocessing/download_wlasl.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Import our config so all paths and settings come from one place
from config import (
    WLASL_JSON,
    RAW_VIDEO_DIR,
    TARGET_SIGNS,
    MAX_VIDEOS_PER_SIGN,
)


def load_wlasl_json(json_path: Path) -> list:
    """
    Load the WLASL dataset JSON file.

    The JSON structure looks like:
    [
      {
        "gloss": "hello",          ← the English word
        "instances": [
          {
            "video_id": "abc123",  ← YouTube video ID
            "url": "https://...",  ← full YouTube URL
            ...
          },
          ...
        ]
      },
      ...
    ]
    """
    if not json_path.exists():
        print(f"ERROR: WLASL JSON file not found at: {json_path}")
        print()
        print("To fix this:")
        print("  1. Go to https://github.com/dxli94/WLASL")
        print("  2. Download WLASL_v0.3.json")
        print(f"  3. Place it at: {json_path}")
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded WLASL JSON with {len(data)} sign entries.")
    return data


def find_sign_entries(wlasl_data: list, sign_word: str) -> list:
    """
    Find all video entries for a given sign word in the WLASL data.

    Returns a list of instance dictionaries, each containing a 'url' key.
    """
    for entry in wlasl_data:
        # WLASL uses "gloss" to mean the English word for the sign
        if entry["gloss"].lower() == sign_word.lower():
            return entry.get("instances", [])

    return []


def download_video(url: str, output_path: Path) -> bool:
    """
    Download a single video from YouTube using yt-dlp.

    Returns True if download succeeded, False otherwise.

    WHY yt-dlp?
    - It's the most reliable YouTube downloader
    - It handles different video formats automatically
    - It can convert to mp4 format
    """
    try:
        # Build the yt-dlp command
        command = [
            "python3", "-m", "yt_dlp",
            "--quiet",                    # Don't print progress bars
            "--no-warnings",              # Don't print warnings
            "--format", "mp4",            # Download as mp4 format
            "--output", str(output_path), # Where to save
            url,
        ]

        # Run the download command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,  # Give up after 60 seconds
        )

        if result.returncode == 0 and output_path.exists():
            return True
        else:
            return False

    except subprocess.TimeoutExpired:
        print(f"    ⏰ Download timed out for: {url}")
        return False
    except FileNotFoundError:
        print("ERROR: yt-dlp is not installed!")
        print("  Install it with: pip install yt-dlp")
        sys.exit(1)
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return False


def download_sign_videos(wlasl_data: list, sign_word: str) -> int:
    """
    Download all available videos for one sign word.

    Creates a folder like: data/raw_videos/hello/
    Downloads videos as:   hello_01.mp4, hello_02.mp4, etc.

    Returns the number of successfully downloaded videos.
    """
    # Create the output folder for this sign
    # Replace spaces with underscores for folder names (e.g., "thank you" → "thank_you")
    safe_name = sign_word.replace(" ", "_")
    sign_folder = RAW_VIDEO_DIR / safe_name
    sign_folder.mkdir(parents=True, exist_ok=True)
    
    search_word = sign_word.replace("_", " ")

    # Find all video entries for this sign in WLASL
    instances = find_sign_entries(wlasl_data, search_word)

    if not instances:
        print(f"  ⚠️  No entries found in WLASL for '{sign_word}'")
        return 0

    print(f"  Found {len(instances)} video entries, downloading up to {MAX_VIDEOS_PER_SIGN}...")

    downloaded = 0
    skipped = 0

    for i, instance in enumerate(instances):
        # Stop if we have enough videos
        if downloaded >= MAX_VIDEOS_PER_SIGN:
            break

        url = instance.get("url", "")
        if not url:
            skipped += 1
            continue

        # Build the output filename: hello_01.mp4, hello_02.mp4, etc.
        video_filename = f"{safe_name}_{downloaded + 1:02d}.mp4"
        output_path = sign_folder / video_filename

        # Skip if already downloaded
        if output_path.exists():
            print(f"    ✅ Already exists: {video_filename}")
            downloaded += 1
            continue

        # Try to download
        success = download_video(url, output_path)

        if success:
            downloaded += 1
            print(f"    ✅ Downloaded: {video_filename}")
        else:
            skipped += 1
            # Don't print for every failure — just count them

    print(f"  → {downloaded} downloaded, {skipped} skipped/failed")
    return downloaded


def main():
    """
    Main function: downloads WLASL videos for all target signs.
    """
    print("=" * 60)
    print("SIGNIFY — STEP 1: Download WLASL Videos")
    print("=" * 60)
    print()

    # Step 1: Load the WLASL dataset JSON
    wlasl_data = load_wlasl_json(WLASL_JSON)

    # Step 2: Create the raw_videos directory
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving videos to: {RAW_VIDEO_DIR}")
    print()

    # Step 3: Download videos for each target sign
    total_downloaded = 0
    summary = {}

    for sign_word in TARGET_SIGNS:
        print(f"[{sign_word.upper()}]")
        count = download_sign_videos(wlasl_data, sign_word)
        summary[sign_word] = count
        total_downloaded += count
        print()

    # Step 4: Print summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for sign_word, count in summary.items():
        status = "✅" if count >= 15 else "⚠️ " if count >= 5 else "❌"
        print(f"  {status} {sign_word:15s} → {count} videos")
    print(f"\n  Total: {total_downloaded} videos downloaded")
    print()

    if total_downloaded == 0:
        print("No videos were downloaded! Check your internet connection")
        print("and make sure WLASL_v0.3.json is in the right place.")
    elif total_downloaded < len(TARGET_SIGNS) * 10:
        print("Some signs have fewer videos than ideal.")
        print("This is normal — some YouTube videos get deleted over time.")
        print("You can supplement with self-recorded videos if needed.")
    else:
        print("Download complete! You can now run extract_landmarks.py")


if __name__ == "__main__":
    main()
