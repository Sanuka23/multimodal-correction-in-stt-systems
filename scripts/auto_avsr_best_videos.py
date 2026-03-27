#!/usr/bin/env python3
"""Run Auto-AVSR lip reading on the 4 best videos (>50% face detection).

Only processes timestamps where MediaPipe detected faces.
Produces actual lip-read text instead of just confidence scores.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Videos with best face detection rates
BEST_VIDEOS = {
    "business_discussion": {
        "video": "/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/New/Screen App Business Discussion- Jim and Matt.mp4",
        "face_rate": 0.67,
    },
    "compliance_discussion": {
        "video": "/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/New/Compliance Discussion- Matt and Tim.mp4",
        "face_rate": 0.53,
    },
    "screenapp_migration_kimi": {
        "video": "/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/ScreenApp Workload Migration to Kimi.mp4",
        "face_rate": 0.50,
    },
    "followup_julien": {
        "video": "/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/Followup onboarding call with Julien.mp4",
        "face_rate": 0.47,
    },
}

AVSR_CACHE_DIR = Path("data/avsr_cache")


def run_auto_avsr_on_video(video_path: str, cache_name: str):
    """Run Auto-AVSR on segments where faces were detected."""
    from asr_correction.avsr import get_avsr_provider

    # Load existing MediaPipe cache to know which timestamps have faces
    mp_cache_path = AVSR_CACHE_DIR / f"{cache_name}.json"
    with open(mp_cache_path) as f:
        mp_cache = json.load(f)

    # Find timestamps with face detection
    face_timestamps = []
    for ts_str, hint in mp_cache.items():
        if "No face" not in hint:
            face_timestamps.append(float(ts_str))

    if not face_timestamps:
        print(f"  No face timestamps found, skipping")
        return

    print(f"  {len(face_timestamps)} timestamps with faces")

    # Initialize Auto-AVSR
    provider = get_avsr_provider("auto_avsr", model_dir="./models/auto_avsr")
    if provider is None:
        print(f"  Auto-AVSR not available, keeping MediaPipe hints")
        return

    # Process each face timestamp — use 10s windows
    updated_cache = dict(mp_cache)  # Start with MediaPipe hints
    success = 0
    failed = 0

    # Sample every 3rd timestamp to keep it fast
    sampled = face_timestamps[::3]
    print(f"  Processing {len(sampled)} timestamps (sampled from {len(face_timestamps)})...")

    for i, ts in enumerate(sampled):
        start_s = max(0, ts - 5)
        end_s = ts + 5

        try:
            hint = provider.analyze_segment(video_path, start_s, end_s)
            if hint and hint.lip_transcript and len(hint.lip_transcript.strip()) > 5:
                # Replace MediaPipe hint with actual lip-read text
                updated_cache[str(ts)] = f"Visual speech suggests: '{hint.lip_transcript}' (confidence: {hint.speaking_confidence:.2f})"
                success += 1
            # else keep the MediaPipe hint
        except Exception as e:
            failed += 1
            continue

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sampled)} done ({success} lip-read, {failed} failed)")

    # Save updated cache
    with open(mp_cache_path, "w") as f:
        json.dump(updated_cache, f, indent=2)

    print(f"  Done: {success} lip-read text, {failed} failed, {len(sampled)-success-failed} kept MediaPipe")


def main():
    print("=== Auto-AVSR on Best Videos ===\n")

    for name, info in BEST_VIDEOS.items():
        video_path = info["video"]
        if not Path(video_path).exists():
            print(f"[{name}] Video not found: {video_path}")
            continue

        size_mb = Path(video_path).stat().st_size / 1e6
        print(f"[{name}] ({size_mb:.0f} MB, {info['face_rate']*100:.0f}% face detection)")
        run_auto_avsr_on_video(video_path, name)
        print()

    print("=== Complete ===")


if __name__ == "__main__":
    main()
