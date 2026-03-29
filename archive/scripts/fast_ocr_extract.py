#!/usr/bin/env python3
"""Fast OCR extraction from videos using PaddleOCR.

Extracts 1 frame every 30 seconds, runs OCR, saves to cache.
Processes all videos that don't have OCR cache yet.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Video → cache name mapping
VIDEO_MAP = {
    "Followup onboarding call with Julien": "followup_julien",
    "Troubleshooting call with Dimiter Ivanov": "troubleshooting_dimiter",
    "Google Meet Recording": "google_meet",
    "Compliance Discussion": "compliance_discussion",
    "GCP Security Command Center": "gcp_security",
    "Onboarding Meeting": "onboarding_andre",
    "Project Update Meeting": "project_update",
    "Zachary Jacobson Onboarding": "zachary_onboarding",
}

# Videos that already have OCR
ALREADY_DONE = {"aws_migration", "business_discussion", "screenapp_migration_kimi"}

VIDEO_DIRS = [
    Path("/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter"),
    Path("/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/New"),
]

OUTPUT_DIR = Path("data/ocr_cache")
FRAME_INTERVAL = 30  # seconds between frames

# Noise filters
NOISE = [
    "unmute", "mute my", "⌘", "start video", "stop video",
    "participants", "share screen", "record", "reactions",
    "end meeting", "security", "breakout", "connecting to audio",
    "phone call", "computer audio", "join audio", "leave meeting",
    "you are muted", "meeting id", "passcode", "this meeting is being",
    "turn on microphone", "let everyone send", "join later",
    "when you leave the call",
]


def match_video_name(filename: str) -> str | None:
    """Match video filename to cache name."""
    for pattern, name in VIDEO_MAP.items():
        if pattern.lower() in filename.lower():
            return name
    return None


def extract_frames(video_path: str, interval_s: int = 30) -> list:
    """Extract frames at regular intervals using cv2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames = []
    for ts in range(0, int(duration), interval_s):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((ts, frame))

    cap.release()
    return frames


def run_ocr_on_frames(frames: list) -> dict:
    """Run PaddleOCR on extracted frames. Returns {timestamp: [text_lines]}."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False,
                    use_gpu=False, det_model_dir=None, rec_model_dir=None)

    cache = {}
    for ts, frame in frames:
        try:
            result = ocr.ocr(frame, cls=False)
            if not result or not result[0]:
                continue

            lines = []
            for line_result in result[0]:
                if len(line_result) >= 2:
                    text = line_result[1][0] if isinstance(line_result[1], (list, tuple)) else str(line_result[1])
                    confidence = line_result[1][1] if isinstance(line_result[1], (list, tuple)) and len(line_result[1]) > 1 else 0.5

                    if confidence < 0.5:
                        continue
                    if len(text.strip()) < 3:
                        continue
                    if any(n in text.lower() for n in NOISE):
                        continue
                    lines.append(text.strip())

            if lines:
                cache[str(float(ts))] = lines

        except Exception as e:
            continue

    return cache


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    videos_to_process = []
    for vdir in VIDEO_DIRS:
        if not vdir.exists():
            continue
        for mp4 in vdir.glob("*.mp4"):
            cache_name = match_video_name(mp4.stem)
            if cache_name is None:
                continue
            if cache_name in ALREADY_DONE:
                print(f"  {cache_name}: already has OCR, skipping")
                continue
            cache_file = OUTPUT_DIR / f"{cache_name}.json"
            if cache_file.exists():
                print(f"  {cache_name}: cache exists, skipping")
                continue
            videos_to_process.append((mp4, cache_name))

    print(f"\n=== Fast OCR Extraction ({len(videos_to_process)} videos) ===\n")

    # Initialize PaddleOCR once
    print("Loading PaddleOCR...")
    import os
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(lang='en')
    print("PaddleOCR ready\n")

    for video_path, cache_name in videos_to_process:
        size_mb = video_path.stat().st_size / 1e6
        print(f"[{cache_name}] {video_path.name} ({size_mb:.0f} MB)")

        # Extract frames
        print(f"  Extracting frames (1 every {FRAME_INTERVAL}s)...", end=" ", flush=True)
        frames = extract_frames(str(video_path), FRAME_INTERVAL)
        print(f"{len(frames)} frames")

        if not frames:
            print(f"  SKIP — no frames extracted")
            continue

        # Run OCR
        print(f"  Running PaddleOCR...", end=" ", flush=True)
        cache = {}
        for ts, frame in frames:
            try:
                result = ocr_engine.ocr(frame, cls=False)
                if not result or not result[0]:
                    continue

                lines = []
                for line_result in result[0]:
                    if len(line_result) >= 2:
                        text = line_result[1][0] if isinstance(line_result[1], (list, tuple)) else str(line_result[1])
                        conf = line_result[1][1] if isinstance(line_result[1], (list, tuple)) and len(line_result[1]) > 1 else 0.5

                        if conf < 0.5 or len(text.strip()) < 3:
                            continue
                        if any(n in text.lower() for n in NOISE):
                            continue
                        lines.append(text.strip())

                if lines:
                    cache[str(float(ts))] = lines
            except Exception:
                continue

        # Save cache
        cache_file = OUTPUT_DIR / f"{cache_name}.json"
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        total_lines = sum(len(v) for v in cache.values())
        print(f"{len(cache)} timestamps, {total_lines} text lines")
        print()

    print("=== OCR Extraction Complete ===")
    print(f"Caches in: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        print(f"  {f.name}: {len(data)} timestamps")


if __name__ == "__main__":
    main()
