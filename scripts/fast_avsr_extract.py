#!/usr/bin/env python3
"""Fast AVSR extraction from videos using MediaPipe FaceMesh.

Samples frames every 5 seconds, detects faces and speaking activity.
Saves hint strings to cache for training pair generation.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_MAP = {
    "AWS Migration Meeting": "aws_migration",
    "Followup onboarding call with Julien": "followup_julien",
    "ScreenApp Workload Migration to Kimi": "screenapp_migration_kimi",
    "Troubleshooting call with Dimiter Ivanov": "troubleshooting_dimiter",
    "Google Meet Recording": "google_meet",
    "Compliance Discussion": "compliance_discussion",
    "GCP Security Command Center": "gcp_security",
    "Onboarding Meeting": "onboarding_andre",
    "Project Update Meeting": "project_update",
    "Screen App Business Discussion": "business_discussion",
    "Zachary Jacobson Onboarding": "zachary_onboarding",
}

VIDEO_DIRS = [
    Path("/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter"),
    Path("/Users/sanukathamuditha/Desktop/FYP/Tests/sample video for tter/New"),
]

OUTPUT_DIR = Path("data/avsr_cache")
FRAME_INTERVAL = 5  # seconds


def match_video_name(filename: str) -> str | None:
    for pattern, name in VIDEO_MAP.items():
        if pattern.lower() in filename.lower():
            return name
    return None


def extract_avsr_hints(video_path: str, interval_s: int = 5) -> dict:
    """Extract AVSR hints using MediaPipe FaceMesh."""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    cache = {}

    for ts in range(0, int(duration), interval_s):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cache[str(float(ts))] = "No face detected in video segment"
            continue

        num_faces = len(results.multi_face_landmarks)

        # Compute MAR for each face
        mars = []
        for fl in results.multi_face_landmarks:
            lm = fl.landmark
            upper, lower = lm[13], lm[14]
            left, right = lm[78], lm[308]
            vert = ((upper.x - lower.x)**2 + (upper.y - lower.y)**2)**0.5
            horiz = ((left.x - right.x)**2 + (left.y - right.y)**2)**0.5
            mar = vert / horiz if horiz > 1e-6 else 0.0
            mars.append(mar)

        max_mar = max(mars) if mars else 0.0

        if num_faces > 1:
            hint = f"Multiple speakers visible ({num_faces} faces detected)"
        elif max_mar > 0.15:
            confidence = min(max_mar / 0.3, 1.0)
            hint = f"Speaker detected, actively speaking (confidence: {confidence:.2f})"
        elif max_mar > 0.05:
            hint = f"Speaker detected, low speaking activity (confidence: {max_mar/0.3:.2f})"
        else:
            hint = f"Face detected but mouth closed"

        cache[str(float(ts))] = hint

    cap.release()
    face_mesh.close()
    return cache


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = []
    for vdir in VIDEO_DIRS:
        if not vdir.exists():
            continue
        for mp4 in vdir.glob("*.mp4"):
            cache_name = match_video_name(mp4.stem)
            if cache_name is None:
                continue
            cache_file = OUTPUT_DIR / f"{cache_name}.json"
            if cache_file.exists():
                print(f"  {cache_name}: cache exists, skipping")
                continue
            videos.append((mp4, cache_name))

    print(f"\n=== Fast AVSR Extraction ({len(videos)} videos) ===\n")

    for video_path, cache_name in videos:
        size_mb = video_path.stat().st_size / 1e6
        print(f"[{cache_name}] ({size_mb:.0f} MB)...", end=" ", flush=True)

        cache = extract_avsr_hints(str(video_path), FRAME_INTERVAL)

        cache_file = OUTPUT_DIR / f"{cache_name}.json"
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        face_count = sum(1 for h in cache.values() if "face" in h.lower() or "speaker" in h.lower())
        print(f"{len(cache)} timestamps ({face_count} with faces)")

    print(f"\n=== AVSR Complete ===")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
