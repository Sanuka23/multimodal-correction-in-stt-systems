#!/usr/bin/env python3
"""Step 1: Parse existing OCR from ScreenApp JSON exports.

Extracts OCR XML from 3 JSON files and converts to timestamp→text cache format.
"""

import json
import re
from pathlib import Path

OCR_DIR = Path("/Users/sanukathamuditha/Desktop/FYP/Tests/sample ocr for tter")
OUTPUT_DIR = Path("data/ocr_cache")

# Noise lines to filter out (Zoom UI elements)
NOISE_PATTERNS = [
    "Unmute", "Mute my", "⌘+", "Start Video", "Stop Video",
    "Participants", "Share Screen", "Record", "Reactions",
    "End Meeting", "Chat", "Security", "Breakout",
    "Connecting to audio", "Phone Call", "Computer Audio",
    "Join Audio", "Leave Meeting", "You are muted",
    "Meeting ID:", "Passcode:", "This meeting is being recorded",
]


def parse_timestamp(ts_str: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' to seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def is_noise(line: str) -> bool:
    """Check if an OCR line is Zoom UI noise."""
    for pattern in NOISE_PATTERNS:
        if pattern.lower() in line.lower():
            return True
    # Skip very short lines (single chars, icons)
    if len(line.strip()) < 3:
        return True
    return False


def parse_ocr_xml(xml_text: str) -> dict:
    """Parse OCR XML into {timestamp_seconds: [text_lines]}.

    Handles two formats:
    1. <frame timestamp="00:00"><text>content</text></frame>
    2. <frame timestamp="00:00">content</frame> (no <text> wrapper)
    """
    cache = {}

    # Match frames — capture everything between <frame ...> and </frame>
    frame_pattern = re.compile(
        r'<frame\s+timestamp="([^"]+)"[^>]*>(.*?)</frame>',
        re.DOTALL
    )

    for match in frame_pattern.finditer(xml_text):
        ts_str = match.group(1)
        content = match.group(2)

        ts_seconds = parse_timestamp(ts_str)

        # Strip <text> tags if present
        content = re.sub(r'</?text>', '', content)
        # Strip any other XML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Split into lines and filter noise
        lines = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and not is_noise(line):
                lines.append(line)

        if lines:
            key = str(ts_seconds)
            if key in cache:
                cache[key].extend(lines)
            else:
                cache[key] = lines

    return cache


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(OCR_DIR.glob("*.json"))
    print(f"Found {len(json_files)} OCR JSON files\n")

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        file_id = data.get("fileId", data.get("_id", json_file.stem))
        spr = data.get("systemPromptResponses", {})
        ocr_data = spr.get("OCR", {})
        ocr_xml = ocr_data.get("responseText", "")

        if not ocr_xml:
            print(f"  {json_file.name}: no OCR data, skipping")
            continue

        cache = parse_ocr_xml(ocr_xml)

        # Try to identify the video name from the OCR content
        # Look for participant names that match known videos
        all_text = " ".join(" ".join(v) for v in cache.values())

        # Heuristic video matching based on participant names
        video_name = json_file.stem  # default to file ID
        if "Bud" in all_text and "Jock" in all_text:
            video_name = "business_discussion"
        elif "Mimi" in all_text and "Kieran" in all_text:
            video_name = "aws_migration"
        elif "Kimi" in all_text or "Andre" in all_text:
            video_name = "screenapp_migration_kimi"

        output_file = OUTPUT_DIR / f"{video_name}.json"
        with open(output_file, "w") as f:
            json.dump(cache, f, indent=2)

        print(f"  {json_file.name} → {video_name}.json ({len(cache)} timestamps, "
              f"{sum(len(v) for v in cache.values())} text lines)")

    print(f"\nDone. OCR caches saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
