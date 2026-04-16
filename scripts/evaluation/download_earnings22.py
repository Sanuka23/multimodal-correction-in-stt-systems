#!/usr/bin/env python3
"""Download Earnings-22 dataset (human-verified earnings calls).

Downloads audio + NLP reference transcripts from GitHub.
Filters for English-speaking calls only.

Usage:
    python scripts/evaluation/download_earnings22.py --max 10
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22"

GITHUB_RAW = "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings22"
GITHUB_API = "https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings22"


def parse_nlp_transcript(text):
    """Parse .nlp format into plain text."""
    words = []
    for line in text.strip().split("\n"):
        if line.startswith("token|") or not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        word = parts[0]
        punctuation = parts[4] if len(parts) > 4 else ""
        prepunctuation = parts[5] if len(parts) > 5 else ""
        if prepunctuation:
            word = prepunctuation + word
        if punctuation:
            word = word + punctuation
        words.append(word)
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=10, help="Max files to download")
    parser.add_argument("--dialect", default="US", help="Filter by dialect (US, UK, etc.)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)
    transcript_dir = OUTPUT_DIR / "transcripts"
    transcript_dir.mkdir(exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    metadata_url = f"{GITHUB_RAW}/metadata.csv"
    resp = urlopen(metadata_url)
    metadata_text = resp.read().decode("utf-8")
    reader = csv.DictReader(StringIO(metadata_text))
    all_files = list(reader)
    print(f"Total files in dataset: {len(all_files)}")

    # Filter for English calls with Major Dialect matching
    english_files = [
        f for f in all_files
        if args.dialect.lower() in (f.get("Major Dialect Family", "") or "").lower()
        or "english" in (f.get("Language Family + Area Based", "") or "").lower()
    ]
    print(f"English ({args.dialect}) files: {len(english_files)}")

    # Sort by file length (prefer shorter ones for faster processing)
    english_files.sort(key=lambda f: int(f.get("File Length (seconds)", 0) or 0))

    manifest = []
    downloaded = 0

    for entry in english_files[:args.max * 2]:  # Try more in case some fail
        if downloaded >= args.max:
            break

        file_id = entry["File ID"]
        duration_s = int(entry.get("File Length (seconds)", 0) or 0)
        country = entry.get("Country by Ticker", "")
        dialect = entry.get("Major Dialect Family", "")

        # Skip very long files (>30 min)
        if duration_s > 1800:
            continue

        audio_path = audio_dir / f"{file_id}.mp3"
        gt_path = transcript_dir / f"{file_id}_ground_truth.txt"

        print(f"\n[{downloaded+1}/{args.max}] {file_id} ({duration_s}s, {country}, {dialect})")

        # Download transcript first (smaller, faster)
        if not gt_path.exists():
            nlp_url = f"{GITHUB_RAW}/transcripts/nlp_references/{file_id}.nlp"
            try:
                resp = urlopen(nlp_url, timeout=15)
                nlp_text = resp.read().decode("utf-8")
                gt_text = parse_nlp_transcript(nlp_text)
                if len(gt_text) < 100:
                    print(f"  SKIP — transcript too short ({len(gt_text)} chars)")
                    continue
                gt_path.write_text(gt_text)
                print(f"  Transcript: {len(gt_text.split())} words")
            except Exception as e:
                print(f"  SKIP — transcript download failed: {e}")
                continue
        else:
            gt_text = gt_path.read_text()
            print(f"  Transcript: cached ({len(gt_text.split())} words)")

        # Download audio
        if not audio_path.exists():
            # Media files are stored with git-lfs, need to use the download URL
            media_url = f"https://media.githubusercontent.com/media/revdotcom/speech-datasets/main/earnings22/media/{file_id}.mp3"
            print(f"  Downloading audio...")
            try:
                r = subprocess.run(
                    ["curl", "-sL", "-o", str(audio_path), media_url],
                    timeout=120
                )
                if r.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 10000:
                    size_mb = audio_path.stat().st_size / 1e6
                    print(f"  Audio: {size_mb:.1f} MB")
                else:
                    print(f"  SKIP — audio download failed or too small")
                    audio_path.unlink(missing_ok=True)
                    continue
            except subprocess.TimeoutExpired:
                print(f"  SKIP — audio download timeout")
                audio_path.unlink(missing_ok=True)
                continue
        else:
            size_mb = audio_path.stat().st_size / 1e6
            print(f"  Audio: cached ({size_mb:.1f} MB)")

        manifest.append({
            "file_id": file_id,
            "title": f"Earnings Call {file_id}",
            "accent": dialect.lower() or "unknown",
            "country": country,
            "domain": "finance",
            "duration_s": duration_s,
            "audio_path": str(audio_path),
            "ground_truth_file": str(gt_path),
            "gt_words": len(gt_text.split()),
        })
        downloaded += 1

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Downloaded {downloaded} earnings calls")
    print(f"Manifest: {manifest_path}")
    for m in manifest:
        print(f"  {m['file_id']}: {m['gt_words']} words, {m['duration_s']}s, {m['accent']}")


if __name__ == "__main__":
    main()
