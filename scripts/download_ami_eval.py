#!/usr/bin/env python3
"""Download AMI Meeting Corpus for evaluation.

Downloads the AMI test split from HuggingFace, extracts audio files
and manual ground truth transcripts, and saves in eval dataset format.

Usage:
    python scripts/download_ami_eval.py --output data/eval_dataset --max-meetings 20
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Download AMI Corpus for evaluation")
    parser.add_argument("--output", default="data/eval_dataset")
    parser.add_argument("--max-meetings", type=int, default=20)
    parser.add_argument("--split", default="test", help="Dataset split: train/validation/test")
    args = parser.parse_args()

    output_dir = Path(args.output)
    audio_dir = output_dir / "ami_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    print("Loading AMI dataset from HuggingFace...")
    from datasets import load_dataset

    # Try different AMI dataset versions on HuggingFace
    # ihm = Individual Headset Mic, sdm = Single Distant Mic
    ami = None
    for dataset_name, config in [
        ("edinburghcstr/ami", "ihm"),
        ("edinburghcstr/ami", "sdm"),
        ("knkarthick/AMI", None),
    ]:
        try:
            kwargs = {"split": args.split}
            if config:
                ami = load_dataset(dataset_name, config, **kwargs)
            else:
                ami = load_dataset(dataset_name, **kwargs)
            print(f"Loaded {len(ami)} segments from {dataset_name} ({config or 'default'}) {args.split} split")
            break
        except Exception as e:
            print(f"  {dataset_name}/{config}: {e}")
            continue

    if ami is None:
        print("ERROR: Could not load AMI dataset from any source")
        sys.exit(1)

    # Group segments by meeting_id
    meetings = defaultdict(list)
    for item in ami:
        meeting_id = item.get("meeting_id", item.get("id", "unknown"))
        meetings[meeting_id].append(item)

    print(f"Found {len(meetings)} unique meetings")

    # Process each meeting
    manifest = []
    collected = 0

    for meeting_id, segments in sorted(meetings.items()):
        if collected >= args.max_meetings:
            break

        print(f"\n[{collected + 1}/{args.max_meetings}] Meeting: {meeting_id} ({len(segments)} segments)")

        # Collect ground truth text from all segments
        gt_parts = []
        for seg in segments:
            text = seg.get("text", "")
            if text:
                speaker = seg.get("speaker_id", "SPEAKER")
                start = seg.get("begin_time", seg.get("start", 0))
                end = seg.get("end_time", seg.get("end", 0))
                gt_parts.append({
                    "text": text.strip(),
                    "speaker": speaker,
                    "start": float(start) if start else 0,
                    "end": float(end) if end else 0,
                })

        if not gt_parts:
            print(f"  SKIP — no transcript text")
            continue

        gt_text = " ".join(p["text"] for p in gt_parts)
        print(f"  Ground truth: {len(gt_text)} chars, {len(gt_parts)} segments")

        # Save audio file (from first segment that has audio)
        audio_path = audio_dir / f"{meeting_id}.wav"
        if not audio_path.exists():
            # Try to get audio from the dataset
            audio_saved = False
            for seg in segments:
                if "audio" in seg and seg["audio"] is not None:
                    audio_data = seg["audio"]
                    if isinstance(audio_data, dict) and "array" in audio_data:
                        sf.write(str(audio_path), audio_data["array"], audio_data["sampling_rate"])
                        audio_saved = True
                        size_mb = audio_path.stat().st_size / 1e6
                        print(f"  Audio saved: {audio_path.name} ({size_mb:.1f} MB)")
                        break
                    elif isinstance(audio_data, dict) and "path" in audio_data:
                        # Audio is a file path
                        import shutil
                        src = audio_data["path"]
                        if os.path.exists(src):
                            shutil.copy2(src, audio_path)
                            audio_saved = True
                            print(f"  Audio copied: {audio_path.name}")
                            break

            if not audio_saved:
                print(f"  WARNING — no audio data available, saving text only")
        else:
            print(f"  Audio exists: {audio_path.name}")

        # Determine accent (AMI has mostly European non-native speakers)
        # AMI speakers are from various countries — approximate from meeting IDs
        speakers = set(p["speaker"] for p in gt_parts)
        accent = "european"  # Default for AMI (mostly non-native English)

        # Save ground truth
        gt_file = transcripts_dir / f"ami_{meeting_id}_ground_truth.txt"
        gt_file.write_text(gt_text)

        # Save segments as JSON
        gt_segments_file = transcripts_dir / f"ami_{meeting_id}_segments.json"
        with open(gt_segments_file, "w") as f:
            json.dump(gt_parts, f, indent=2)

        entry = {
            "video_id": f"ami_{meeting_id}",
            "title": f"AMI Meeting {meeting_id}",
            "accent": accent,
            "domain": "meeting",
            "audio_path": str(audio_path) if audio_path.exists() else "",
            "video_path": "",  # AMI video needs separate download
            "ground_truth_file": str(gt_file),
            "ground_truth_chars": len(gt_text),
            "speakers": list(speakers),
            "num_segments": len(gt_parts),
        }
        manifest.append(entry)
        collected += 1

    # Save manifest
    manifest_file = output_dir / "ami_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== AMI Download Complete ===")
    print(f"Meetings: {collected}")
    print(f"Manifest: {manifest_file}")
    print(f"Audio dir: {audio_dir}")
    print(f"\nNext step: Upload audio to ScreenApp:")
    print(f"  python training/screenapp_transcribe.py \\")
    print(f"    --audio-dir {audio_dir} \\")
    print(f"    --output {transcripts_dir}/ami_transcripts.jsonl \\")
    print(f"    --limit {collected}")


if __name__ == "__main__":
    main()
