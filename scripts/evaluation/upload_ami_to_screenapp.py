#!/usr/bin/env python3
"""Upload AMI audio to ScreenApp, get transcripts, compare with ground truth.

Usage:
    # Load .env first
    source .env 2>/dev/null
    export SCREENAPP_PAT_TOKEN SCREENAPP_TEAM_ID SCREENAPP_FOLDER_ID

    python scripts/evaluation/upload_ami_to_screenapp.py
    python scripts/evaluation/upload_ami_to_screenapp.py --skip-upload   # Compare only (if already uploaded)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference, hypothesis):
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        ref_words = len(reference.split())
        return {
            "wer": round(wer * 100, 2),
            "ref_words": ref_words,
            "hyp_words": len(hypothesis.split()),
        }
    except ImportError:
        ref_w = reference.split()
        hyp_w = hypothesis.split()
        # Simple approximation
        errors = abs(len(ref_w) - len(hyp_w))
        return {"wer": round(errors / max(len(ref_w), 1) * 100, 2), "ref_words": len(ref_w), "hyp_words": len(hyp_w)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload, just compare existing")
    parser.add_argument("--max", type=int, default=10, help="Max meetings to process")
    parser.add_argument("--poll-timeout", type=float, default=600, help="Max seconds to wait for transcript")
    args = parser.parse_args()

    # Load .env manually
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    manifest_path = PROJECT_ROOT / "data" / "eval_dataset" / "ami_manifest.json"
    if not manifest_path.exists():
        print("ERROR: Run download_ami_eval.py first")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n=== AMI → ScreenApp Upload & Compare ({len(manifest)} meetings) ===\n")

    results = []

    for i, entry in enumerate(manifest[:args.max]):
        meeting_id = entry["video_id"].replace("ami_", "")
        audio_path = PROJECT_ROOT / "data" / "eval_dataset" / "ami_audio" / f"{meeting_id}.wav"
        gt_file = Path(entry["ground_truth_file"])

        print(f"[{i+1}/{min(len(manifest), args.max)}] {meeting_id}")

        if not audio_path.exists():
            print(f"  SKIP — no audio file")
            continue
        if not gt_file.exists():
            print(f"  SKIP — no ground truth")
            continue

        gt_text = normalize_text(gt_file.read_text())
        print(f"  Ground truth: {len(gt_text.split())} words")

        # Check if already transcribed
        sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "transcripts" / f"ami_{meeting_id}_screenapp.json"

        if not sa_file.exists() and not args.skip_upload:
            # Upload to ScreenApp
            try:
                from training.screenapp_transcribe import ScreenAppTranscriber
                transcriber = ScreenAppTranscriber(max_poll_time=args.poll_timeout)

                size_mb = audio_path.stat().st_size / 1e6
                print(f"  Uploading {size_mb:.1f} MB...")

                result = transcriber.transcribe_file(str(audio_path))
                if result and result.get("transcript_text"):
                    sa_data = {
                        "text": result["transcript_text"],
                        "segments": result.get("segments", []),
                        "file_id": result.get("file_id", meeting_id),
                    }
                    with open(sa_file, "w") as f:
                        json.dump(sa_data, f, indent=2)
                    print(f"  Transcript: {len(sa_data['text'])} chars, {len(sa_data['segments'])} segments")
                    entry["screenapp_file"] = str(sa_file)
                    entry["file_id"] = result.get("file_id")
                else:
                    print(f"  WARN: No transcript returned (timeout?)")
                    continue
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        elif sa_file.exists():
            print(f"  Already transcribed")
            entry["screenapp_file"] = str(sa_file)
        else:
            print(f"  SKIP — no transcript (use without --skip-upload)")
            continue

        # Compare
        if sa_file.exists():
            with open(sa_file) as f:
                sa_data = json.load(f)
            sa_text = normalize_text(sa_data.get("text", ""))

            wer = compute_wer(gt_text, sa_text)
            print(f"  ScreenApp WER: {wer['wer']:.1f}% ({wer['ref_words']} ref / {wer['hyp_words']} hyp words)")

            results.append({
                "meeting_id": meeting_id,
                "title": entry["title"],
                "accent": entry.get("accent", "european"),
                "gt_words": wer["ref_words"],
                "sa_words": wer["hyp_words"],
                "baseline_wer": wer["wer"],
                "audio_path": str(audio_path),
                "gt_file": str(gt_file),
                "screenapp_file": str(sa_file),
                "file_id": entry.get("file_id", ""),
            })
        print()

    # Save results
    if results:
        output_path = PROJECT_ROOT / "data" / "eval_dataset" / "ami_baseline_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        print(f"{'='*60}")
        print(f"SUMMARY ({len(results)} meetings)")
        print(f"{'='*60}")
        print(f"{'Meeting':<12} {'GT Words':>10} {'SA Words':>10} {'WER':>8}")
        print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        for r in results:
            print(f"{r['meeting_id']:<12} {r['gt_words']:>10} {r['sa_words']:>10} {r['baseline_wer']:>7.1f}%")

        avg_wer = sum(r["baseline_wer"] for r in results) / len(results)
        print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        print(f"{'AVERAGE':<12} {'':>10} {'':>10} {avg_wer:>7.1f}%")
        print(f"\nResults saved: {output_path}")

    # Update manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
