#!/usr/bin/env python3
"""Upload Earnings-22 audio to ScreenApp, get transcripts, compare with ground truth."""

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(ref, hyp):
    try:
        import jiwer
        w = jiwer.wer(ref, hyp)
        return round(w * 100, 2)
    except ImportError:
        return -1


def main():
    manifest_path = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n=== Earnings-22 → ScreenApp ({len(manifest)} calls) ===\n")

    from training.screenapp_transcribe import ScreenAppTranscriber
    transcriber = ScreenAppTranscriber(max_poll_time=900)

    results = []
    for i, entry in enumerate(manifest):
        fid = entry["file_id"]
        audio = Path(entry["audio_path"])
        gt_file = Path(entry["ground_truth_file"])

        print(f"[{i+1}/{len(manifest)}] {fid} ({entry['accent']}, {entry['country']})")

        if not audio.exists():
            print(f"  SKIP — no audio"); continue
        if not gt_file.exists():
            print(f"  SKIP — no ground truth"); continue

        gt_text = normalize(gt_file.read_text())

        # Check if already transcribed
        sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "transcripts" / f"{fid}_screenapp.json"

        if not sa_file.exists():
            size_mb = audio.stat().st_size / 1e6
            print(f"  Uploading {size_mb:.1f} MB...")
            result = transcriber.transcribe_file(str(audio))
            if result and result.get("transcript_text"):
                sa_data = {
                    "text": result["transcript_text"],
                    "segments": result.get("segments", []),
                    "file_id": result.get("file_id", fid),
                }
                with open(sa_file, "w") as f:
                    json.dump(sa_data, f, indent=2)
                print(f"  Transcript: {len(sa_data['text'].split())} words")
            else:
                print(f"  WARN: No transcript returned")
                continue
        else:
            print(f"  Already transcribed")

        with open(sa_file) as f:
            sa_data = json.load(f)
        sa_text = normalize(sa_data.get("text", ""))

        wer = compute_wer(gt_text, sa_text)
        print(f"  WER: {wer}% (GT: {len(gt_text.split())} words, SA: {len(sa_text.split())} words)")

        results.append({
            "meeting_id": fid,
            "title": entry["title"],
            "accent": entry["accent"],
            "country": entry["country"],
            "gt_words": len(gt_text.split()),
            "sa_words": len(sa_text.split()),
            "baseline_wer": wer,
            "audio_path": str(audio),
            "gt_file": str(gt_file),
            "screenapp_file": str(sa_file),
        })
        print()

    # Save
    out = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        print(f"{'='*60}")
        print(f"{'Meeting':<12} {'Accent':<15} {'GT':>8} {'SA':>8} {'WER':>8}")
        print(f"{'-'*12} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            print(f"{r['meeting_id']:<12} {r['accent']:<15} {r['gt_words']:>8} {r['sa_words']:>8} {r['baseline_wer']:>7.1f}%")
        avg = sum(r["baseline_wer"] for r in results) / len(results)
        print(f"{'-'*12} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'AVERAGE':<12} {'':15} {'':>8} {'':>8} {avg:>7.1f}%")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
