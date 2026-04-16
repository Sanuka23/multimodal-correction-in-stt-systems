#!/usr/bin/env python3
"""Upload AMI v2 (proper corpus with videos) audio to ScreenApp, compare with ground truth."""

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AMI_ROOT = Path("/Users/sanukathamuditha/Desktop/FYP/Data/AMI/organized")
OUT_DIR = PROJECT_ROOT / "data" / "eval_dataset" / "ami_v2"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def parse_ami_transcript(txt_path: Path) -> str:
    """Parse tab-separated AMI transcript: utt_XXXX\\tstart\\tend\\tspeaker\\ttext"""
    lines = []
    for raw in txt_path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) >= 5:
            lines.append(parts[4].strip())
        elif len(parts) >= 2:
            lines.append(parts[-1].strip())
    return " ".join(lines)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(ref: str, hyp: str) -> float:
    import jiwer
    return round(jiwer.wer(ref, hyp) * 100, 2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Build manifest
    entries = []
    for vid_dir in sorted(AMI_ROOT.iterdir()):
        if not vid_dir.is_dir():
            continue
        mid = vid_dir.name
        audio_dir = vid_dir / "audio"
        video_dir = vid_dir / "video"
        txt = vid_dir / "transcript" / f"{mid}.txt"

        audio_files = list(audio_dir.glob("*.wav")) if audio_dir.exists() else []
        video_files = list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4")) if video_dir.exists() else []

        if not audio_files or not txt.exists():
            print(f"  [{mid}] skip — missing files")
            continue

        gt_text = parse_ami_transcript(txt)
        gt_out = TRANSCRIPT_DIR / f"{mid}_ground_truth.txt"
        gt_out.write_text(gt_text)

        entries.append({
            "meeting_id": mid,
            "file_id": mid,
            "audio_path": str(audio_files[0]),
            "video_path": str(video_files[0]) if video_files else "",
            "ground_truth_file": str(gt_out),
        })

    print(f"\n=== AMI v2 → ScreenApp ({len(entries)} meetings) ===\n")

    from training.screenapp_transcribe import ScreenAppTranscriber
    transcriber = ScreenAppTranscriber(max_poll_time=1800)

    results = []
    for i, entry in enumerate(entries):
        mid = entry["meeting_id"]
        audio = Path(entry["audio_path"])
        gt_file = Path(entry["ground_truth_file"])

        print(f"[{i+1}/{len(entries)}] {mid}")

        gt_text = normalize(gt_file.read_text())
        sa_file = TRANSCRIPT_DIR / f"{mid}_screenapp.json"

        if not sa_file.exists():
            size_mb = audio.stat().st_size / 1e6
            print(f"  Uploading {size_mb:.1f} MB...")
            result = transcriber.transcribe_file(str(audio))
            if result and result.get("transcript_text"):
                sa_data = {
                    "text": result["transcript_text"],
                    "segments": result.get("segments", []),
                    "file_id": result.get("file_id", mid),
                }
                sa_file.write_text(json.dumps(sa_data, indent=2))
                print(f"  Transcript: {len(sa_data['text'].split())} words")
            else:
                print(f"  WARN: no transcript returned")
                continue
        else:
            print(f"  Already transcribed")

        sa_data = json.loads(sa_file.read_text())
        sa_text = normalize(sa_data.get("text", ""))
        wer = compute_wer(gt_text, sa_text)
        print(f"  WER: {wer}%  (GT: {len(gt_text.split())}w, SA: {len(sa_text.split())}w)")

        results.append({
            "meeting_id": mid,
            "file_id": mid,
            "title": f"AMI {mid}",
            "gt_words": len(gt_text.split()),
            "sa_words": len(sa_text.split()),
            "baseline_wer": wer,
            "audio_path": str(audio),
            "video_path": entry["video_path"],
            "gt_file": str(gt_file),
            "screenapp_file": str(sa_file),
        })
        print()

    out = OUT_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    (OUT_DIR / "manifest.json").write_text(json.dumps(entries, indent=2))

    if results:
        print("=" * 56)
        print(f"{'Meeting':<12} {'GT':>8} {'SA':>8} {'WER':>8}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            print(f"{r['meeting_id']:<12} {r['gt_words']:>8} {r['sa_words']:>8} {r['baseline_wer']:>7.1f}%")
        avg = sum(r["baseline_wer"] for r in results) / len(results)
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'AVERAGE':<12} {'':>8} {'':>8} {avg:>7.1f}%")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
