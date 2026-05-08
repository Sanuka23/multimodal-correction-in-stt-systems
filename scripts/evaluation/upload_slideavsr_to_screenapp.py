#!/usr/bin/env python3
"""Upload SlideAVSR audio to ScreenApp, compare transcripts with ground truth."""

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SLIDEAVSR_ROOT = Path("/Users/sanukathamuditha/Desktop/FYP/Data/SlideAVSR/dataset/organized")
OUT_DIR = PROJECT_ROOT / "data" / "eval_dataset" / "slideavsr"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def parse_ground_truth(txt_path: Path) -> str:
    """Strip `utt_XXXX utt start end -X ` prefix from each line and join."""
    lines = []
    for raw in txt_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # utt_0000 utt 0.06 4.27 -X TEXT...
        m = re.match(r"^utt_\d+\s+utt\s+[\d.]+\s+[\d.]+\s+-X\s+(.*)$", raw)
        if m:
            lines.append(m.group(1).strip())
        else:
            lines.append(raw)
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
    for vid_dir in sorted(SLIDEAVSR_ROOT.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid_id = vid_dir.name
        audio = vid_dir / "audio" / f"{vid_id}.wav"
        video = vid_dir / "video" / f"{vid_id}.mp4"
        txt = vid_dir / "transcript" / f"{vid_id}.txt"
        if not (audio.exists() and txt.exists()):
            print(f"  [{vid_id}] skip — missing files")
            continue

        gt_text = parse_ground_truth(txt)
        gt_out = TRANSCRIPT_DIR / f"{vid_id}_ground_truth.txt"
        gt_out.write_text(gt_text)

        entries.append({
            "file_id": vid_id,
            "audio_path": str(audio),
            "video_path": str(video),
            "ground_truth_file": str(gt_out),
        })

    print(f"\n=== SlideAVSR → ScreenApp ({len(entries)} videos) ===\n")

    from training.screenapp_transcribe import ScreenAppTranscriber
    transcriber = ScreenAppTranscriber(max_poll_time=900)

    results = []
    for i, entry in enumerate(entries):
        fid = entry["file_id"]
        audio = Path(entry["audio_path"])
        gt_file = Path(entry["ground_truth_file"])

        print(f"[{i+1}/{len(entries)}] {fid}")

        gt_text = normalize(gt_file.read_text())
        sa_file = TRANSCRIPT_DIR / f"{fid}_screenapp.json"

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
            "meeting_id": fid,
            "file_id": fid,
            "title": fid,
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

    manifest_out = OUT_DIR / "manifest.json"
    manifest_out.write_text(json.dumps(entries, indent=2))

    if results:
        print("=" * 56)
        print(f"{'Video':<14} {'GT':>8} {'SA':>8} {'WER':>8}")
        print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            print(f"{r['file_id']:<14} {r['gt_words']:>8} {r['sa_words']:>8} {r['baseline_wer']:>7.1f}%")
        avg = sum(r["baseline_wer"] for r in results) / len(results)
        print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'AVERAGE':<14} {'':>8} {'':>8} {avg:>7.1f}%")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
