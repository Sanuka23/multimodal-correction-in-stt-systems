#!/usr/bin/env python3
"""Download evaluation videos, upload to ScreenApp, and compare with ground truth.

Flow:
1. Download YouTube videos + subtitles (ground truth)
2. Upload to ScreenApp → get real ScreenApp ASR transcript
3. Run our correction pipeline on ScreenApp transcript
4. Compare corrected vs ground truth (WER)

Usage:
    pip install yt-dlp
    python scripts/evaluation/download_eval_videos.py                    # Download only
    python scripts/evaluation/download_eval_videos.py --upload           # Download + upload to ScreenApp
    python scripts/evaluation/download_eval_videos.py --skip-download    # Upload already downloaded videos
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Videos selected for diverse testing
EVAL_VIDEOS = [
    {
        "id": "aircAruvnKk",
        "title": "Fireship - God Tier Developer Roadmap",
        "domain": "tech",
        "expected_terms": ["Kubernetes", "Docker", "TypeScript", "React", "AWS", "PostgreSQL"],
    },
    {
        "id": "kCc8FmEb1nY",
        "title": "TED - The next outbreak We re not ready (Bill Gates)",
        "domain": "science",
        "expected_terms": ["Ebola", "epidemic", "WHO", "pathogen", "vaccine", "microbe"],
    },
    {
        "id": "rfscVS0vtbw",
        "title": "FreeCodeCamp - Learn Python Full Course (first 10 min)",
        "domain": "programming",
        "expected_terms": ["Python", "variable", "function", "string", "boolean", "tuple"],
        "start_time": "00:00",
        "end_time": "10:00",
    },
    {
        "id": "WXsD0ZgxjRw",
        "title": "Google I/O - Whats new in AI",
        "domain": "ai",
        "expected_terms": ["Gemini", "transformer", "LLM", "fine-tuning", "multimodal", "TPU"],
    },
    {
        "id": "SqcY0GlETPk",
        "title": "TED - What makes a good life (Robert Waldinger)",
        "domain": "general",
        "expected_terms": ["relationships", "loneliness", "happiness", "Harvard", "study"],
    },
]

OUTPUT_DIR = Path("data/eval_videos")


def check_ytdlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_video(video: dict, output_dir: Path) -> dict:
    """Download video + subtitles using yt-dlp."""
    video_id = video["id"]
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    video_file = video_dir / f"{video_id}.mp4"
    gt_file = video_dir / f"{video_id}_ground_truth.txt"

    result = {"video_id": video_id, "title": video["title"], "domain": video["domain"]}

    # Skip if already downloaded
    if video_file.exists() and gt_file.exists():
        print(f"  Already downloaded: {video_id}")
        result["video_path"] = str(video_file)
        result["video_url"] = str(video_file)
        result["ground_truth_file"] = str(gt_file)
        result["status"] = "cached"
        return result

    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en",
        "--sub-format", "vtt",
        "--output", str(video_file),
        "--no-playlist",
    ]

    if "start_time" in video and "end_time" in video:
        cmd.extend(["--download-sections", f"*{video['start_time']}-{video['end_time']}"])

    cmd.append(url)

    print(f"  Downloading: {video['title'][:60]}...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"  [ERROR] yt-dlp failed: {proc.stderr[:200]}")
            result["status"] = "failed"
            return result
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Download timed out")
        result["status"] = "timeout"
        return result

    # Find subtitle file
    possible_subs = list(video_dir.glob(f"*.vtt")) + list(video_dir.glob(f"*.srt"))

    if possible_subs:
        gt_text = extract_text_from_subs(possible_subs[0])
        with open(gt_file, "w") as f:
            f.write(gt_text)
        print(f"  Ground truth: {len(gt_text)} chars")
        result["ground_truth_file"] = str(gt_file)
        result["ground_truth_chars"] = len(gt_text)

    # Find video file
    if video_file.exists():
        result["video_path"] = str(video_file)
        result["video_url"] = str(video_file)
        size_mb = video_file.stat().st_size / 1024 / 1024
        print(f"  Video: {size_mb:.1f} MB")
    else:
        mp4s = list(video_dir.glob("*.mp4"))
        if mp4s:
            result["video_path"] = str(mp4s[0])
            result["video_url"] = str(mp4s[0])

    result["status"] = "downloaded"
    result["expected_terms"] = video.get("expected_terms", [])
    return result


def extract_text_from_subs(subs_path: Path) -> str:
    """Extract plain text from VTT/SRT subtitle file."""
    text = subs_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"WEBVTT.*?\n\n", "", text, flags=re.DOTALL)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        if not line:
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"align:.*|position:.*|line:.*", "", line)
        if line.strip():
            lines.append(line.strip())

    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return " ".join(deduped)


def upload_to_screenapp(manifest: list) -> list:
    """Upload videos to ScreenApp and get ASR transcripts."""
    from training.screenapp_transcribe import ScreenAppTranscriber

    transcriber = ScreenAppTranscriber()
    print(f"\n=== Uploading to ScreenApp ===\n")

    for entry in manifest:
        video_path = entry.get("video_path")
        if not video_path or not Path(video_path).exists():
            print(f"  [{entry['video_id']}] No video file — skipping")
            continue

        screenapp_file = Path(video_path).parent / f"{entry['video_id']}_screenapp.json"

        # Skip if already transcribed
        if screenapp_file.exists():
            print(f"  [{entry['video_id']}] Already transcribed")
            entry["screenapp_file"] = str(screenapp_file)
            continue

        print(f"  [{entry['video_id']}] Uploading {entry['title'][:50]}...")
        try:
            result = transcriber.transcribe_file(video_path)
            if result and result.get("transcript_text"):
                transcript = {
                    "text": result["transcript_text"],
                    "segments": result.get("segments", []),
                    "file_id": result.get("file_id", entry["video_id"]),
                }
                with open(screenapp_file, "w") as f:
                    json.dump(transcript, f, indent=2)
                entry["screenapp_file"] = str(screenapp_file)
                entry["file_id"] = result.get("file_id")
                print(f"  → {len(transcript['segments'])} segments, {len(transcript['text'])} chars")
            else:
                print(f"  [WARN] No transcript returned")
        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}")

    return manifest


def run_evaluation(manifest: list, output_dir: Path):
    """Run correction pipeline and compare with ground truth."""
    from evaluation.wer import WERCalculator

    wer_calc = WERCalculator(normalize=True)
    print(f"\n=== Running Evaluation ===\n")

    results = []
    for entry in manifest:
        video_id = entry["video_id"]
        gt_file = entry.get("ground_truth_file")
        sa_file = entry.get("screenapp_file")

        if not gt_file or not sa_file:
            print(f"  [{video_id}] Missing ground truth or ScreenApp transcript — skipping")
            continue

        gt_text = Path(gt_file).read_text().strip()
        with open(sa_file) as f:
            sa_transcript = json.load(f)
        asr_text = sa_transcript.get("text", "")

        # Normalize for WER
        gt_norm = re.sub(r'\s+', ' ', gt_text.lower().strip())
        asr_norm = re.sub(r'\s+', ' ', asr_text.lower().strip())

        # Baseline WER
        baseline = wer_calc.compute(gt_norm, asr_norm)
        baseline_wer = round(baseline.wer * 100, 2)

        print(f"  [{video_id}] {entry['title'][:50]}")
        print(f"    Baseline WER: {baseline_wer:.1f}%")

        # Run correction
        try:
            from asr_correction import correct_transcript
            from asr_correction.config import CorrectionConfig

            config = CorrectionConfig()
            video_url = entry.get("video_url")

            t0 = time.perf_counter()
            enhanced, report = correct_transcript(
                transcript=sa_transcript,
                file_id=video_id,
                custom_vocabulary=entry.get("expected_terms", []),
                video_url=video_url,
                config=config,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            corrected_text = enhanced.get("text", asr_text)
            corrected_norm = re.sub(r'\s+', ' ', corrected_text.lower().strip())

            corrected = wer_calc.compute(gt_norm, corrected_norm)
            corrected_wer = round(corrected.wer * 100, 2)
            wer_delta = round(baseline_wer - corrected_wer, 2)

            print(f"    Corrected WER: {corrected_wer:.1f}% (delta: {wer_delta:+.1f}%)")
            print(f"    Corrections: {report.corrections_applied} | Time: {elapsed_ms:.0f}ms")

            results.append({
                "video_id": video_id,
                "title": entry["title"],
                "domain": entry["domain"],
                "baseline_wer": baseline_wer,
                "corrected_wer": corrected_wer,
                "wer_improvement": wer_delta,
                "corrections": report.corrections_applied,
                "latency_ms": round(elapsed_ms),
            })

        except Exception as e:
            print(f"    [ERROR] Correction failed: {e}")
            results.append({
                "video_id": video_id,
                "title": entry["title"],
                "domain": entry["domain"],
                "baseline_wer": baseline_wer,
                "corrected_wer": baseline_wer,
                "wer_improvement": 0,
                "corrections": 0,
                "error": str(e),
            })

        print()

    # Save results
    results_file = output_dir / "eval_results_v2.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    if results:
        avg_baseline = sum(r["baseline_wer"] for r in results) / len(results)
        avg_corrected = sum(r["corrected_wer"] for r in results) / len(results)
        avg_improvement = sum(r["wer_improvement"] for r in results) / len(results)
        print(f"=== Summary ({len(results)} videos) ===")
        print(f"  Avg Baseline WER:  {avg_baseline:.1f}%")
        print(f"  Avg Corrected WER: {avg_corrected:.1f}%")
        print(f"  Avg Improvement:   {avg_improvement:+.1f}%")
        print(f"  Results saved: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Download eval videos + ScreenApp transcription")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, use existing videos")
    parser.add_argument("--upload", action="store_true", help="Upload to ScreenApp for transcription")
    parser.add_argument("--evaluate", action="store_true", help="Run correction pipeline and compare WER")
    parser.add_argument("--all", action="store_true", help="Download + upload + evaluate")
    args = parser.parse_args()

    if args.all:
        args.upload = True
        args.evaluate = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download videos
    manifest = []
    if not args.skip_download:
        if not check_ytdlp():
            print("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
            sys.exit(1)

        print(f"=== Downloading {len(EVAL_VIDEOS)} evaluation videos ===\n")
        for i, video in enumerate(EVAL_VIDEOS):
            print(f"[{i+1}/{len(EVAL_VIDEOS)}] {video['title']}")
            result = download_video(video, OUTPUT_DIR)
            manifest.append(result)
            print()
    else:
        # Load existing manifest
        manifest_path = OUTPUT_DIR / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"Loaded {len(manifest)} videos from manifest")
        else:
            print("No manifest.json found. Run without --skip-download first.")
            sys.exit(1)

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Step 2: Upload to ScreenApp
    if args.upload:
        manifest = upload_to_screenapp(manifest)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Step 3: Evaluate
    if args.evaluate:
        run_evaluation(manifest, OUTPUT_DIR)

    downloaded = sum(1 for m in manifest if m.get("status") in ("downloaded", "cached"))
    with_gt = sum(1 for m in manifest if m.get("ground_truth_file"))
    with_sa = sum(1 for m in manifest if m.get("screenapp_file"))
    print(f"\n  Downloaded: {downloaded}/{len(EVAL_VIDEOS)}")
    print(f"  With ground truth: {with_gt}")
    print(f"  With ScreenApp transcript: {with_sa}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
