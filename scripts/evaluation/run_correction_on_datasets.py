#!/usr/bin/env python3
"""Run correction pipeline on Earnings-22 and AMI datasets, compare with ground truth."""

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
for line in (PROJECT_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(ref, hyp):
    import jiwer
    return round(jiwer.wer(ref, hyp) * 100, 2)


def run_correction(transcript_data, audio_path=None):
    """Run the ASR correction pipeline on a transcript."""
    from asr_correction import correct_transcript
    from asr_correction.config import CorrectionConfig

    config = CorrectionConfig()
    # Pass audio path as video_url — faster-whisper handles audio files directly
    enhanced, report = correct_transcript(
        transcript=transcript_data,
        file_id="eval",
        custom_vocabulary=[],
        video_url=audio_path,
        config=config,
    )
    return enhanced, report


def process_dataset(name, results_path, transcript_dir, gt_suffix="_ground_truth.txt", sa_suffix="_screenapp.json"):
    """Process a dataset: run correction, compare baseline vs corrected vs ground truth."""
    if not results_path.exists():
        print(f"  No baseline results found: {results_path}")
        return []

    with open(results_path) as f:
        baseline = json.load(f)

    results = []
    for entry in baseline:
        mid = entry["meeting_id"]
        gt_file = Path(entry.get("gt_file", transcript_dir / f"{mid}{gt_suffix}"))
        sa_file = Path(entry.get("screenapp_file", transcript_dir / f"{mid}{sa_suffix}"))

        if not gt_file.exists() or not sa_file.exists():
            print(f"  [{mid}] SKIP — missing files")
            continue

        gt_text = normalize(gt_file.read_text())
        with open(sa_file) as f:
            sa_data = json.load(f)

        sa_text = normalize(sa_data.get("text", ""))
        baseline_wer = compute_wer(gt_text, sa_text)

        audio_path = entry.get("audio_path", "")
        print(f"  [{mid}] Baseline WER: {baseline_wer}% — running correction...")

        t0 = time.time()
        try:
            enhanced, report = run_correction(sa_data, audio_path=audio_path if audio_path and Path(audio_path).exists() else None)
            elapsed = time.time() - t0

            corrected_text = normalize(enhanced.get("text", sa_data.get("text", "")))
            corrected_wer = compute_wer(gt_text, corrected_text)
            delta = round(baseline_wer - corrected_wer, 2)

            print(f"           Corrected WER: {corrected_wer}% (delta: {delta:+.1f}%, "
                  f"{report.corrections_applied} corrections, {elapsed:.0f}s)")

            results.append({
                "meeting_id": mid,
                "title": entry.get("title", mid),
                "accent": entry.get("accent", ""),
                "country": entry.get("country", ""),
                "gt_words": len(gt_text.split()),
                "sa_words": len(sa_text.split()),
                "baseline_wer": baseline_wer,
                "corrected_wer": corrected_wer,
                "wer_improvement": delta,
                "corrections_applied": report.corrections_applied,
                "corrections_attempted": report.corrections_attempted,
                "latency_s": round(elapsed, 1),
            })
        except Exception as e:
            print(f"           ERROR: {e}")
            results.append({
                "meeting_id": mid,
                "title": entry.get("title", mid),
                "accent": entry.get("accent", ""),
                "country": entry.get("country", ""),
                "gt_words": len(gt_text.split()),
                "sa_words": len(sa_text.split()),
                "baseline_wer": baseline_wer,
                "corrected_wer": baseline_wer,
                "wer_improvement": 0,
                "corrections_applied": 0,
                "corrections_attempted": 0,
                "latency_s": 0,
                "error": str(e),
            })

    return results


def main():
    print("=" * 60)
    print("RUNNING CORRECTION PIPELINE ON EVALUATION DATASETS")
    print("=" * 60)

    all_results = {}

    # Earnings-22
    print(f"\n--- Earnings-22 (4 calls) ---\n")
    e22_results = process_dataset(
        "earnings22",
        PROJECT_ROOT / "data/eval_dataset/earnings22/baseline_results.json",
        PROJECT_ROOT / "data/eval_dataset/earnings22/transcripts",
    )
    if e22_results:
        out = PROJECT_ROOT / "data/eval_dataset/earnings22/corrected_results.json"
        with open(out, "w") as f:
            json.dump(e22_results, f, indent=2)
        all_results["earnings22"] = e22_results
        print(f"\n  Saved: {out}")

    # AMI (worst 4)
    print(f"\n--- AMI Meeting Corpus (worst 4) ---\n")
    ami_results = process_dataset(
        "ami",
        PROJECT_ROOT / "data/eval_dataset/ami_baseline_results.json",
        PROJECT_ROOT / "data/eval_dataset/transcripts",
        gt_suffix="_ground_truth.txt",
        sa_suffix="_screenapp.json",
    )
    if ami_results:
        # Sort by baseline WER descending, take worst 4
        ami_results.sort(key=lambda r: r["baseline_wer"], reverse=True)
        out = PROJECT_ROOT / "data/eval_dataset/ami_corrected_results.json"
        with open(out, "w") as f:
            json.dump(ami_results, f, indent=2)
        all_results["ami"] = ami_results
        print(f"\n  Saved: {out}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Meeting':<12} {'Accent':<15} {'Baseline':>9} {'Corrected':>10} {'Delta':>8} {'Fixes':>6}")
    print(f"{'-'*12} {'-'*12} {'-'*15} {'-'*9} {'-'*10} {'-'*8} {'-'*6}")

    for dataset, results in all_results.items():
        for r in results:
            print(f"{dataset:<12} {r['meeting_id']:<12} {r.get('accent',''):<15} "
                  f"{r['baseline_wer']:>8.1f}% {r['corrected_wer']:>9.1f}% "
                  f"{r['wer_improvement']:>+7.1f}% {r['corrections_applied']:>5}")

    # Overall averages
    all_r = [r for results in all_results.values() for r in results]
    if all_r:
        avg_base = sum(r["baseline_wer"] for r in all_r) / len(all_r)
        avg_corr = sum(r["corrected_wer"] for r in all_r) / len(all_r)
        avg_delta = sum(r["wer_improvement"] for r in all_r) / len(all_r)
        total_fixes = sum(r["corrections_applied"] for r in all_r)
        print(f"{'-'*12} {'-'*12} {'-'*15} {'-'*9} {'-'*10} {'-'*8} {'-'*6}")
        print(f"{'OVERALL':<12} {'':<12} {'':<15} {avg_base:>8.1f}% {avg_corr:>9.1f}% {avg_delta:>+7.1f}% {total_fixes:>5}")
        rel_improve = (avg_base - avg_corr) / avg_base * 100 if avg_base > 0 else 0
        print(f"\nRelative WER improvement: {rel_improve:.1f}%")


if __name__ == "__main__":
    main()
