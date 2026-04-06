#!/usr/bin/env python3
"""Full evaluation pipeline for multimodal ASR correction.

Measures all 4 research questions:
  RQ1: WER/TTER before vs after correction
  RQ2: End-to-end latency p50/p95/p99
  RQ3: Per-accent WER breakdown
  RQ4: AVSR ablation (with vs without lip reading)

Usage:
    python scripts/run_full_eval.py --dataset data/eval_dataset --output data/eval_results
    python scripts/run_full_eval.py --dataset data/eval_dataset --output data/eval_results --ablation
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.wer import WERCalculator
from evaluation.tter import compute_tter
from asr_correction.vocabulary import load_domain_vocab, merge_vocabularies
from asr_correction.config import CorrectionConfig


def load_manifest(dataset_dir: str) -> list:
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {dataset_dir}. Run collect_dataset.py first.")
    with open(manifest_path) as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def normalize_text(text: str) -> str:
    """Basic text normalization for WER comparison."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    except ImportError:
        # Fallback: lowercase + collapse whitespace
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())


def run_correction(transcript: dict, video_path: str = None,
                   use_avsr: bool = False, dry_run: bool = False,
                   skip_ocr: bool = False) -> tuple:
    """Run the v2 correction pipeline on one transcript."""
    from asr_correction import correct_transcript

    config = CorrectionConfig(dry_run=dry_run)

    # Video URL for OCR + Whisper Pass 2
    video_url = None if skip_ocr else video_path

    t0 = time.perf_counter()
    enhanced, report = correct_transcript(
        transcript=transcript,
        file_id="eval",
        custom_vocabulary=[],
        video_url=video_url,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return enhanced, report, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Full ASR correction evaluation")
    parser.add_argument("--dataset", default="data/eval_dataset")
    parser.add_argument("--output", default="data/eval_results")
    parser.add_argument("--ablation", action="store_true",
                        help="Run with/without AVSR for RQ4 ablation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode (no model inference)")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="Skip OCR extraction (much faster for long videos)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wer_calc = WERCalculator(normalize=True)
    domain_vocab = load_domain_vocab()
    vocab_terms = merge_vocabularies([], domain_vocab)
    term_list = vocab_terms  # compute_tter expects list of dicts, not strings

    manifest = load_manifest(args.dataset)
    print(f"\n=== Full Evaluation ({len(manifest)} videos) ===\n")

    per_video = []
    latencies_ms = []
    avsr_ablation = []

    for i, entry in enumerate(manifest):
        video_id = entry["video_id"]
        title = entry["title"]
        accent = entry["accent"]
        video_path = entry.get("video_path")

        print(f"[{i + 1}/{len(manifest)}] {title[:50]}")
        print(f"          accent={accent}")

        # Load texts
        gt_text = load_text(entry["ground_truth_file"])
        asr_text = load_text(entry["asr_file"])
        gt_norm = normalize_text(gt_text)
        asr_norm = normalize_text(asr_text)

        with open(entry["screenapp_file"]) as f:
            sa_transcript = json.load(f)

        # Baseline WER/CER
        baseline = wer_calc.compute(gt_norm, asr_norm)
        baseline_wer = round(baseline.wer * 100, 2)
        baseline_cer = round(baseline.cer * 100, 2)

        # Baseline TTER
        baseline_tter = compute_tter(gt_norm, asr_norm, term_list)
        baseline_tter_val = baseline_tter.get("overall_tter", 0)

        print(f"          Baseline  WER={baseline_wer:.1f}%  CER={baseline_cer:.1f}%  TTER={baseline_tter_val:.1f}%")

        # Run correction
        try:
            enhanced, report, elapsed_ms = run_correction(
                sa_transcript, video_path, use_avsr=True,
                dry_run=args.dry_run, skip_ocr=args.skip_ocr,
            )
            latencies_ms.append(elapsed_ms)

            corrected_text = enhanced.get("text", asr_text)
            corrected_norm = normalize_text(corrected_text)

            corrected_result = wer_calc.compute(gt_norm, corrected_norm)
            corrected_wer = round(corrected_result.wer * 100, 2)
            corrected_cer = round(corrected_result.cer * 100, 2)

            corrected_tter = compute_tter(gt_norm, corrected_norm, term_list)
            corrected_tter_val = corrected_tter.get("overall_tter", 0)

            wer_delta = round(baseline_wer - corrected_wer, 2)
            tter_delta = round(baseline_tter_val - corrected_tter_val, 2)

            print(f"          Corrected WER={corrected_wer:.1f}%  CER={corrected_cer:.1f}%  "
                  f"TTER={corrected_tter_val:.1f}%  latency={elapsed_ms:.0f}ms")
            print(f"          WER delta={wer_delta:+.1f}%  TTER delta={tter_delta:+.1f}%  "
                  f"corrections={report.corrections_applied}/{report.corrections_attempted}")

        except Exception as e:
            print(f"          [ERROR] Correction failed: {e}")
            corrected_wer = baseline_wer
            corrected_cer = baseline_cer
            corrected_tter_val = baseline_tter_val
            elapsed_ms = 0
            wer_delta = 0
            tter_delta = 0
            report = None

        row = {
            "video_id": video_id,
            "title": title,
            "accent": accent,
            "domain": entry.get("domain", ""),
            "baseline_wer": baseline_wer,
            "baseline_cer": baseline_cer,
            "baseline_tter": round(baseline_tter_val, 2),
            "corrected_wer": corrected_wer,
            "corrected_cer": corrected_cer,
            "corrected_tter": round(corrected_tter_val, 2),
            "wer_improvement": wer_delta,
            "tter_improvement": tter_delta,
            "latency_ms": round(elapsed_ms, 1),
            "corrections_applied": report.corrections_applied if report else 0,
        }
        per_video.append(row)

        # AVSR ablation
        if args.ablation and not args.dry_run and elapsed_ms > 0:
            try:
                enhanced_no_avsr, _, _ = run_correction(
                    sa_transcript, video_path, use_avsr=False
                )
                no_avsr_norm = normalize_text(enhanced_no_avsr.get("text", asr_text))
                no_avsr_wer = round(wer_calc.compute(gt_norm, no_avsr_norm).wer * 100, 2)

                avsr_gain = round(no_avsr_wer - corrected_wer, 2)
                avsr_ablation.append({
                    "video_id": video_id,
                    "accent": accent,
                    "wer_without_avsr": no_avsr_wer,
                    "wer_with_avsr": corrected_wer,
                    "avsr_gain": avsr_gain,
                })
                print(f"          AVSR ablation: without={no_avsr_wer:.1f}%  "
                      f"with={corrected_wer:.1f}%  gain={avsr_gain:+.1f}%")
            except Exception as e:
                print(f"          [WARN] AVSR ablation failed: {e}")

        print()

    # === Aggregate Statistics ===
    n = len(per_video)
    if n == 0:
        print("No videos evaluated!")
        return

    avg_baseline_wer = float(np.mean([r["baseline_wer"] for r in per_video]))
    avg_corrected_wer = float(np.mean([r["corrected_wer"] for r in per_video]))
    avg_wer_delta = float(np.mean([r["wer_improvement"] for r in per_video]))
    rel_wer_improve = (avg_baseline_wer - avg_corrected_wer) / avg_baseline_wer * 100 if avg_baseline_wer > 0 else 0

    avg_baseline_tter = float(np.mean([r["baseline_tter"] for r in per_video]))
    avg_corrected_tter = float(np.mean([r["corrected_tter"] for r in per_video]))
    rel_tter_improve = (avg_baseline_tter - avg_corrected_tter) / avg_baseline_tter * 100 if avg_baseline_tter > 0 else 0

    # Latency
    lat = latencies_ms if latencies_ms else [0]
    lat_p50 = float(np.percentile(lat, 50))
    lat_p95 = float(np.percentile(lat, 95))
    lat_p99 = float(np.percentile(lat, 99))

    # Per-accent
    accent_groups = defaultdict(list)
    for r in per_video:
        accent_groups[r["accent"]].append(r)

    accent_breakdown = {}
    for accent, rows in accent_groups.items():
        accent_breakdown[accent] = {
            "n": len(rows),
            "avg_baseline_wer": round(float(np.mean([r["baseline_wer"] for r in rows])), 2),
            "avg_corrected_wer": round(float(np.mean([r["corrected_wer"] for r in rows])), 2),
            "avg_wer_improvement": round(float(np.mean([r["wer_improvement"] for r in rows])), 2),
        }

    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nVideos: {n}")
    print(f"\n--- RQ1: Accuracy ---")
    print(f"  Baseline WER:  {avg_baseline_wer:.2f}%")
    print(f"  Corrected WER: {avg_corrected_wer:.2f}%")
    print(f"  WER delta:     {avg_wer_delta:+.2f}%")
    print(f"  Relative:      {rel_wer_improve:.1f}% improvement")
    print(f"  Baseline TTER: {avg_baseline_tter:.2f}%")
    print(f"  Corrected TTER:{avg_corrected_tter:.2f}%")
    print(f"  TTER relative: {rel_tter_improve:.1f}% improvement")

    print(f"\n--- RQ2: Latency ---")
    print(f"  p50={lat_p50:.0f}ms  p95={lat_p95:.0f}ms  p99={lat_p99:.0f}ms")

    print(f"\n--- RQ3: Accent Breakdown ---")
    for accent, stats in sorted(accent_breakdown.items()):
        print(f"  {accent:15s} n={stats['n']}  "
              f"baseline={stats['avg_baseline_wer']:.1f}%  "
              f"corrected={stats['avg_corrected_wer']:.1f}%  "
              f"delta={stats['avg_wer_improvement']:+.1f}%")

    if avsr_ablation:
        avg_avsr_gain = float(np.mean([r["avsr_gain"] for r in avsr_ablation]))
        print(f"\n--- RQ4: AVSR Contribution ---")
        print(f"  Average WER gain from AVSR: {avg_avsr_gain:+.2f}%")

    # Save outputs
    summary = {
        "n_videos": n,
        "rq1_accuracy": {
            "avg_baseline_wer": round(avg_baseline_wer, 2),
            "avg_corrected_wer": round(avg_corrected_wer, 2),
            "absolute_wer_improvement": round(avg_wer_delta, 2),
            "relative_wer_improvement_pct": round(rel_wer_improve, 1),
            "avg_baseline_tter": round(avg_baseline_tter, 2),
            "avg_corrected_tter": round(avg_corrected_tter, 2),
            "relative_tter_improvement_pct": round(rel_tter_improve, 1),
        },
        "rq2_latency": {
            "p50_ms": round(lat_p50, 1),
            "p95_ms": round(lat_p95, 1),
            "p99_ms": round(lat_p99, 1),
        },
        "rq3_accent_breakdown": accent_breakdown,
        "rq4_avsr_ablation": {
            "measured": len(avsr_ablation) > 0,
            "avg_avsr_gain_wer_pct": round(
                float(np.mean([r["avsr_gain"] for r in avsr_ablation])), 2
            ) if avsr_ablation else None,
            "details": avsr_ablation,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "per_video.csv", "w", newline="") as f:
        if per_video:
            writer = csv.DictWriter(f, fieldnames=per_video[0].keys())
            writer.writeheader()
            writer.writerows(per_video)

    accent_rows = [{"accent": a, **s} for a, s in accent_breakdown.items()]
    with open(output_dir / "accent_breakdown.csv", "w", newline="") as f:
        if accent_rows:
            writer = csv.DictWriter(f, fieldnames=accent_rows[0].keys())
            writer.writeheader()
            writer.writerows(accent_rows)

    print(f"\n\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
