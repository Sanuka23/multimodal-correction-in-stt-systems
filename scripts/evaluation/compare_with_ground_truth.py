#!/usr/bin/env python3
"""Compare pipeline corrections with ground truth transcripts.

Fetches original + corrected transcripts from MongoDB,
compares with YouTube subtitle ground truth using WER.

Usage:
    python scripts/evaluation/compare_with_ground_truth.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Map file_ids from ScreenApp to video IDs (update these after uploading)
FILE_ID_MAP = {
    # 3Blue1Brown - Neural Networks
    "e8e83097-cacf-402f-b11d-3d86861d92c1": {
        "video_id": "aircAruvnKk",
        "title": "3Blue1Brown - Neural Networks",
        "ground_truth": "data/eval_videos/aircAruvnKk/aircAruvnKk_ground_truth.txt",
    },
    # Andrej Karpathy - GPT from scratch (2hr)
    "f9045221-a9a8-427b-870e-dca3334048a4": {
        "video_id": "kCc8FmEb1nY",
        "title": "Karpathy - GPT from scratch",
        "ground_truth": "data/eval_videos/kCc8FmEb1nY/kCc8FmEb1nY_ground_truth.txt",
    },
    # TED - What makes a good life (Robert Waldinger)
    "58c8617e-7cbd-4945-81ec-0a6c82de342e": {
        "video_id": "SqcY0GlETPk",
        "title": "TED - What makes a good life",
        "ground_truth": "data/eval_videos/SqcY0GlETPk/SqcY0GlETPk_ground_truth.txt",
    },
}


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate using jiwer (fast C implementation)."""
    try:
        import jiwer
        wer_score = jiwer.wer(reference, hypothesis)
        ref_words = len(reference.split())
        return {
            "wer": round(wer_score * 100, 2),
            "errors": int(wer_score * ref_words),
            "ref_words": ref_words,
            "hyp_words": len(hypothesis.split()),
        }
    except ImportError:
        # Fallback: simple WER on first 5000 words (to avoid O(n²) on huge transcripts)
        ref_words = reference.split()[:5000]
        hyp_words = hypothesis.split()[:5000]
        # Use rapidfuzz for fast edit distance
        from rapidfuzz.distance import Levenshtein
        errors = Levenshtein.distance(ref_words, hyp_words)
        wer = errors / max(len(ref_words), 1) * 100
        return {
            "wer": round(wer, 2),
            "errors": errors,
            "ref_words": len(ref_words),
            "hyp_words": len(hyp_words),
        }


def fetch_from_mongodb():
    """Fetch corrections from MongoDB."""
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27018")
        db = client["asr_correction_dashboard"]
        corrections = db["corrections"]

        results = {}
        for file_id, info in FILE_ID_MAP.items():
            doc = corrections.find_one({"file_id": file_id}, sort=[("created_at", -1)])
            if doc:
                results[file_id] = {
                    "original": doc.get("original_text", ""),
                    "corrected": doc.get("enhanced_text", ""),
                    "corrections_applied": doc.get("corrections_applied", 0),
                }
            else:
                print(f"  [WARN] No correction found for {file_id} ({info['title']})")

        return results
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return {}


def main():
    print("=== Ground Truth Comparison ===\n")

    # Fetch from MongoDB
    print("Fetching corrections from MongoDB...")
    mongo_data = fetch_from_mongodb()

    if not mongo_data:
        print("No data found in MongoDB. Make sure you've run corrections through ScreenApp.")
        print("\nAlternative: comparing from logs manually")

    results = []

    for file_id, info in FILE_ID_MAP.items():
        gt_path = Path(info["ground_truth"])
        if not gt_path.exists():
            print(f"\n[{info['title']}] Ground truth not found: {gt_path}")
            continue

        gt_text = normalize_text(gt_path.read_text())

        print(f"\n{'='*60}")
        print(f"Video: {info['title']}")
        print(f"File ID: {file_id}")
        print(f"Ground truth: {len(gt_text.split())} words")

        if file_id in mongo_data:
            original = normalize_text(mongo_data[file_id]["original"])
            corrected = normalize_text(mongo_data[file_id]["corrected"])
            n_corrections = mongo_data[file_id]["corrections_applied"]

            # Compare original ASR vs ground truth
            baseline = compute_wer(gt_text, original)
            print(f"\nBaseline (ScreenApp ASR vs Ground Truth):")
            print(f"  WER: {baseline['wer']:.1f}% ({baseline['errors']} errors / {baseline['ref_words']} words)")

            # Compare corrected vs ground truth
            corrected_result = compute_wer(gt_text, corrected)
            print(f"\nCorrected (Our Pipeline vs Ground Truth):")
            print(f"  WER: {corrected_result['wer']:.1f}% ({corrected_result['errors']} errors / {corrected_result['ref_words']} words)")

            delta = round(baseline["wer"] - corrected_result["wer"], 2)
            print(f"\nImprovement: {delta:+.1f}% WER")
            print(f"Corrections applied: {n_corrections}")

            if delta > 0:
                print(f"  → IMPROVED by {delta:.1f}% ✓")
            elif delta < 0:
                print(f"  → DEGRADED by {abs(delta):.1f}% ✗")
            else:
                print(f"  → NO CHANGE")

            results.append({
                "title": info["title"],
                "file_id": file_id,
                "baseline_wer": baseline["wer"],
                "corrected_wer": corrected_result["wer"],
                "improvement": delta,
                "corrections": n_corrections,
            })
        else:
            print(f"  No MongoDB data — skipping comparison")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"=== SUMMARY ({len(results)} videos) ===")
        print(f"{'='*60}")
        print(f"{'Video':<40} {'Baseline':>10} {'Corrected':>10} {'Delta':>8} {'Fixes':>6}")
        print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
        for r in results:
            print(f"{r['title'][:40]:<40} {r['baseline_wer']:>9.1f}% {r['corrected_wer']:>9.1f}% {r['improvement']:>+7.1f}% {r['corrections']:>5}")

        avg_baseline = sum(r["baseline_wer"] for r in results) / len(results)
        avg_corrected = sum(r["corrected_wer"] for r in results) / len(results)
        avg_delta = sum(r["improvement"] for r in results) / len(results)
        total_fixes = sum(r["corrections"] for r in results)

        print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
        print(f"{'AVERAGE':<40} {avg_baseline:>9.1f}% {avg_corrected:>9.1f}% {avg_delta:>+7.1f}% {total_fixes:>5}")

        # Save results
        output_path = Path("data/eval_videos/comparison_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
