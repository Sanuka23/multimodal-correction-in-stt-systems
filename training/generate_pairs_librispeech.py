"""Generate correction pairs from LibriSpeech dataset.

Since ScreenApp renames uploaded files (e.g., "103-1240-0000.flac" becomes
"Mrs Rachel Lynde Surprised"), we match transcripts by text similarity
rather than filename.

Usage:
    python training/generate_pairs_librispeech.py \
        --librispeech-dir data/datasets/LibriSpeech/train-clean-100 \
        --transcripts data/transcripts/librispeech.jsonl \
        --output data/training_pairs/librispeech_pairs.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from generate_pairs import generate_pairs_from_alignment, save_pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def load_librispeech_transcripts(librispeech_dir: str) -> list:
    """Load ground truth transcripts from LibriSpeech .trans.txt files.

    Returns: list of (utterance_id, full_text, normalized_text)
    """
    transcripts = []
    base = Path(librispeech_dir)

    for trans_file in base.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    transcripts.append((utterance_id, text, normalize(text)))

    logger.info("Loaded %d LibriSpeech ground truth transcripts", len(transcripts))
    return transcripts


def load_screenapp_transcripts(jsonl_path: str) -> list:
    """Load ScreenApp ASR transcripts from JSONL."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info("Loaded %d ScreenApp transcripts", len(entries))
    return entries


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    try:
        return jiwer.wer(ref, hyp)
    except (ValueError, ZeroDivisionError):
        return 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-dir", required=True)
    parser.add_argument("--transcripts", required=True, help="ScreenApp transcripts JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wer-threshold", type=float, default=0.5,
                        help="Max WER to consider a match (default: 0.5)")
    args = parser.parse_args()

    gt_list = load_librispeech_transcripts(args.librispeech_dir)
    screenapp_entries = load_screenapp_transcripts(args.transcripts)

    # Build lookup by normalized prefix (first 60 chars) for fast candidate filtering
    from collections import defaultdict
    prefix_index = defaultdict(list)
    for uid, text, norm_text in gt_list:
        # Index by multiple prefix lengths for better recall
        for plen in (30, 50, 80):
            prefix = norm_text[:plen]
            if len(prefix) >= 10:
                prefix_index[prefix].append((uid, text, norm_text))

    all_pairs = []
    matched = 0
    unmatched_samples = []

    for entry in screenapp_entries:
        asr_text = entry.get("transcript_text", "").strip()
        if not asr_text or len(asr_text) < 10:
            continue

        asr_norm = normalize(asr_text)

        # Pass 1: Try exact normalized match
        best_match = None
        best_wer = 1.0

        for uid, text, norm_text in gt_list:
            if asr_norm == norm_text:
                best_match = (uid, text)
                best_wer = 0.0
                break

        # Pass 2: Prefix-based candidate filtering + WER scoring
        if best_match is None:
            candidates = set()
            for plen in (30, 50, 80):
                asr_prefix = asr_norm[:plen]
                if len(asr_prefix) < 10:
                    continue
                # Check prefixes that share first 10 chars
                for gt_prefix, gt_entries in prefix_index.items():
                    if gt_prefix[:10] == asr_prefix[:10]:
                        for c in gt_entries:
                            candidates.add(c)

            # Score candidates by WER
            for uid, text, norm_text in candidates:
                # Quick length filter: skip if lengths differ by >2x
                if len(norm_text) > 0 and (
                    len(asr_norm) / len(norm_text) > 2.0 or
                    len(norm_text) / len(asr_norm) > 2.0
                ):
                    continue

                wer_score = compute_wer(norm_text, asr_norm)
                if wer_score < best_wer:
                    best_wer = wer_score
                    best_match = (uid, text)

        # Pass 3: Brute-force WER on short texts (< 200 chars) if still no match
        if best_match is None and len(asr_norm) < 200:
            for uid, text, norm_text in gt_list:
                if abs(len(norm_text) - len(asr_norm)) > len(asr_norm) * 0.5:
                    continue
                wer_score = compute_wer(norm_text, asr_norm)
                if wer_score < best_wer:
                    best_wer = wer_score
                    best_match = (uid, text)

        if best_match and best_wer <= args.wer_threshold:
            uid, gt_text = best_match
            pairs = generate_pairs_from_alignment(
                asr_text=asr_text,
                ground_truth=gt_text,
                source="librispeech",
                file_id=entry.get("file_id", uid),
            )
            all_pairs.extend(pairs)
            matched += 1
        else:
            if len(unmatched_samples) < 10:
                unmatched_samples.append({
                    "file_name": entry.get("file_name", ""),
                    "asr_preview": asr_text[:80],
                    "best_wer": round(best_wer, 3),
                })

    logger.info("Matched %d/%d transcripts (%.1f%%), generated %d pairs",
                matched, len(screenapp_entries),
                100 * matched / max(len(screenapp_entries), 1), len(all_pairs))
    if unmatched_samples:
        logger.info("Sample unmatched transcripts:")
        for s in unmatched_samples:
            logger.info("  %s (WER=%.3f): %s", s["file_name"], s["best_wer"], s["asr_preview"])
    save_pairs(all_pairs, args.output)


if __name__ == "__main__":
    main()
