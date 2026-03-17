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

sys.path.insert(0, str(Path(__file__).parent))
from generate_pairs import generate_pairs_from_alignment, save_pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching — first 100 chars, lowered."""
    return " ".join(text.lower().split())[:100]


def load_librispeech_transcripts(librispeech_dir: str) -> dict:
    """Load ground truth transcripts from LibriSpeech .trans.txt files.

    Returns: dict mapping normalized_text_prefix → (utterance_id, full_text)
    """
    transcripts = {}
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
                    key = normalize_for_matching(text)
                    transcripts[key] = (utterance_id, text)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-dir", required=True)
    parser.add_argument("--transcripts", required=True, help="ScreenApp transcripts JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gt_lookup = load_librispeech_transcripts(args.librispeech_dir)
    screenapp_entries = load_screenapp_transcripts(args.transcripts)

    all_pairs = []
    matched = 0

    for entry in screenapp_entries:
        asr_text = entry.get("transcript_text", "").strip()
        if not asr_text or len(asr_text) < 10:
            continue

        asr_key = normalize_for_matching(asr_text)

        # Try exact prefix match first
        best_match = None
        best_score = 0.0

        for gt_key, (uid, gt_text) in gt_lookup.items():
            # Exact normalized match
            if asr_key == gt_key:
                best_match = (uid, gt_text)
                best_score = 1.0
                break

            # Fuzzy: compare first 50 chars
            min_len = min(len(asr_key), len(gt_key), 50)
            if min_len < 10:
                continue
            common = sum(1 for a, b in zip(asr_key[:min_len], gt_key[:min_len]) if a == b)
            score = common / min_len
            if score > best_score and score > 0.7:
                best_score = score
                best_match = (uid, gt_text)

        if best_match:
            uid, gt_text = best_match
            pairs = generate_pairs_from_alignment(
                asr_text=asr_text,
                ground_truth=gt_text,
                source="librispeech",
                file_id=entry.get("file_id", uid),
            )
            all_pairs.extend(pairs)
            matched += 1

    logger.info("Matched %d/%d transcripts, generated %d pairs",
                matched, len(screenapp_entries), len(all_pairs))
    save_pairs(all_pairs, args.output)


if __name__ == "__main__":
    main()
