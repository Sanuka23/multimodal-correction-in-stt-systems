"""Generate correction pairs from AMI Meeting Corpus.

Since AMI audio is not available locally, we use the word-level XML annotations
as ground truth and generate synthetic ASR errors (common substitutions,
homophones, word boundary errors) to create training pairs.

Usage:
    python training/generate_pairs_ami.py \
        --ami-dir data/datasets/ami \
        --output data/training_pairs/ami_pairs.jsonl
"""

import argparse
import json
import logging
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_pairs import generate_pairs_from_alignment, save_pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Common ASR error patterns (word → likely ASR mistake)
COMMON_SUBSTITUTIONS = {
    "their": ["there", "they're"],
    "there": ["their", "they're"],
    "they're": ["their", "there"],
    "your": ["you're", "ur"],
    "you're": ["your"],
    "its": ["it's"],
    "it's": ["its"],
    "than": ["then"],
    "then": ["than"],
    "affect": ["effect"],
    "effect": ["affect"],
    "accept": ["except"],
    "except": ["accept"],
    "whether": ["weather"],
    "weather": ["whether"],
    "which": ["witch"],
    "right": ["write"],
    "write": ["right"],
    "know": ["no"],
    "no": ["know"],
    "new": ["knew"],
    "knew": ["new"],
    "here": ["hear"],
    "hear": ["here"],
    "would": ["wood"],
    "through": ["threw"],
    "two": ["to", "too"],
    "to": ["two", "too"],
    "too": ["to", "two"],
    "where": ["were", "wear"],
    "were": ["where", "wear"],
    "break": ["brake"],
    "quite": ["quiet"],
    "lose": ["loose"],
    "principal": ["principle"],
    "principle": ["principal"],
}


def load_ami_transcripts(ami_dir: str) -> dict:
    """Load AMI meeting transcripts from word-level XML annotations.

    Returns: dict mapping meeting_id -> full transcript text
    """
    transcripts = {}
    words_dir = Path(ami_dir) / "words"

    if not words_dir.exists():
        logger.error("AMI words directory not found at %s", words_dir)
        return transcripts

    for xml_file in sorted(words_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            words = []
            for w in root.iter("w"):
                text = w.text
                if text and text.strip():
                    words.append(text.strip())

            # Group by base meeting ID (e.g., "ES2002a.A" → "ES2002a")
            meeting_id = xml_file.stem.rsplit(".", 1)[0]  # e.g., "ES2002a.A"
            base_meeting = meeting_id.split(".")[0]  # e.g., "ES2002a"

            if base_meeting not in transcripts:
                transcripts[base_meeting] = []
            transcripts[base_meeting].extend(words)
        except ET.ParseError:
            logger.warning("Failed to parse %s", xml_file)

    result = {k: " ".join(v) for k, v in transcripts.items()}
    logger.info("Loaded %d AMI meeting transcripts (avg %.0f words)",
                len(result),
                sum(len(v.split()) for v in result.values()) / max(len(result), 1))
    return result


def introduce_asr_errors(text: str, error_rate: float = 0.05) -> str:
    """Introduce synthetic ASR-like errors into clean text.

    Args:
        text: Clean ground truth text.
        error_rate: Probability of introducing an error per word.

    Returns: Text with synthetic ASR errors.
    """
    words = text.split()
    corrupted = []

    for word in words:
        if random.random() < error_rate:
            lower = word.lower()
            # Try homophone substitution
            if lower in COMMON_SUBSTITUTIONS:
                replacement = random.choice(COMMON_SUBSTITUTIONS[lower])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                corrupted.append(replacement)
            # Random word deletion (simulate missed words)
            elif random.random() < 0.3:
                continue  # Skip word
            # Random word duplication (simulate stuttering)
            elif random.random() < 0.2:
                corrupted.append(word)
                corrupted.append(word)
            else:
                corrupted.append(word)
        else:
            corrupted.append(word)

    return " ".join(corrupted)


def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list:
    """Split text into overlapping chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words) - chunk_size + 1, step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Add remainder if significant
    if len(words) > chunk_size and len(words) % step > chunk_size // 2:
        chunks.append(" ".join(words[-chunk_size:]))

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--error-rate", type=float, default=0.05,
                        help="ASR error rate for synthetic errors (default: 0.05)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Words per chunk (default: 50)")
    parser.add_argument("--max-chunks-per-meeting", type=int, default=20,
                        help="Max chunks per meeting (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ground_truth = load_ami_transcripts(args.ami_dir)

    if not ground_truth:
        logger.error("No AMI transcripts found")
        sys.exit(1)

    all_pairs = []

    for meeting_id, gt_text in ground_truth.items():
        # Split into manageable chunks
        chunks = chunk_text(gt_text, chunk_size=args.chunk_size)

        # Limit chunks per meeting to keep dataset balanced
        if len(chunks) > args.max_chunks_per_meeting:
            chunks = random.sample(chunks, args.max_chunks_per_meeting)

        for chunk_gt in chunks:
            # Create synthetic ASR output with errors
            chunk_asr = introduce_asr_errors(chunk_gt, error_rate=args.error_rate)

            # Skip if identical (no errors introduced)
            if chunk_asr == chunk_gt:
                continue

            pairs = generate_pairs_from_alignment(
                asr_text=chunk_asr,
                ground_truth=chunk_gt,
                source="ami",
                file_id=meeting_id,
            )
            all_pairs.extend(pairs)

    logger.info("Generated %d pairs from %d AMI meetings", len(all_pairs), len(ground_truth))
    save_pairs(all_pairs, args.output)


if __name__ == "__main__":
    main()
