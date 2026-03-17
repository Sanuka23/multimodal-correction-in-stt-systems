"""Merge training data from all sources and create train/valid splits.

Usage:
    python training/merge_and_split.py \
        --sources data/training_pairs/librispeech_pairs.jsonl \
                  data/training_pairs/ami_pairs.jsonl \
                  asr_correction/collected_data/corrections.jsonl \
        --output-dir data/collected_data \
        --screenapp-upsample 3
"""

import argparse, json, logging, random, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import stratified_split, save_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_jsonl(path: str) -> list:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, help="JSONL source files")
    parser.add_argument("--output-dir", default="data/collected_data")
    parser.add_argument("--screenapp-upsample", type=int, default=3,
                        help="Upsample ScreenApp data by this factor (domain relevance)")
    parser.add_argument("--max-negative-ratio", type=float, default=0.3,
                        help="Max ratio of negative (no-change) examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_examples = []
    source_counts = Counter()

    for source_path in args.sources:
        entries = load_jsonl(source_path)
        source_name = Path(source_path).stem

        # Detect if this is ScreenApp production data (upsample for domain relevance)
        is_screenapp = "corrections" in source_name or "screenapp" in source_name

        if is_screenapp and args.screenapp_upsample > 1:
            entries = entries * args.screenapp_upsample
            logger.info("Upsampled %s by %dx -> %d examples", source_name, args.screenapp_upsample, len(entries))

        all_examples.extend(entries)
        source_counts[source_name] += len(entries)
        logger.info("Loaded %d examples from %s", len(entries), source_path)

    # Separate positive and negative examples
    positive = [e for e in all_examples if not e.get("metadata", {}).get("is_negative", False)]
    negative = [e for e in all_examples if e.get("metadata", {}).get("is_negative", False)]

    # Cap negative examples
    max_negative = int(len(positive) * args.max_negative_ratio / (1 - args.max_negative_ratio))
    if len(negative) > max_negative:
        random.seed(args.seed)
        negative = random.sample(negative, max_negative)
        logger.info("Capped negative examples: %d -> %d", len(all_examples) - len(positive), max_negative)

    combined = positive + negative
    random.seed(args.seed)
    random.shuffle(combined)

    # Split
    train, valid = stratified_split(combined, train_ratio=0.85, seed=args.seed)

    # Save
    output_dir = Path(args.output_dir)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(valid, output_dir / "valid.jsonl")

    logger.info("=== Merge Complete ===")
    logger.info("Sources: %s", dict(source_counts))
    logger.info("Total: %d (positive: %d, negative: %d)", len(combined), len(positive), len(negative))
    logger.info("Train: %d | Valid: %d", len(train), len(valid))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
