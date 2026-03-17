"""Generate training pairs from ScreenApp ASR output vs ground truth transcripts.

Uses jiwer word alignment to identify ASR errors and create correction examples
in the ChatML JSONL format used by the LoRA fine-tuning pipeline.

Usage:
    python training/generate_pairs.py \
        --transcripts data/transcripts/librispeech.jsonl \
        --ground-truth data/datasets/LibriSpeech/train-clean-100/ \
        --dataset-type librispeech \
        --output data/training_pairs/librispeech_pairs.jsonl
"""

import argparse, json, logging, re, sys
from pathlib import Path
from typing import List, Tuple
import jiwer
from prepare_data import SYSTEM_PROMPT, build_user_prompt, build_assistant_response

logger = logging.getLogger(__name__)

# Simple heuristics for auto-detecting word categories
def detect_category(word: str) -> str:
    if word[0].isupper() and len(word) > 1:
        return "proper_noun"
    if word.isupper() and len(word) >= 2:
        return "tech_acronym"
    if any(c.isdigit() for c in word):
        return "numeric_term"
    return "content_word"

def extract_context(text: str, position: int, window: int = 80) -> str:
    start = max(0, position - window)
    end = min(len(text), position + window)
    return text[start:end]

def generate_pairs_from_alignment(
    asr_text: str,
    ground_truth: str,
    source: str = "unknown",
    file_id: str = "",
) -> List[dict]:
    """Compare ASR output vs ground truth using jiwer word alignment.

    Returns list of ChatML JSONL entries (both positive and negative examples).
    """
    pairs = []

    # Word-level alignment using jiwer
    try:
        wer_result = jiwer.process_words(ground_truth, asr_text)
    except Exception as e:
        logger.warning("Alignment failed: %s", e)
        return pairs

    asr_words = asr_text.split()
    ref_words = ground_truth.split()

    # Process alignments
    for chunk in wer_result.alignments[0]:
        if chunk.type == "substitute":
            # ASR got it wrong — this is a correction example
            for ref_idx, hyp_idx in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                if ref_idx < len(ref_words) and hyp_idx < len(asr_words):
                    error_word = asr_words[hyp_idx]
                    correct_word = ref_words[ref_idx]

                    if error_word.lower() == correct_word.lower():
                        continue  # Skip case-only differences

                    # Find position for context
                    pos = asr_text.find(error_word)
                    context = extract_context(asr_text, pos)
                    category = detect_category(correct_word)

                    # Build corrected context
                    corrected_context = context.replace(error_word, correct_word, 1)

                    entry = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(
                                context, [correct_word], category
                            )},
                            {"role": "assistant", "content": build_assistant_response(
                                corrected_context,
                                [f"{error_word} → {correct_word}"],
                                0.95,
                                False,
                            )},
                        ],
                        "metadata": {
                            "source": source,
                            "applied": True,
                            "term": correct_word,
                            "category": category,
                            "error_found": error_word,
                            "file_id": file_id,
                        },
                    }
                    pairs.append(entry)

        elif chunk.type == "equal":
            # ASR got it right — negative example (no correction needed)
            # Sample ~10% of equal segments for negative examples
            import random
            for hyp_idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if random.random() < 0.1 and hyp_idx < len(asr_words):
                    word = asr_words[hyp_idx]
                    if len(word) < 3:
                        continue
                    pos = asr_text.find(word)
                    context = extract_context(asr_text, pos)
                    category = detect_category(word)

                    entry = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(
                                context, [word], category
                            )},
                            {"role": "assistant", "content": build_assistant_response(
                                context,  # No changes
                                [],
                                0.99,
                                False,
                            )},
                        ],
                        "metadata": {
                            "source": source,
                            "applied": False,
                            "term": word,
                            "category": category,
                            "error_found": word,
                            "file_id": file_id,
                            "is_negative": True,
                        },
                    }
                    pairs.append(entry)

    return pairs


def save_pairs(pairs: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved %d pairs to %s", len(pairs), output_path)
