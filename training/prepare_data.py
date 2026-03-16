"""Training data preparation utilities.

Extracted from FYP/Tests/prepare_training_data.py.
"""

import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an ASR transcript correction model. Given a noisy ASR transcript "
    "segment and context signals, detect errors in context-critical terms and "
    "output the corrected transcript with changes noted."
)


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def build_user_prompt(asr_text: str, vocab: list, category: str,
                      ocr_hints: list = None, lip_hint: str = None) -> str:
    return (
        "Correct this ASR transcript segment using the provided context.\n\n"
        f"ASR transcript: {asr_text}\n"
        f"Custom vocabulary: {json.dumps(vocab)}\n"
        f"Category: {category}\n"
        f"OCR hints: {json.dumps(ocr_hints or [])}\n"
        f"Lip reading hint: {lip_hint or 'null'}"
    )


def build_assistant_response(corrected: str, changes: list,
                              confidence: float, need_lip: bool = False) -> str:
    return json.dumps({
        "corrected": corrected,
        "changes": changes,
        "confidence": confidence,
        "need_lip": need_lip,
    })


def save_jsonl(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stratified_split(examples: list, train_ratio: float = 0.85, seed: int = 42) -> tuple:
    """Split examples into train/valid sets."""
    random.seed(seed)
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]
