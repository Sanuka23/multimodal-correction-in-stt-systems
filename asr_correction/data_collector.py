"""Correction data collection for future model retraining.

Stores each correction attempt in ChatML JSONL format,
matching the training data structure used by prepare_training_data.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .types import CorrectionResult


def collect_correction_data(
    results: List[CorrectionResult],
    system_prompt: str,
    output_dir: Optional[str] = None,
) -> str:
    """Append correction results to training data JSONL.

    Each result becomes a ChatML training example with metadata.
    Applied corrections with high confidence serve as positive examples.
    Rejected corrections serve as potential negative examples.

    Returns path to the output file.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "collected_data")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "corrections.jsonl")

    entries = []
    for result in results:
        candidate = result.candidate

        user_content = (
            "Correct this ASR transcript segment using the provided context.\n\n"
            f"ASR transcript: {candidate.context}\n"
            f"Custom vocabulary: {json.dumps([candidate.term])}\n"
            f"Category: {candidate.category}\n"
            f"OCR hints: {json.dumps(result.ocr_hints_used)}\n"
            "Lip reading hint: null"
        )

        assistant_content = json.dumps(
            {
                "corrected": result.corrected_text,
                "changes": result.changes,
                "confidence": result.confidence,
                "need_lip": result.need_lip,
            }
        )

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "applied": result.applied,
                "timestamp": datetime.now().isoformat(),
                "term": candidate.term,
                "category": candidate.category,
                "error_found": candidate.error_found,
                "ocr_available": bool(result.ocr_hints_used),
            },
        }
        entries.append(entry)

    with open(output_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return output_path
