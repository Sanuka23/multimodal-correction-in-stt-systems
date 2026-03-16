"""Word Error Rate (WER) and Character Error Rate (CER) calculation.

Uses jiwer for edit-distance-based metrics at both word and character level.
"""

from dataclasses import dataclass, field
from typing import List

import jiwer


@dataclass
class WERResult:
    """Detailed WER/CER result with error breakdown."""

    wer: float
    cer: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    ref_word_count: int
    hyp_word_count: int
    reference: str
    hypothesis: str
    aligned_ref: List[str] = field(default_factory=list)
    aligned_hyp: List[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Normalize text for fair comparison."""
    text = text.lower().strip()
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("'", " "):
            cleaned.append(ch)
    text = "".join(cleaned)
    return " ".join(text.split())


class WERCalculator:
    """Compute WER and CER between reference and hypothesis text."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, reference: str, hypothesis: str) -> WERResult:
        ref = _normalize(reference) if self.normalize else reference
        hyp = _normalize(hypothesis) if self.normalize else hypothesis

        if not ref and not hyp:
            return WERResult(
                wer=0.0, cer=0.0,
                substitutions=0, insertions=0, deletions=0, hits=0,
                ref_word_count=0, hyp_word_count=0,
                reference=ref, hypothesis=hyp,
            )
        if not ref:
            hyp_words = hyp.split()
            return WERResult(
                wer=float("inf"), cer=float("inf"),
                substitutions=0, insertions=len(hyp_words), deletions=0, hits=0,
                ref_word_count=0, hyp_word_count=len(hyp_words),
                reference=ref, hypothesis=hyp,
            )
        if not hyp:
            ref_words = ref.split()
            return WERResult(
                wer=1.0, cer=1.0,
                substitutions=0, insertions=0, deletions=len(ref_words), hits=0,
                ref_word_count=len(ref_words), hyp_word_count=0,
                reference=ref, hypothesis=hyp,
            )

        word_output = jiwer.process_words(ref, hyp)
        wer_val = word_output.wer
        cer_val = jiwer.cer(ref, hyp)

        aligned_ref = []
        aligned_hyp = []
        for chunk in word_output.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    aligned_ref.append(word_output.references[0][chunk.ref_start_idx + i])
                    aligned_hyp.append(word_output.hypotheses[0][chunk.hyp_start_idx + i])
            elif chunk.type == "substitute":
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    aligned_ref.append(word_output.references[0][chunk.ref_start_idx + i])
                for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                    aligned_hyp.append(word_output.hypotheses[0][chunk.hyp_start_idx + i])
            elif chunk.type == "delete":
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    aligned_ref.append(word_output.references[0][chunk.ref_start_idx + i])
                    aligned_hyp.append("***")
            elif chunk.type == "insert":
                for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                    aligned_ref.append("***")
                    aligned_hyp.append(word_output.hypotheses[0][chunk.hyp_start_idx + i])

        return WERResult(
            wer=wer_val,
            cer=cer_val,
            substitutions=word_output.substitutions,
            insertions=word_output.insertions,
            deletions=word_output.deletions,
            hits=word_output.hits,
            ref_word_count=len(ref.split()),
            hyp_word_count=len(hyp.split()),
            reference=ref,
            hypothesis=hyp,
            aligned_ref=aligned_ref,
            aligned_hyp=aligned_hyp,
        )
