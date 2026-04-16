"""Target Term Error Rate (TTER) metric.

Computes a word-level error rate restricted to positions where the reference
word is in a user-supplied set of "target terms" (domain-relevant vocabulary).
See docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.7.
"""

from __future__ import annotations

from typing import Iterable, TypedDict

import jiwer


class TTERResult(TypedDict):
    tter: float
    substitutions: int
    deletions: int
    insertions: int
    target_total: int


def _normalize(tokens: Iterable[str]) -> list[str]:
    """Lowercase for case-insensitive comparison and target term matching."""
    return [t.lower() for t in tokens]


def compute_tter(
    reference: str,
    hypothesis: str,
    target_terms: set[str],
) -> TTERResult:
    """Compute Target Term Error Rate.

    Args:
        reference: ground-truth transcript text.
        hypothesis: predicted transcript text.
        target_terms: set of terms (case-insensitive) to restrict error counting to.

    Returns:
        Dict with `tter`, `substitutions`, `deletions`, `insertions`, `target_total`.
        `tter` is 0.0 when `target_total == 0` (undefined → 0 by convention).
    """
    target_lower = {t.lower() for t in target_terms}

    # Lowercase both sides before alignment so jiwer treats case differences
    # as equal (case-insensitive matching per §4.7).
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()

    alignment = jiwer.process_words(ref_lower, hyp_lower)
    # process_words returns a WordOutput with `.references`, `.hypotheses`,
    # and `.alignments`. Each alignment is a list of AlignmentChunk objects
    # per (ref, hyp) pair.

    ref_tokens = alignment.references[0]   # already lowercased
    hyp_tokens = alignment.hypotheses[0]   # already lowercased
    chunks = alignment.alignments[0]

    substitutions = 0
    deletions = 0
    insertions = 0
    target_total = sum(1 for t in ref_tokens if t in target_lower)

    for chunk in chunks:
        op = chunk.type  # one of: equal, substitute, delete, insert
        if op == "equal":
            continue
        if op == "substitute":
            # Reference positions [chunk.ref_start_idx : chunk.ref_end_idx]
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if ref_tokens[i] in target_lower:
                    substitutions += 1
        elif op == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if ref_tokens[i] in target_lower:
                    deletions += 1
        elif op == "insert":
            # No reference word — check if inserted hyp words are themselves targets
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if hyp_tokens[j] in target_lower:
                    insertions += 1

    if target_total == 0:
        tter = 0.0
    else:
        tter = (substitutions + deletions + insertions) / target_total

    return TTERResult(
        tter=tter,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        target_total=target_total,
    )
