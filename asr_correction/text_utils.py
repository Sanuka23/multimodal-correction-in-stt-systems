"""Text processing utilities for ASR correction."""

from __future__ import annotations

import re
from typing import List, Tuple


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def find_occurrences(
    text: str, term: str, word_boundary: bool = True
) -> List[Tuple[int, int]]:
    """Find all occurrences of a term in text.

    Uses regex word boundaries for short terms (<=4 chars) to avoid
    false positives like 'Al' inside 'also'.
    """
    norm_text = normalize(text)
    norm_term = normalize(term)
    positions = []

    if word_boundary and len(norm_term) <= 4:
        pattern = re.compile(r'\b' + re.escape(norm_term) + r'\b')
        for match in pattern.finditer(norm_text):
            positions.append((match.start(), match.end()))
    else:
        start = 0
        while True:
            idx = norm_text.find(norm_term, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(norm_term)))
            start = idx + 1

    return positions


def extract_context(text: str, position: int, window: int = 80) -> str:
    """Extract a context window around a character position."""
    norm = normalize(text)
    start = max(0, position - window)
    end = min(len(norm), position + window)
    return norm[start:end]


def estimate_timestamp_for_position(
    text: str, char_pos: int, segments: list
) -> float:
    """Estimate the timestamp for a character position in the full text.

    Maps character offset to time by walking through segments and
    interpolating within the matching segment.
    """
    if not segments:
        return 0.0

    running_offset = 0
    for seg in segments:
        seg_text = seg.get("text", "")
        seg_len = len(seg_text)
        if running_offset + seg_len >= char_pos:
            ratio = (char_pos - running_offset) / max(seg_len, 1)
            start = seg.get("start", 0.0)
            end = seg.get("end", start)
            return start + ratio * (end - start)
        running_offset += seg_len + 1  # +1 for space between segments

    # Fallback to last segment's end
    return segments[-1].get("end", 0.0)
