"""Type definitions for the ASR correction module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


# --- ScreenApp OutputTranscript types ---

class WordEntry(TypedDict, total=False):
    word: str
    start: float
    end: float
    speaker: str
    probability: float
    language: str
    hallucination_score: float


class SegmentEntry(TypedDict, total=False):
    id: int
    start: float
    end: float
    text: str
    speaker: str
    words: List[WordEntry]


class OutputTranscript(TypedDict, total=False):
    task: str
    language: str
    text: str
    words: List[WordEntry]
    segments: List[SegmentEntry]
    compact_segments: list
    meta: dict
    request_id: str


# --- Correction internal types ---

@dataclass
class CorrectionCandidate:
    """A term occurrence in the transcript that may need correction."""

    term: str
    category: str
    known_errors: List[str]
    error_found: str
    char_position: int
    timestamp_start: float
    timestamp_end: float
    context: str


@dataclass
class CorrectionResult:
    """Result of running the model on a single candidate."""

    candidate: CorrectionCandidate
    corrected_text: str
    changes: List[str]
    confidence: float
    need_lip: bool
    ocr_hints_used: List[str]
    applied: bool


@dataclass
class CorrectionReport:
    """Full correction report for one transcript."""

    file_id: str
    corrections_attempted: int
    corrections_applied: int
    results: List[CorrectionResult]
    processing_time_ms: float
