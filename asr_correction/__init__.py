"""ScreenApp ASR Correction Module (v2 — Whisper Reconciliation).

The package is organised so each numbered stage lives in its own file —
this keeps the main pipeline readable and makes "where is the code for X?"
a one-glance question during reviews and vivas.

Pipeline stages
---------------
1.   Vocabulary merge          → :mod:`asr_correction.vocabulary`
2.   LLM error detection       → :mod:`asr_correction.llm_detector`
2.5  Topic classification      → :mod:`asr_correction.stages.topic_classification`
2.6  Web vocab enrichment      → :mod:`asr_correction.stages.web_vocab`
2.7  Candidate validation      → :mod:`asr_correction.stages.candidate_validation`
3.   OCR orchestration         → :mod:`asr_correction.stages.ocr_orchestration`
3.5  OCR vocab extraction      → :mod:`asr_correction.stages.ocr_vocab`
4.   Whisper Pass 2            → :mod:`asr_correction.whisper_pass2`
4.5  AVSR (lip-reading) hints  → :mod:`asr_correction.stages.avsr_hints`
5.   LLM reconciliation        → :mod:`asr_correction.reconciler`
6.   Training-data collection  → :mod:`asr_correction.data_collector`

Top-level orchestrator
----------------------
The full pipeline lives in :mod:`asr_correction.pipeline`; this package
re-exports its two public entry points:

- :func:`correct_transcript` — full correction pipeline
- :func:`analyze_transcript` — lightweight detect+classify path

Usage
-----
::

    from asr_correction import correct_transcript

    enhanced, report = correct_transcript(
        transcript=screenapp_output_transcript,
        file_id="abc-123",
        custom_vocabulary=[{"term": "ScreenApp", "category": "product_name"}],
        video_url="http://...",
    )
"""

import logging

from .config import CorrectionConfig
from .pipeline import analyze_transcript, correct_transcript
from .stages.avsr_hints import (
    AVSR_ELIGIBLE_CATEGORIES,
    gather_avsr_hints,
)
from .stages.candidate_validation import validate_candidates_via_web
from .stages.ocr_orchestration import (
    compute_ocr_timestamps,
    create_targeted_ocr_provider,
)
from .stages.ocr_vocab import extract_vocab_from_ocr
from .stages.topic_classification import classify_topic
from .stages.web_vocab import (
    ALLOWED_WEB_VOCAB_CATEGORIES,
    enrich_vocab_from_web,
    parse_web_vocab_json,
    truncate_at_sentence,
)
from .types import CorrectionReport

__version__ = "4.2.0"  # Codebase restructure — per-stage modules

logger = logging.getLogger(__name__)


# ── Backwards-compat aliases (underscored old names) ─────────────────────
# Older call sites (and the test suite) reach into these helpers via the
# private underscore-prefixed names that existed before the restructure.
# Keep aliases so nothing breaks; new code should import from the public
# names above (or directly from :mod:`asr_correction.stages`).
_ALLOWED_WEB_VOCAB_CATEGORIES = ALLOWED_WEB_VOCAB_CATEGORIES
_AVSR_ELIGIBLE_CATEGORIES = AVSR_ELIGIBLE_CATEGORIES
_truncate_at_sentence = truncate_at_sentence
_parse_web_vocab_json = parse_web_vocab_json
_enrich_vocab_from_web = enrich_vocab_from_web
_validate_candidates_via_web = validate_candidates_via_web
_extract_vocab_from_ocr = extract_vocab_from_ocr
_classify_topic = classify_topic
_gather_avsr_hints = gather_avsr_hints
_compute_ocr_timestamps = compute_ocr_timestamps
_create_targeted_ocr_provider = create_targeted_ocr_provider


__all__ = [
    "ALLOWED_WEB_VOCAB_CATEGORIES",
    "AVSR_ELIGIBLE_CATEGORIES",
    "CorrectionConfig",
    "CorrectionReport",
    "analyze_transcript",
    "classify_topic",
    "compute_ocr_timestamps",
    "correct_transcript",
    "create_targeted_ocr_provider",
    "enrich_vocab_from_web",
    "extract_vocab_from_ocr",
    "gather_avsr_hints",
    "parse_web_vocab_json",
    "truncate_at_sentence",
    "validate_candidates_via_web",
]
