"""Per-stage modules for the ASR correction pipeline.

Each module owns one numbered stage of the pipeline so that, during a viva
or code review, "where is X?" has a one-glance answer.

Stage map
---------
- :mod:`asr_correction.stages.topic_classification` — Stage 2.5
- :mod:`asr_correction.stages.web_vocab`            — Stage 2.6
- :mod:`asr_correction.stages.candidate_validation` — Stage 2.7
- :mod:`asr_correction.stages.ocr_orchestration`    — Stage 3 (helpers)
- :mod:`asr_correction.stages.ocr_vocab`            — Stage 3.5
- :mod:`asr_correction.stages.avsr_hints`           — Stage 4.5

The orchestrator that calls these modules in order lives in
:mod:`asr_correction.pipeline`.
"""

from .avsr_hints import AVSR_ELIGIBLE_CATEGORIES, gather_avsr_hints
from .candidate_validation import validate_candidates_via_web
from .ocr_orchestration import compute_ocr_timestamps, create_targeted_ocr_provider
from .ocr_vocab import extract_vocab_from_ocr
from .topic_classification import classify_topic
from .web_vocab import (
    ALLOWED_WEB_VOCAB_CATEGORIES,
    enrich_vocab_from_web,
    parse_web_vocab_json,
    truncate_at_sentence,
)

__all__ = [
    "AVSR_ELIGIBLE_CATEGORIES",
    "ALLOWED_WEB_VOCAB_CATEGORIES",
    "classify_topic",
    "compute_ocr_timestamps",
    "create_targeted_ocr_provider",
    "enrich_vocab_from_web",
    "extract_vocab_from_ocr",
    "gather_avsr_hints",
    "parse_web_vocab_json",
    "truncate_at_sentence",
    "validate_candidates_via_web",
]
