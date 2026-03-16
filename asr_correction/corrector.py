"""Main correction orchestrator.

Scans a transcript for error candidates, requests OCR for problem areas,
runs the correction model, and applies fixes.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Tuple, Union

from .config import CorrectionConfig
from .model import build_prompt, load_model, run_inference
from .ocr_parser import (
    OCRCallback,
    OCRProvider,
    extract_hints_from_frames,
    parse_ocr_xml,
)
from .text_utils import (
    estimate_timestamp_for_position,
    extract_context,
    find_occurrences,
    normalize,
)
from .types import CorrectionCandidate, CorrectionReport, CorrectionResult

logger = logging.getLogger(__name__)


def identify_candidates(
    transcript: dict,
    vocab_terms: List[dict],
    config: CorrectionConfig,
) -> List[CorrectionCandidate]:
    """Scan transcript for terms that may have ASR errors.

    For each vocab term with known_errors, check if any known error
    pattern appears in the transcript text.
    """
    full_text = transcript.get("text", "")
    segments = transcript.get("segments", [])

    if not full_text:
        return []

    candidates = []

    for term_info in vocab_terms:
        term = term_info["term"]
        category = term_info["category"]
        known_errors = term_info.get("known_errors", [])

        if not known_errors:
            continue

        for err in known_errors:
            err_positions = find_occurrences(full_text, err)
            for pos_start, pos_end in err_positions:
                # Skip if the correct term is actually at this position
                context = extract_context(
                    full_text, pos_start, config.context_window_chars
                )
                if len(context) < config.min_context_length:
                    continue

                # Estimate timestamp for OCR request
                ts = estimate_timestamp_for_position(full_text, pos_start, segments)
                ts_start = max(0, ts - config.ocr_window_seconds)
                ts_end = ts + config.ocr_window_seconds

                candidates.append(
                    CorrectionCandidate(
                        term=term,
                        category=category,
                        known_errors=known_errors,
                        error_found=err,
                        char_position=pos_start,
                        timestamp_start=ts_start,
                        timestamp_end=ts_end,
                        context=context,
                    )
                )

    return candidates


def correct_candidates(
    candidates: List[CorrectionCandidate],
    transcript: dict,
    file_id: str,
    ocr_provider: Optional[Union[OCRProvider, OCRCallback]],
    config: CorrectionConfig,
) -> Tuple[dict, CorrectionReport]:
    """Run correction model on each candidate and apply results.

    1. For each candidate, request OCR if provider is available
    2. Build prompt with context + vocab + OCR hints
    3. Run model inference (or dry-run rule-based)
    4. Apply correction if confidence >= threshold
    5. Return enhanced transcript + report
    """
    t0 = time.time()

    model, tokenizer = None, None
    if not config.dry_run:
        try:
            model, tokenizer = load_model(
                adapter_path=config.adapter_path,
                model_path=config.model_path,
                base_model=config.base_model,
                backend=config.backend,
            )
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            model, tokenizer = None, None
        # If model failed to load, fall back to dry run
        if model is None:
            logger.warning("Model not available, falling back to dry-run mode.")
            config.dry_run = True

    results: List[CorrectionResult] = []
    corrected_text = transcript.get("text", "")
    applied_count = 0

    # Process candidates in reverse char-position order to preserve offsets
    sorted_candidates = sorted(
        candidates, key=lambda c: c.char_position, reverse=True
    )

    for candidate in sorted_candidates:
        # Step 1: Get OCR hints
        ocr_hints = _fetch_ocr_hints(candidate, file_id, ocr_provider, config)

        # Step 2: Build prompt and run inference
        if config.dry_run:
            result_data = _dry_run_correction(candidate)
        else:
            prompt = build_prompt(
                candidate.context,
                [candidate.term],
                candidate.category,
                ocr_hints,
            )
            result_data = run_inference(
                prompt, config.system_prompt, model, tokenizer, config.max_tokens
            )

        # Step 3: Decide whether to apply
        confidence = result_data.get("confidence", 0.0)
        changes = result_data.get("changes", [])
        applied = False

        should_apply = False
        if confidence >= config.confidence_threshold and changes:
            # Model explicitly suggested changes
            if result_data.get("need_lip", False) and confidence < 0.8:
                pass
            else:
                should_apply = True
        elif not changes and candidate.error_found.lower() != candidate.term.lower():
            # Model returned no changes, but we have a known error→term mapping.
            # Force-apply the vocab correction (the error was already matched).
            should_apply = True
            changes = [f"{candidate.error_found} → {candidate.term}"]

        if should_apply:
            pattern = re.compile(re.escape(candidate.error_found), re.IGNORECASE)
            new_text = pattern.sub(candidate.term, corrected_text, count=1)
            if new_text != corrected_text:
                corrected_text = new_text
                applied = True
                applied_count += 1

        results.append(
            CorrectionResult(
                candidate=candidate,
                corrected_text=result_data.get("corrected", ""),
                changes=changes,
                confidence=confidence,
                need_lip=result_data.get("need_lip", False),
                ocr_hints_used=ocr_hints,
                applied=applied,
            )
        )

    # Build enhanced transcript
    enhanced = dict(transcript)
    enhanced["text"] = corrected_text

    # Rebuild segments with corrected text
    if corrected_text != transcript.get("text", ""):
        enhanced["segments"] = _rebuild_segments(
            transcript.get("segments", []),
            transcript.get("text", ""),
            corrected_text,
        )

    # Add correction metadata to meta
    if "meta" not in enhanced:
        enhanced["meta"] = {}
    enhanced["meta"]["asr_correction"] = {
        "version": "1.0.0",
        "corrections_attempted": len(candidates),
        "corrections_applied": applied_count,
        "dry_run": config.dry_run,
        "confidence_threshold": config.confidence_threshold,
    }

    report = CorrectionReport(
        file_id=file_id,
        corrections_attempted=len(candidates),
        corrections_applied=applied_count,
        results=results,
        processing_time_ms=(time.time() - t0) * 1000,
    )

    return enhanced, report


def _fetch_ocr_hints(
    candidate: CorrectionCandidate,
    file_id: str,
    ocr_provider: Optional[Union[OCRProvider, OCRCallback]],
    config: CorrectionConfig,
) -> List[str]:
    """Fetch and parse OCR hints for a candidate."""
    if ocr_provider is None:
        return []

    try:
        raw_ocr = None
        if callable(ocr_provider) and not isinstance(ocr_provider, OCRProvider):
            raw_ocr = ocr_provider(
                file_id, candidate.timestamp_start, candidate.timestamp_end
            )
        elif isinstance(ocr_provider, OCRProvider):
            raw_ocr = ocr_provider.get_ocr(
                file_id, candidate.timestamp_start, candidate.timestamp_end
            )

        if raw_ocr:
            frames = parse_ocr_xml(raw_ocr)
            center_ts = (candidate.timestamp_start + candidate.timestamp_end) / 2
            return extract_hints_from_frames(
                frames,
                center_ts,
                window=config.ocr_window_seconds,
                max_hints=config.max_ocr_hints,
            )
    except Exception as e:
        logger.warning("OCR fetch failed for %s: %s", candidate.term, e)

    return []


def _dry_run_correction(candidate: CorrectionCandidate) -> dict:
    """Rule-based correction for dry-run mode."""
    corrected = candidate.context.replace(
        normalize(candidate.error_found), candidate.term
    )
    return {
        "corrected": corrected,
        "changes": [f"{candidate.error_found} \u2192 {candidate.term}"],
        "confidence": 0.95,
        "need_lip": False,
    }


def _rebuild_segments(
    original_segments: list, original_text: str, corrected_text: str
) -> list:
    """Rebuild segments after text correction.

    Maps corrections from the flat text back to individual segments
    using character-offset tracking.
    """
    if original_text == corrected_text:
        return original_segments

    new_segments = []
    offset = 0

    for seg in original_segments:
        seg_text = seg.get("text", "")
        idx = original_text.find(seg_text, offset)
        if idx == -1:
            new_segments.append(dict(seg))
            continue

        seg_end = idx + len(seg_text)
        new_seg = dict(seg)
        new_seg["text"] = corrected_text[idx:seg_end]
        new_segments.append(new_seg)
        offset = seg_end

    return new_segments
