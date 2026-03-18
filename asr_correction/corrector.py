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

    # Import plausibility scoring to filter false positives
    from .segment_selector import _plausibility_score, PLAUSIBILITY_THRESHOLD

    for term_info in vocab_terms:
        term = term_info["term"]
        category = term_info["category"]
        known_errors = term_info.get("known_errors", [])

        if not known_errors:
            continue

        for err in known_errors:
            # Multi-signal plausibility check: is this a plausible ASR error?
            # Rejects "and"→"Andre" (low phonetic/edit similarity)
            # Accepts "quadrant"→"Qdrant" (high phonetic/edit similarity)
            plausibility, _ = _plausibility_score(err, term)
            if plausibility < PLAUSIBILITY_THRESHOLD:
                logger.debug("Skipping implausible error '%s'→'%s' (plausibility=%.2f)",
                            err, term, plausibility)
                continue
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
    corrected_segments = [dict(s) for s in transcript.get("segments", [])]
    applied_count = 0

    # Process candidates in reverse char-position order to preserve offsets
    sorted_candidates = sorted(
        candidates, key=lambda c: c.char_position, reverse=True
    )

    for idx, candidate in enumerate(sorted_candidates):
        logger.info("--- Processing candidate %d/%d: '%s' (error='%s') ---",
                     idx + 1, len(sorted_candidates), candidate.term, candidate.error_found)

        # Step 1: Get OCR hints
        ocr_hints = _fetch_ocr_hints(candidate, file_id, ocr_provider, config)
        logger.info("  OCR hints: %d hints fetched %s",
                     len(ocr_hints), ocr_hints[:3] if ocr_hints else "[]")

        # Step 2: Build prompt and run inference
        if config.dry_run:
            logger.info("  Mode: dry-run (rule-based)")
            result_data = _dry_run_correction(candidate)
        else:
            prompt = build_prompt(
                candidate.context,
                [candidate.term],
                candidate.category,
                ocr_hints,
            )
            logger.info("  Mode: ML inference")
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
                logger.info("  Decision: SKIP (need_lip=True, confidence=%.2f < 0.8)", confidence)
            else:
                should_apply = True
        elif not changes and candidate.error_found.lower() != candidate.term.lower():
            # Model returned no changes. Check if the model's corrected
            # output already contains the correct term — if so, the text
            # is fine and we should trust the model. If not, force-apply.
            corrected_output = (result_data.get("corrected") or "").lower()
            # Use word boundaries to avoid matching inside other words
            term_pattern = re.compile(r'\b' + re.escape(candidate.term.lower()) + r'\b')
            error_pattern = re.compile(r'\b' + re.escape(candidate.error_found.lower()) + r'\b')
            term_already_present = bool(term_pattern.search(corrected_output))
            error_still_present = bool(error_pattern.search(corrected_output))

            if term_already_present:
                # Model output already has the correct term → text is fine
                logger.info("  Decision: SKIP — model output already contains '%s' (confidence=%.2f)",
                             candidate.term, confidence)
            elif error_still_present:
                # Error still in model output, correct term absent → force apply
                should_apply = True
                changes = [f"{candidate.error_found} → {candidate.term}"]
                logger.info("  Decision: Force-apply — error '%s' still in model output, term '%s' absent",
                             candidate.error_found, candidate.term)
            elif confidence < config.confidence_threshold:
                # Model unsure (low confidence, no clear output) → SKIP
                # Do NOT force-apply when the model can't confirm the error
                logger.info("  Decision: SKIP — model unsure (confidence=%.2f), not applying", confidence)
            else:
                # Model confident, term not found but error also gone → trust model
                logger.info("  Decision: SKIP — model confident, error resolved (confidence=%.2f)", confidence)

        if should_apply:
            # Use word boundaries to avoid replacing inside other words
            # e.g. 'SOC' should NOT match inside 'process', 'associated', etc.
            pattern = re.compile(
                r'\b' + re.escape(candidate.error_found) + r'\b',
                re.IGNORECASE,
            )
            # Apply on flat text
            new_text = pattern.sub(candidate.term, corrected_text, count=1)
            if new_text != corrected_text:
                corrected_text = new_text
                # Also apply directly on the matching segment to avoid
                # offset-drift from _rebuild_segments when text length changes
                for seg in corrected_segments:
                    seg_new = pattern.sub(candidate.term, seg.get("text", ""), count=1)
                    if seg_new != seg.get("text", ""):
                        seg["text"] = seg_new
                        break  # count=1: only first match
                applied = True
                applied_count += 1
                logger.info("  Result: APPLIED — '%s' → '%s' (confidence=%.2f)",
                             candidate.error_found, candidate.term, confidence)
            else:
                logger.info("  Result: NO CHANGE — pattern not found as whole word in text")
        else:
            if not should_apply and confidence < config.confidence_threshold:
                logger.info("  Result: SKIPPED — confidence %.2f < threshold %.2f",
                             confidence, config.confidence_threshold)
            elif not changes:
                logger.info("  Result: SKIPPED — no changes suggested")

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
    enhanced["segments"] = corrected_segments

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

    total_ms = (time.time() - t0) * 1000
    logger.info("=== Correction Summary for %s ===", file_id)
    logger.info("  Total candidates: %d | Applied: %d | Skipped: %d | Time: %.0fms",
                len(candidates), applied_count, len(candidates) - applied_count, total_ms)

    report = CorrectionReport(
        file_id=file_id,
        corrections_attempted=len(candidates),
        corrections_applied=applied_count,
        results=results,
        processing_time_ms=total_ms,
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
