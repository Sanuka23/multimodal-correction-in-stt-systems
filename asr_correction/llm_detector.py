"""LLM Pass 1 — Error Detection.

Sends transcript chunks + vocabulary to the LLM and gets back a list of
suspected ASR errors. Detection only — no corrections.

This replaces the rule-based segment_selector as the primary detector,
with rule-based kept as fallback if the LLM fails or returns nothing.

Design:
  - Chunks transcript at 5000 chars (larger than batch corrector since
    detection output is compact)
  - Structured JSON output: suspects list with word + likely_correct + needs_avsr
  - Falls back to rule-based select_segments_rules() on failure
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .config import CorrectionConfig
from .types import CorrectionCandidate
from .text_utils import extract_context, estimate_timestamp_for_position, find_occurrences

logger = logging.getLogger(__name__)

# Categories where OCR can help verify the term (visible on screen)
OCR_ELIGIBLE = {"product_name", "tech_term", "feature", "metric", "brand", "tool",
                "tech_acronym", "cloud_service", "ai_model", "platform", "compliance",
                "business_term", "business_metric", "domain_term"}

# Categories where AVSR (lip reading) can help verify the term
AVSR_ELIGIBLE = {"person_name", "content_word", "custom"}

# Detection prompt chunk size (larger than batch corrector since output is compact)
DETECT_CHUNK_SIZE = 5000
DETECT_CHUNK_OVERLAP = 200

DETECTION_SYSTEM_PROMPT = (
    "You are an ASR error detector for meeting transcripts. "
    "You identify words that are likely misrecognitions of domain-specific vocabulary. "
    "You output only valid JSON. You never flag common English words unless they "
    "clearly sound like a specific vocabulary term in context."
)


@dataclass
class DetectedError:
    """A suspected ASR error found by the LLM detector."""
    word: str
    likely_correct: str
    reason: str = ""
    needs_avsr: bool = False
    char_position: int = 0


def _build_detection_prompt(
    transcript_chunk: str,
    vocab_terms: List[dict],
) -> str:
    """Build the Pass 1 detection prompt — compact, detection-only."""
    # Include top 50 vocab terms with known errors for grounding
    term_entries = []
    for t in vocab_terms[:50]:
        errs = t.get("known_errors", [])
        if errs:
            term_entries.append(f"{t['term']} (often misheard as: {', '.join(errs[:3])})")
        else:
            term_entries.append(t["term"])
    vocab_str = json.dumps(term_entries)

    prompt = (
        "Review this ASR transcript and find words that are likely misrecognitions "
        "of the vocabulary terms below.\n\n"
        f"ASR transcript: {transcript_chunk}\n\n"
        f"Vocabulary: {vocab_str}\n\n"
        "Find words that SOUND LIKE a vocabulary term but are spelled differently. "
        "Do NOT flag words that are already correct. Do NOT flag common English words.\n\n"
        'Respond with JSON: {"suspects": [{"word": "quadrant", "likely_correct": "Qdrant"}, '
        '{"word": "sock two", "likely_correct": "SOC 2"}], "confidence": 0.9}\n'
        'If no errors: {"suspects": [], "confidence": 0.99}'
    )
    return prompt


def _chunk_text(text: str, chunk_size: int = DETECT_CHUNK_SIZE,
                overlap: int = DETECT_CHUNK_OVERLAP) -> List[dict]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [{"text": text, "start": 0, "end": len(text)}]

    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            for break_char in ['. ', '? ', '! ', '\n']:
                break_pos = text.rfind(break_char, pos + chunk_size - 500, end)
                if break_pos > pos:
                    end = break_pos + len(break_char)
                    break

        chunks.append({"text": text[pos:end], "start": pos, "end": end})
        pos = end - overlap if end < len(text) else end

    return chunks


def _parse_detection_response(response: str) -> List[dict]:
    """Parse the LLM detection JSON response. Robust to malformed output."""
    if not response:
        return []

    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(cleaned)

        suspects = data.get("suspects", data.get("errors", []))
        if not isinstance(suspects, list):
            return []

        # Validate each suspect has required fields
        valid = []
        for s in suspects:
            if isinstance(s, dict) and s.get("word") and s.get("likely_correct"):
                valid.append(s)
        return valid

    except (json.JSONDecodeError, ValueError, AttributeError):
        # Try to extract JSON from within the response
        match = re.search(r'\{[^{}]*"suspects"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("suspects", [])
            except Exception:
                pass
        logger.warning("Could not parse LLM detection response: %s", response[:200])
        return []


def _run_llm_detection(
    transcript: dict,
    vocab_terms: List[dict],
    model,
    tokenizer,
    config: CorrectionConfig,
) -> List[DetectedError]:
    """Run LLM detection on transcript chunks. Returns list of detected errors."""
    from .model import run_inference

    full_text = transcript.get("text", "")
    segments = transcript.get("segments", [])

    if not full_text:
        return []

    chunks = _chunk_text(full_text)
    all_detected = []

    logger.info("LLM Pass 1: scanning %d chunks (%d chars total)",
                len(chunks), len(full_text))

    for i, chunk in enumerate(chunks):
        prompt = _build_detection_prompt(chunk["text"], vocab_terms)

        result = run_inference(
            prompt, DETECTION_SYSTEM_PROMPT, model, tokenizer,
            max_tokens=256,
        )

        # The model may return suspects in the changes field or raw response
        raw = result.get("corrected", "") or ""
        changes = result.get("changes", [])

        # Try parsing the raw corrected text as detection JSON
        suspects = _parse_detection_response(raw)

        # If model returned changes in "word → correct" format, convert those
        if not suspects and changes:
            for change in changes:
                s = str(change)
                if "→" in s:
                    parts = s.split("→")
                    if len(parts) == 2:
                        suspects.append({
                            "word": parts[0].strip(),
                            "likely_correct": parts[1].strip(),
                        })

        if suspects:
            logger.info("  Chunk %d/%d: %d suspects found", i + 1, len(chunks), len(suspects))
        else:
            logger.info("  Chunk %d/%d: clean", i + 1, len(chunks))

        for s in suspects:
            word = s.get("word", "").strip()
            likely = s.get("likely_correct", "").strip()
            if not word or not likely:
                continue
            # Skip if word is too long (likely hallucination)
            if len(word) > 40 or len(likely) > 40:
                continue

            # Find character position in original text
            lower_text = full_text.lower()
            lower_word = word.lower()
            pos = lower_text.find(lower_word, chunk["start"])
            if pos == -1:
                pos = lower_text.find(lower_word)
            if pos == -1:
                pos = chunk["start"]  # Approximate

            all_detected.append(DetectedError(
                word=word,
                likely_correct=likely,
                reason=s.get("reason", ""),
                needs_avsr=s.get("needs_avsr", False),
                char_position=pos,
            ))

    return all_detected


def _detections_to_segment_analyses(
    detections: List[DetectedError],
    transcript: dict,
    vocab_terms: List[dict],
    config: CorrectionConfig,
) -> "List[SegmentAnalysis]":
    """Convert LLM detections to SegmentAnalysis objects (same format as rule-based)."""
    from .segment_selector import SegmentAnalysis

    full_text = transcript.get("text", "")
    segments = transcript.get("segments", [])
    vocab_map = {t["term"].lower(): t for t in vocab_terms}

    # Group detections by segment
    segment_detections: dict = {}  # segment_id → list of detections

    for det in detections:
        # Find which segment this detection belongs to
        ts = estimate_timestamp_for_position(full_text, det.char_position, segments)
        best_seg_id = 0
        for seg in segments:
            if seg.get("start", 0) <= ts <= seg.get("end", 0):
                best_seg_id = seg.get("id", 0)
                break

        if best_seg_id not in segment_detections:
            segment_detections[best_seg_id] = []
        segment_detections[best_seg_id].append((det, ts))

    # Build SegmentAnalysis objects
    analyses = []
    ocr_window = getattr(config, "ocr_window_seconds", 15.0)

    for seg in segments:
        seg_id = seg.get("id", 0)
        seg_dets = segment_detections.get(seg_id, [])

        candidates = []
        needs_ocr = False
        needs_avsr = False

        for det, ts in seg_dets:
            # Look up vocab term info
            term_info = vocab_map.get(det.likely_correct.lower())
            if not term_info:
                term_info = {
                    "term": det.likely_correct,
                    "category": "custom",
                    "known_errors": [det.word],
                }

            category = term_info.get("category", "custom")
            if category in OCR_ELIGIBLE:
                needs_ocr = True
            if category in AVSR_ELIGIBLE or det.needs_avsr:
                needs_avsr = True

            context = extract_context(full_text, det.char_position, 80)

            candidates.append(CorrectionCandidate(
                term=term_info["term"],
                category=category,
                known_errors=term_info.get("known_errors", [det.word]),
                error_found=det.word,
                char_position=det.char_position,
                timestamp_start=max(0, ts - ocr_window),
                timestamp_end=ts + ocr_window,
                context=context,
            ))

        analysis = SegmentAnalysis(
            segment_id=seg_id,
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            text=seg.get("text", ""),
            needs_correction=len(candidates) > 0,
            needs_ocr=needs_ocr,
            needs_avsr=needs_avsr,
            candidates=candidates,
            reasons=[f"llm_detected:{det.word}→{det.likely_correct}" for det, _ in seg_dets],
            rule_score=5.0 if candidates else 0.0,
            model_confirmed=True,
        )
        analyses.append(analysis)

    return analyses


def detect_errors(
    transcript: dict,
    vocab_terms: List[dict],
    model=None,
    tokenizer=None,
    config: CorrectionConfig = None,
) -> "List[SegmentAnalysis]":
    """LLM-based error detection with rule-based fallback.

    Primary: Sends transcript + vocab to LLM, gets suspected errors.
    Fallback: If LLM fails or returns nothing, uses rule-based segment selector.

    Returns List[SegmentAnalysis] — same type as select_segments() for compatibility.
    """
    from .segment_selector import select_segments

    if config is None:
        config = CorrectionConfig()

    # If no model available, go straight to rules
    if config.dry_run or model is None or tokenizer is None:
        logger.info("LLM Pass 1: skipped (dry_run or no model), using rule-based")
        return select_segments(
            transcript, vocab_terms, model, tokenizer,
            context_window=getattr(config, "context_window_chars", 80),
            ocr_window_seconds=getattr(config, "ocr_window_seconds", 15.0),
        )

    t0 = time.time()

    try:
        detections = _run_llm_detection(
            transcript, vocab_terms, model, tokenizer, config
        )

        if not detections:
            logger.info("LLM Pass 1: no errors detected (%.0fms), falling back to rules",
                       (time.time() - t0) * 1000)
            return select_segments(
                transcript, vocab_terms, model, tokenizer,
                context_window=getattr(config, "context_window_chars", 80),
                ocr_window_seconds=getattr(config, "ocr_window_seconds", 15.0),
            )

        analyses = _detections_to_segment_analyses(
            detections, transcript, vocab_terms, config
        )

        flagged = [a for a in analyses if a.needs_correction]
        total_candidates = sum(len(a.candidates) for a in flagged)

        logger.info(
            "LLM Pass 1: %d errors detected in %d segments, %d candidates (%.0fms)",
            len(detections), len(flagged), total_candidates,
            (time.time() - t0) * 1000,
        )
        return analyses

    except Exception as e:
        logger.error("LLM Pass 1 failed: %s, falling back to rules", e, exc_info=True)
        return select_segments(
            transcript, vocab_terms, model, tokenizer,
            context_window=getattr(config, "context_window_chars", 80),
            ocr_window_seconds=getattr(config, "ocr_window_seconds", 15.0),
        )
