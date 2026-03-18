"""Segment Selector — Two-layer analysis to identify which segments need correction.

Layer 1: Fast rule-based scoring using vocab matching, ASR confidence, edit distance.
Layer 2: Lightweight model check on flagged segments to catch high-confidence ASR errors.

This replaces the old approach of processing OCR on the entire video upfront.
Instead, we first identify ~5-20% of segments that actually need attention,
then only fetch OCR/AVSR for those targeted time ranges.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import jellyfish
from rapidfuzz import fuzz, process

from .types import CorrectionCandidate
from .text_utils import estimate_timestamp_for_position, extract_context

logger = logging.getLogger(__name__)

# Categories where the term is likely visible on screen (slides, UI, code)
OCR_ELIGIBLE_CATEGORIES = {"product_name", "tech_term", "feature", "metric", "brand", "tool"}
# Categories where lip reading might help more than OCR
AVSR_ELIGIBLE_CATEGORIES = {"person_name", "content_word", "custom"}

# Minimum suspicion score to flag a segment
SEGMENT_FLAG_THRESHOLD = 3.0
# ASR confidence below this adds to suspicion (but is NOT conclusive alone)
LOW_CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Multi-signal plausibility scoring
# ---------------------------------------------------------------------------
# Instead of a blocklist, every (error_word, vocab_term) pair is scored
# across multiple signals. Only pairs with high plausibility become candidates.
# This prevents "and"→"Andre" while allowing "quadrant"→"Qdrant".

def _plausibility_score(error_word: str, term: str) -> tuple[float, list[str]]:
    """Score how plausible it is that `error_word` is an ASR error for `term`.

    Returns (score 0.0-1.0, list of reason strings).
    Higher = more likely a real ASR error.

    Key insight: ASR errors produce words of SIMILAR length and sound.
    "quadrant"→"Qdrant" is plausible (similar length, same sound).
    "and"→"Andre" is NOT (short word is just a prefix of the term).

    Signals:
      1. Phonetic similarity (Metaphone)
      2. Normalized edit distance
      3. Word length ratio — heavily penalize large length differences
      4. Substring penalty — if error is a prefix/substring of term, penalize
    """
    e = error_word.lower().strip(".,!?;:'\"()[]")
    t = term.lower()

    if not e or not t:
        return 0.0, []

    # Identical words → perfect score (already correct)
    if e == t:
        return 1.0, ["exact_match"]

    score = 0.0
    reasons = []

    # --- PENALTY: Substring/prefix check ---
    # If the error word is just a prefix or substring of the term,
    # it's almost certainly a common word, not an ASR error.
    # e.g., "and" is a prefix of "Andre", "can" is a prefix of "Canva"
    if t.startswith(e) or e.startswith(t):
        # Only a real error if the words are very similar in length
        len_diff = abs(len(e) - len(t))
        if len_diff > 1:
            reasons.append(f"substring_penalty:diff={len_diff}")
            return 0.05, reasons  # Very low score — almost certainly false positive

    # --- PENALTY: Short words ---
    # Short common words match too many vocab terms by coincidence.
    # Apply strict rules based on word length:
    len_ratio = min(len(e), len(t)) / max(len(e), len(t))
    edit_dist = jellyfish.levenshtein_distance(e, t)

    if len(e) <= 3:
        # Very short words (≤3 chars): ONLY accept exact match or plural
        # "rag"→"RAG" OK, "LLM"→"LLMs" OK, "but"→"Bud" REJECT
        if edit_dist > 0 and not (t.startswith(e) and len(t) - len(e) <= 1):
            reasons.append(f"very_short_word:len={len(e)},edit={edit_dist}")
            return 0.05, reasons
    elif len(e) <= 4:
        # Short words (4 chars): allow edit distance 1 only if same length
        # "data"→"ATLA" REJECT (edit=2), "rack"→"RAG" REJECT (diff length)
        if edit_dist > 1 or len_ratio < 0.75:
            reasons.append(f"short_word:len={len(e)},edit={edit_dist},ratio={len_ratio:.2f}")
            return 0.1, reasons

    # Signal 1: Phonetic similarity via Metaphone (weight: 0.35)
    phonetic_score = 0.0
    try:
        e_metaphone = jellyfish.metaphone(e)
        t_metaphone = jellyfish.metaphone(t)
        if e_metaphone and t_metaphone:
            # Full metaphone comparison using edit distance on codes
            meta_dist = jellyfish.levenshtein_distance(e_metaphone, t_metaphone)
            meta_max = max(len(e_metaphone), len(t_metaphone))
            phonetic_score = 1.0 - (meta_dist / meta_max if meta_max > 0 else 1.0)
            if phonetic_score > 0.3:
                score += 0.35 * phonetic_score
                reasons.append(f"phonetic:{e_metaphone}≈{t_metaphone}({phonetic_score:.2f})")
    except Exception:
        pass

    # Signal 2: Normalized edit distance (weight: 0.30)
    edit_dist = jellyfish.levenshtein_distance(e, t)
    max_len = max(len(e), len(t))
    edit_similarity = 1.0 - (edit_dist / max_len if max_len > 0 else 1.0)
    if edit_similarity > 0.4:
        score += 0.30 * edit_similarity
        reasons.append(f"edit_sim:{edit_similarity:.2f}")

    # Signal 3: Word length ratio (weight: 0.20)
    if len_ratio >= 0.65:
        score += 0.20 * len_ratio
        reasons.append(f"len_ratio:{len_ratio:.2f}")

    # Signal 4: Character overlap (weight: 0.15)
    e_chars = set(e)
    t_chars = set(t)
    if e_chars | t_chars:
        char_overlap = len(e_chars & t_chars) / len(e_chars | t_chars)
        if char_overlap >= 0.4:
            score += 0.15 * char_overlap
            reasons.append(f"char_overlap:{char_overlap:.2f}")

    return score, reasons


# Minimum plausibility to treat (error, term) as a real ASR error
PLAUSIBILITY_THRESHOLD = 0.65


@dataclass
class SegmentAnalysis:
    """Result of analyzing one transcript segment."""

    segment_id: int
    start: float
    end: float
    text: str
    needs_correction: bool = False
    needs_ocr: bool = False
    needs_avsr: bool = False
    candidates: List[CorrectionCandidate] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    rule_score: float = 0.0
    model_confirmed: bool = False


def _build_term_lookup(vocab_terms: List[dict]) -> dict:
    """Build fast lookup structures from vocabulary.

    Returns dict with:
        terms: {lowered_term: term_info}
        errors: {lowered_error: term_info}
        all_terms: [term strings for fuzzy matching]
    """
    terms = {}
    errors = {}
    all_terms = []

    for t in vocab_terms:
        term = t["term"]
        lowered = term.lower()
        terms[lowered] = t
        all_terms.append(term)

        for err in t.get("known_errors", []):
            errors[err.lower()] = t

    return {"terms": terms, "errors": errors, "all_terms": all_terms}


def _score_word(
    word: str,
    probability: float,
    lookup: dict,
) -> tuple[float, list[str], Optional[dict]]:
    """Score a single word for suspicion using multi-signal plausibility.

    Instead of a blocklist, every (word, vocab_term) pair is scored for
    phonetic similarity, edit distance, length ratio, and character overlap.
    Only plausible matches become candidates.

    Returns (score, reasons, matched_term_info).
    """
    score = 0.0
    reasons = []
    matched_term = None
    w_lower = word.lower().strip(".,!?;:'\"()[]")

    if not w_lower or len(w_lower) < 2:
        return 0.0, [], None

    # Signal 1: Known error exact match — but verify plausibility
    if w_lower in lookup["errors"]:
        term_info = lookup["errors"][w_lower]
        plausibility, plaus_reasons = _plausibility_score(w_lower, term_info["term"])

        if plausibility >= PLAUSIBILITY_THRESHOLD:
            # Plausible ASR error (e.g., "quadrant"→"Qdrant")
            score += 5
            matched_term = term_info
            reasons.append(f"known_error:{term_info['term']}(plaus={plausibility:.2f})")
            reasons.extend(plaus_reasons)
        else:
            # Implausible match (e.g., "and"→"Andre") — skip
            logger.debug("Rejected known_error '%s'→'%s' (plausibility=%.2f < %.2f)",
                        w_lower, term_info["term"], plausibility, PLAUSIBILITY_THRESHOLD)

    # Signal 2: Fuzzy match to vocab term — with plausibility check
    if not matched_term and lookup["all_terms"]:
        matches = process.extract(
            w_lower, [t.lower() for t in lookup["all_terms"]],
            scorer=fuzz.ratio, limit=3, score_cutoff=60,
        )
        for match_term_lower, match_score, _ in (matches or []):
            if match_term_lower == w_lower:
                continue  # Already correct
            term_info = lookup["terms"].get(match_term_lower)
            if not term_info:
                continue

            plausibility, plaus_reasons = _plausibility_score(w_lower, term_info["term"])
            if plausibility >= PLAUSIBILITY_THRESHOLD:
                matched_term = term_info
                score += 3
                reasons.append(f"vocab_near_miss:{match_term_lower}(plaus={plausibility:.2f})")
                reasons.extend(plaus_reasons)
                break  # Use first plausible match

    # Signal 3: Low ASR confidence (+2) — only adds to score, never standalone
    if probability < LOW_CONFIDENCE_THRESHOLD:
        score += 2
        reasons.append(f"low_confidence:{probability:.2f}")

    # Signal 4: Proper noun near vocab term — with plausibility check
    if word[0].isupper() and len(word) > 1 and not matched_term:
        matches = process.extract(
            w_lower, [t.lower() for t in lookup["all_terms"]],
            scorer=fuzz.ratio, limit=1, score_cutoff=60,
        )
        if matches:
            match_term_lower, match_score, _ = matches[0]
            if match_term_lower != w_lower:
                term_info = lookup["terms"].get(match_term_lower)
                if term_info:
                    plausibility, plaus_reasons = _plausibility_score(w_lower, term_info["term"])
                    if plausibility >= PLAUSIBILITY_THRESHOLD:
                        matched_term = term_info
                        score += 2
                        reasons.append(f"proper_noun:{match_term_lower}(plaus={plausibility:.2f})")
                        reasons.extend(plaus_reasons)

    return score, reasons, matched_term


def select_segments_rules(
    transcript: dict,
    vocab_terms: List[dict],
    context_window: int = 80,
    ocr_window_seconds: float = 15.0,
) -> List[SegmentAnalysis]:
    """Layer 1: Rule-based segment selection.

    Analyzes each segment's words against vocabulary using multiple signals.
    Returns list of SegmentAnalysis with flagged segments.
    """
    segments = transcript.get("segments", [])
    full_text = transcript.get("text", "")

    if not segments or not full_text:
        return []

    lookup = _build_term_lookup(vocab_terms)
    analyses = []

    for seg in segments:
        seg_id = seg.get("id", 0)
        seg_text = seg.get("text", "")
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        words = seg.get("words", [])

        analysis = SegmentAnalysis(
            segment_id=seg_id,
            start=seg_start,
            end=seg_end,
            text=seg_text,
        )

        # Score each word in the segment
        max_score = 0.0
        seg_reasons = []

        for w_entry in words:
            word = w_entry.get("word", "")
            prob = w_entry.get("probability", 1.0)

            word_score, word_reasons, matched_term = _score_word(word, prob, lookup)

            if word_score > 0:
                seg_reasons.extend(word_reasons)

            if word_score > max_score:
                max_score = word_score

            # If word is suspicious enough and we have a matched vocab term,
            # create a CorrectionCandidate
            if word_score >= SEGMENT_FLAG_THRESHOLD and matched_term:
                # Find char position in full text
                # Search for this word near the segment's text position
                seg_text_pos = full_text.find(seg_text)
                if seg_text_pos >= 0:
                    word_pos_in_seg = seg_text.lower().find(word.lower())
                    if word_pos_in_seg >= 0:
                        char_pos = seg_text_pos + word_pos_in_seg
                    else:
                        char_pos = seg_text_pos
                else:
                    char_pos = 0

                context = extract_context(full_text, char_pos, context_window)
                ts = estimate_timestamp_for_position(full_text, char_pos, segments)
                ts_start = max(0, ts - ocr_window_seconds)
                ts_end = ts + ocr_window_seconds

                candidate = CorrectionCandidate(
                    term=matched_term["term"],
                    category=matched_term.get("category", "custom"),
                    known_errors=matched_term.get("known_errors", []),
                    error_found=word,
                    char_position=char_pos,
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    context=context,
                )
                analysis.candidates.append(candidate)

        analysis.rule_score = max_score
        analysis.reasons = seg_reasons

        if max_score >= SEGMENT_FLAG_THRESHOLD:
            analysis.needs_correction = True

            # Decide OCR vs AVSR based on term categories
            for c in analysis.candidates:
                if c.category in OCR_ELIGIBLE_CATEGORIES:
                    analysis.needs_ocr = True
                if c.category in AVSR_ELIGIBLE_CATEGORIES:
                    analysis.needs_avsr = True

            # Default: if no specific modality decided, use OCR
            if not analysis.needs_ocr and not analysis.needs_avsr:
                analysis.needs_ocr = True

        analyses.append(analysis)

    return analyses


def select_segments_with_model(
    analyses: List[SegmentAnalysis],
    vocab_terms: List[dict],
    model,
    tokenizer,
    max_segments: int = 50,
) -> List[SegmentAnalysis]:
    """Layer 2: Lightweight model check on segments near the threshold.

    For segments that scored between 1-2 (below threshold but have some signal),
    run a quick model check to catch high-confidence ASR errors.

    Also confirms flagged segments to reduce false positives.
    """
    if model is None or tokenizer is None:
        logger.info("Model not available for segment selection — using rules only")
        return analyses

    # Find borderline segments (have some signal but below threshold)
    borderline = [a for a in analyses if 0 < a.rule_score < SEGMENT_FLAG_THRESHOLD]

    if not borderline:
        logger.info("No borderline segments to model-check")
        return analyses

    # Limit to avoid spending too long
    borderline = borderline[:max_segments]
    logger.info("Layer 2: Model-checking %d borderline segments", len(borderline))

    # Build vocab list string for prompt
    term_list = [t["term"] for t in vocab_terms[:50]]  # Limit vocab in prompt
    vocab_str = json.dumps(term_list)

    from .model import run_inference

    system_prompt = (
        "You are an ASR error detector. Given a transcript segment and vocabulary, "
        "identify if any words are likely ASR errors (wrong words that sound similar "
        "to the correct vocabulary terms). Respond with JSON only."
    )

    for analysis in borderline:
        try:
            prompt = (
                f"Does this ASR segment contain errors?\n"
                f"Segment: \"{analysis.text}\"\n"
                f"Vocabulary: {vocab_str}\n"
                f"Respond with: {{\"has_error\": true/false, \"suspect_words\": "
                f"[{{\"word\": \"...\", \"should_be\": \"...\"}}]}}"
            )

            result = run_inference(
                prompt, system_prompt, model, tokenizer, max_tokens=128
            )

            # Check if model found errors
            if result.get("corrected") or result.get("changes"):
                analysis.needs_correction = True
                analysis.model_confirmed = True
                analysis.needs_ocr = True  # Default to OCR for model-flagged
                analysis.reasons.append("model_flagged")
                logger.info("  Model flagged segment %d: '%s'",
                           analysis.segment_id, analysis.text[:60])

        except Exception as e:
            logger.debug("Model check failed for segment %d: %s", analysis.segment_id, e)

    return analyses


def select_segments(
    transcript: dict,
    vocab_terms: List[dict],
    model=None,
    tokenizer=None,
    context_window: int = 80,
    ocr_window_seconds: float = 15.0,
) -> List[SegmentAnalysis]:
    """Main entry point: run both layers of segment selection.

    Args:
        transcript: ScreenApp transcript with segments and word-level data.
        vocab_terms: Merged vocabulary list.
        model: Optional model for Layer 2 check.
        tokenizer: Optional tokenizer for Layer 2 check.

    Returns:
        List of SegmentAnalysis for all segments, with flagged ones marked.
    """
    import time
    t0 = time.time()

    # Layer 1: Rule-based scoring
    analyses = select_segments_rules(
        transcript, vocab_terms, context_window, ocr_window_seconds
    )

    flagged = [a for a in analyses if a.needs_correction]
    needs_ocr = [a for a in analyses if a.needs_ocr]
    needs_avsr = [a for a in analyses if a.needs_avsr]

    logger.info("Segment selector Layer 1 (rules): %d/%d segments flagged "
                "(ocr=%d, avsr=%d) in %.0fms",
                len(flagged), len(analyses), len(needs_ocr), len(needs_avsr),
                (time.time() - t0) * 1000)

    # Layer 2: Model check on borderline segments
    if model is not None:
        t1 = time.time()
        analyses = select_segments_with_model(analyses, vocab_terms, model, tokenizer)

        flagged_after = [a for a in analyses if a.needs_correction]
        new_flags = len(flagged_after) - len(flagged)
        if new_flags > 0:
            logger.info("Segment selector Layer 2 (model): +%d segments flagged in %.0fms",
                        new_flags, (time.time() - t1) * 1000)

    total_flagged = [a for a in analyses if a.needs_correction]
    total_candidates = sum(len(a.candidates) for a in total_flagged)
    logger.info("Segment selector total: %d/%d segments, %d candidates",
                len(total_flagged), len(analyses), total_candidates)

    return analyses
