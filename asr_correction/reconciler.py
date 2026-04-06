"""LLM Reconciliation — holistic correction using all gathered evidence.

The LLM receives two transcriptions of the same audio PLUS all evidence
gathered by the pipeline (error candidates, topic info, OCR, web vocab)
and produces the best corrected version.

No regex replacement — the LLM outputs the final segment text directly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RECONCILIATION_SYSTEM_PROMPT = (
    "You are an ASR transcript correction expert. You use multiple evidence sources "
    "(two transcriptions, vocabulary, OCR screen text, detected errors) to produce "
    "the best corrected version. Only fix misheard words — never add or remove content."
)


def _build_reconciliation_prompt(
    original_text: str,
    whisper_text: str,
    vocab_terms: List[str],
    ocr_hints: List[str],
    protected_terms: List[str] = None,
    error_candidates: List[dict] = None,
    topic_info: dict = None,
    web_vocab: List[dict] = None,
) -> str:
    """Build holistic correction prompt using all available evidence."""
    # Merge domain + web vocab into one deduped list
    all_vocab = list(vocab_terms or [])
    if web_vocab:
        for v in web_vocab:
            term = v.get("term", "")
            if term and term not in all_vocab:
                all_vocab.append(term)
    vocab_str = json.dumps(all_vocab[:80]) if all_vocab else "[]"

    # Topic context (compact — 1 line)
    topic_section = ""
    if topic_info and topic_info.get("topic", "unknown") != "unknown":
        topic_section = (
            f"Meeting context: {topic_info.get('field', '')} — {topic_info.get('topic', '')}\n\n"
        )

    # Error candidates for this segment (from Step 2 detection)
    candidates_section = ""
    if error_candidates:
        lines = []
        for ec in error_candidates[:10]:
            lines.append(f"  - '{ec['error_found']}' is likely '{ec['likely_correct']}' ({ec.get('category', '')})")
        if lines:
            candidates_section = (
                "Suspected ASR errors detected in this segment:\n"
                + "\n".join(lines) + "\n\n"
            )

    ocr_section = ""
    if ocr_hints:
        ocr_section = (
            "Screen text visible during this segment (from OCR):\n"
            + "\n".join(f"  - {h}" for h in ocr_hints[:15])
            + "\n\n"
        )

    protected_section = ""
    if protected_terms:
        protected_section = (
            "PROTECTED TERMS (confirmed by on-screen text — do NOT change these spellings):\n"
            + ", ".join(protected_terms)
            + "\nIf either version contains one of these terms spelled correctly, KEEP that spelling.\n\n"
        )

    return (
        "You have two ASR transcriptions of the SAME audio segment, plus multiple evidence sources. "
        "Produce the best corrected version.\n\n"
        f"{topic_section}"
        f"Version A (original): {original_text}\n\n"
        f"Version B (re-transcription): {whisper_text}\n\n"
        f"Known vocabulary (correct spellings): {vocab_str}\n\n"
        f"{candidates_section}"
        f"{protected_section}"
        f"{ocr_section}"
        "Rules:\n"
        "1. Start with Version A as the base text (keep its structure, length, and flow)\n"
        "2. Pick the best word from EITHER version, OR from the vocabulary list\n"
        "3. Use the suspected errors list as strong hints — if a word was flagged, "
        "strongly consider the suggested correction\n"
        "4. If a word sounds like a vocabulary term but is misspelled, replace it "
        "(e.g. 'OpenDI'→'OpenAI', 'Cloudware'→'Cloudflare', 'GROC'→'Groq')\n"
        "5. This applies even if BOTH versions have the same wrong word\n"
        "6. For protected terms: NEVER change these spellings\n"
        "7. Each swap must be exactly ONE word → ONE word\n"
        "8. NEVER add or remove words. Same word count as Version A\n"
        "9. NEVER change sentence structure or word order\n\n"
        'Respond with JSON:\n'
        '{"text": "the corrected version", "swaps": ["wrong → correct", ...]}\n'
        'If no corrections needed: {"text": "copy of version A unchanged", "swaps": []}'
    )


def _parse_response(response: str, original_text: str) -> dict:
    """Parse the LLM reconciliation JSON response."""
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            text = parsed.get("text", "")
            swaps = parsed.get("swaps", parsed.get("changes", []))
            confidence = float(parsed.get("confidence", 0.9))

            # Filter no-op swaps (old == new)
            real_swaps = []
            for s in swaps:
                if "→" in str(s):
                    parts = str(s).split("→")
                    if len(parts) == 2:
                        old = parts[0].strip().strip("'\"")
                        new = parts[1].strip().strip("'\"")
                        if old.lower() != new.lower() and old and new:
                            # Strict: only 1 word → 1 word swaps
                            if len(old.split()) > 1 or len(new.split()) > 1:
                                continue
                            real_swaps.append(f"{old} → {new}")

            if text and text.strip():
                return {"text": text.strip(), "swaps": real_swaps, "confidence": confidence}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    logger.debug("Could not parse reconciliation response: %s", response[:200])
    return {"text": original_text, "swaps": [], "confidence": 0.0}


def reconcile_segments(
    original_transcript: dict,
    whisper_segments: Dict[Tuple[float, float], str],
    vocab_terms: List[str],
    ocr_hints: List[str],
    model,
    tokenizer,
    config,
    protected_terms: List[str] = None,
    error_candidates: List[dict] = None,
    topic_info: dict = None,
    web_vocab: List[dict] = None,
) -> Tuple[dict, list]:
    """Holistic correction using all gathered evidence per segment.

    The LLM outputs the final segment text directly — no regex replacement.
    This prevents duplication bugs and phrase-level rewrites.

    Returns:
        (enhanced_transcript, all_swaps)
    """
    from .model import run_inference_raw

    t0 = time.time()
    enhanced = dict(original_transcript)
    enhanced_segments = [dict(s) for s in original_transcript.get("segments", [])]
    all_swaps = []

    if not whisper_segments:
        logger.info("Step 5: No Whisper segments to reconcile")
        enhanced["segments"] = enhanced_segments
        return enhanced, all_swaps

    segments = original_transcript.get("segments", [])
    logger.info("Step 5: Reconciling %d segments...", len(whisper_segments))

    for (ws_start, ws_end), whisper_text in whisper_segments.items():
        if not whisper_text:
            continue

        # Find original segment(s) that overlap this Whisper time range
        for i, seg in enumerate(segments):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            # Check overlap
            if seg_start >= ws_end or seg_end <= ws_start:
                continue

            original_text = seg.get("text", "").strip()
            if not original_text:
                continue

            # Skip if texts are identical (normalized)
            if original_text.lower() == whisper_text.lower():
                continue

            # Extract the relevant portion of Whisper text for this segment
            # Instead of sending the full 80s chunk, find the ~matching section
            # Use word count ratio to estimate position in the Whisper text
            whisper_portion = whisper_text
            orig_word_count = len(original_text.split())
            whisper_word_count = len(whisper_text.split())
            if whisper_word_count > orig_word_count * 3:
                # Whisper text is much longer — try to find the matching portion
                # Use fuzzy matching to find where this segment starts in Whisper text
                orig_words = original_text.lower().split()[:5]  # first 5 words
                whisper_words = whisper_text.lower().split()
                best_pos = 0
                best_score = 0
                for pos in range(len(whisper_words) - 3):
                    score = sum(1 for w in orig_words if w in whisper_words[pos:pos+10])
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                # Extract portion with some padding
                start = max(0, best_pos - 3)
                end = min(len(whisper_words), start + orig_word_count + 10)
                whisper_portion = " ".join(whisper_words[start:end])

            # Filter error candidates to this segment's time range
            seg_candidates = None
            if error_candidates:
                seg_candidates = [
                    ec for ec in error_candidates
                    if (ec.get("timestamp_start", 0) <= seg_end and
                        ec.get("timestamp_end", 0) >= seg_start)
                ]

            # Build and run reconciliation prompt with all evidence
            prompt = _build_reconciliation_prompt(
                original_text, whisper_portion, vocab_terms, ocr_hints,
                protected_terms=protected_terms,
                error_candidates=seg_candidates if seg_candidates else None,
                topic_info=topic_info,
                web_vocab=web_vocab,
            )

            raw = run_inference_raw(
                prompt, RECONCILIATION_SYSTEM_PROMPT,
                model, tokenizer, max_tokens=512,
            )

            result = _parse_response(raw, original_text)
            reconciled_text = result["text"]
            swaps = result["swaps"]
            confidence = result["confidence"]

            # Skip if no real swaps
            if not swaps:
                continue

            # ── Layer 2: Vocab-only filter ──
            # Only keep swaps where the word matches a known vocab term,
            # OCR term, protected term, or a Step 2 error candidate.
            all_known = set()
            for t in (vocab_terms or []):
                all_known.add(t.lower())
            for t in (protected_terms or []):
                all_known.add(t.lower())
            for h in (ocr_hints or []):
                for w in h.split():
                    if len(w) > 2 and w[0].isupper():
                        all_known.add(w.lower())
            # Include web vocab terms so web-sourced corrections pass the filter
            if web_vocab:
                for v in web_vocab:
                    term = v.get("term", "")
                    if term:
                        all_known.add(term.lower())

            # Build set of trusted swaps from Step 2 error candidates
            # The detector already validated these — trust them
            candidate_swaps = set()
            if seg_candidates:
                for ec in seg_candidates:
                    err = ec.get("error_found", "").lower().strip()
                    fix = ec.get("likely_correct", "").lower().strip()
                    if err and fix:
                        candidate_swaps.add((err, fix))
                        # Also add the correction target to all_known
                        all_known.add(fix)

            vocab_swaps = []
            for swap_str in swaps:
                if "→" not in swap_str:
                    continue
                parts = swap_str.split("→")
                old_word = parts[0].strip().strip("'\"").lower()
                new_word = parts[1].strip().strip("'\"").lower()

                # Check 1: Does this swap match a Step 2 error candidate?
                if (old_word, new_word) in candidate_swaps:
                    vocab_swaps.append(swap_str)
                    continue

                # Check 2: Does EITHER old or new word relate to a known vocab/OCR term?
                old_matches = old_word in all_known or any(old_word in k for k in all_known if len(k) > 3)
                new_matches = new_word in all_known or any(new_word in k for k in all_known if len(k) > 3)

                if old_matches or new_matches:
                    vocab_swaps.append(swap_str)
                else:
                    logger.info("    Filtered: %s (neither word in vocab/candidates)", swap_str)

            if not vocab_swaps:
                continue
            swaps = vocab_swaps

            # Sanity check: reconciled text shouldn't be drastically different in length
            orig_words = len(original_text.split())
            recon_words = len(reconciled_text.split())
            if abs(orig_words - recon_words) > max(3, orig_words * 0.3):
                logger.warning("  [%.1f-%.1fs] skipped — word count mismatch (orig=%d, recon=%d)",
                               seg_start, seg_end, orig_words, recon_words)
                continue

            # ── Layer 3: Protected terms post-check ──
            # If any protected term was in original but missing from reconciled, revert
            if protected_terms:
                orig_lower = original_text.lower()
                recon_lower = reconciled_text.lower()
                removed_term = None
                for term in protected_terms:
                    if term.lower() in orig_lower and term.lower() not in recon_lower:
                        removed_term = term
                        break
                if removed_term:
                    logger.warning("  [%.1f-%.1fs] skipped — protected term '%s' was removed",
                                   seg_start, seg_end, removed_term)
                    continue

            # Apply: directly replace the segment text
            enhanced_segments[i]["text"] = reconciled_text
            all_swaps.extend(swaps)

            logger.info("  [%.1f-%.1fs] %d swaps:", seg_start, seg_end, len(swaps))
            for swap in swaps:
                logger.info("    %s", swap)

    # Rebuild full text from segments
    enhanced["segments"] = enhanced_segments
    enhanced["text"] = " ".join(s.get("text", "") for s in enhanced_segments)

    duration_ms = (time.time() - t0) * 1000
    logger.info("Step 5 complete: %d swaps applied in %.0fms", len(all_swaps), duration_ms)

    return enhanced, all_swaps
