"""LLM Reconciliation — compare original vs Whisper transcription per segment.

The LLM compares two transcriptions of the same audio and outputs the best
combined version. It keeps the structure of the original but swaps in better
words from the Whisper version where appropriate.

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
    "You are an ASR transcript reconciliation expert. You compare two transcriptions "
    "of the same audio and produce the best combined version. You output the final "
    "text directly — never add or remove content, only fix misheard words."
)


def _build_reconciliation_prompt(
    original_text: str,
    whisper_text: str,
    vocab_terms: List[str],
    ocr_hints: List[str],
) -> str:
    """Build prompt for comparing two transcriptions."""
    vocab_str = json.dumps(vocab_terms[:50]) if vocab_terms else "[]"

    ocr_section = ""
    if ocr_hints:
        ocr_section = (
            "Screen text visible during this segment (from OCR):\n"
            + "\n".join(f"  - {h}" for h in ocr_hints[:10])
            + "\n\n"
        )

    return (
        "You have two ASR transcriptions of the SAME audio segment. "
        "Produce the best combined version.\n\n"
        f"Version A (original): {original_text}\n\n"
        f"Version B (re-transcription): {whisper_text}\n\n"
        f"Known vocabulary: {vocab_str}\n\n"
        f"{ocr_section}"
        "Rules:\n"
        "1. Start with Version A as the base text (keep its structure, length, and flow)\n"
        "2. Only SWAP individual misheard words — never rewrite phrases or sentences\n"
        "3. For names: if Version B has a name that matches OCR or vocabulary, use it\n"
        "4. For tech terms: if Version B matches known vocabulary better, use it\n"
        "5. For common words: keep Version A unless Version B is clearly better\n"
        "6. NEVER add or remove words. The output must have the same number of words as Version A\n"
        "7. NEVER change sentence structure or word order\n\n"
        'Respond with JSON:\n'
        '{"text": "the final best version based on Version A with word swaps from B", '
        '"swaps": ["wordA → wordB", ...]}\n'
        'If no swaps needed: {"text": "copy of version A unchanged", "swaps": []}'
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
) -> Tuple[dict, list]:
    """Compare original vs Whisper transcription per segment, output best version.

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

            # Find the portion of Whisper text that corresponds to this segment
            # (Whisper may cover multiple segments in one chunk)
            # Use the segment's original text length as a rough guide
            whisper_portion = whisper_text  # Use full Whisper output for context

            # Build and run reconciliation prompt
            prompt = _build_reconciliation_prompt(
                original_text, whisper_portion, vocab_terms, ocr_hints,
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

            # Sanity check: reconciled text shouldn't be drastically different in length
            orig_words = len(original_text.split())
            recon_words = len(reconciled_text.split())
            if abs(orig_words - recon_words) > max(3, orig_words * 0.3):
                logger.warning("  [%.1f-%.1fs] skipped — word count mismatch (orig=%d, recon=%d)",
                               seg_start, seg_end, orig_words, recon_words)
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
