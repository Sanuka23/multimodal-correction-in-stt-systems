"""Batch correction — sends whole transcript chunks to the model instead of word-by-word.

Old approach: 370 candidates × 7 sec each = ~43 minutes
New approach: ~20 chunks × 10 sec each = ~3 minutes

The model sees the full context and decides what to fix in one pass.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional, Tuple

from .config import CorrectionConfig
from .model import load_model, run_inference
from .types import CorrectionReport, CorrectionResult, CorrectionCandidate

logger = logging.getLogger(__name__)

# Max characters per chunk (model context is ~2048 tokens ≈ ~4000 chars)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def build_batch_prompt(
    transcript_chunk: str,
    vocab_terms: List[str],
    ocr_hints: Optional[List[str]] = None,
    lip_hint: Optional[str] = None,
) -> str:
    """Build a prompt for batch correction — asks for changes only, not full text.

    This is much faster because the model only outputs the diff (~256 tokens)
    instead of repeating the entire transcript (~2000 chars).
    """
    vocab_str = json.dumps(vocab_terms[:50])

    prompt = (
        "Review this ASR transcript and find words that are wrong based on the vocabulary.\n"
        "Return ONLY the changes as JSON. Do NOT repeat the full transcript.\n\n"
        f"ASR transcript: {transcript_chunk}\n"
        f"Vocabulary: {vocab_str}\n"
    )

    if ocr_hints:
        prompt += (
            "Screen text (OCR):\n"
            + "\n".join(f"  - {h}" for h in ocr_hints[:15]) + "\n"
        )

    if lip_hint and lip_hint != "null":
        prompt += f"Lip reading hint: {lip_hint}\n"

    prompt += (
        '\nRespond with JSON: {"changes": ["wrong_word → correct_word", ...], "confidence": 0.95}\n'
        'If no errors found, respond: {"changes": [], "confidence": 0.99}'
    )

    return prompt


# Use the same system prompt as training
BATCH_SYSTEM_PROMPT = (
    "You are an ASR transcript correction model. Given a noisy ASR transcript "
    "segment and context signals, detect errors in context-critical terms and "
    "output the corrected transcript with changes noted."
)


def _chunk_transcript(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """Split transcript into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [{"text": text, "start": 0, "end": len(text)}]

    chunks = []
    pos = 0

    while pos < len(text):
        end = min(pos + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end (. ! ?) near the chunk boundary
            for break_char in ['. ', '? ', '! ', '\n']:
                break_pos = text.rfind(break_char, pos + chunk_size - 300, end)
                if break_pos > pos:
                    end = break_pos + len(break_char)
                    break

        chunks.append({
            "text": text[pos:end],
            "start": pos,
            "end": end,
        })

        # Move forward with overlap
        pos = end - overlap if end < len(text) else end

    return chunks


def correct_transcript_batch(
    transcript: dict,
    file_id: str,
    vocab_terms: List[dict],
    ocr_provider=None,
    config: CorrectionConfig = None,
) -> Tuple[dict, CorrectionReport]:
    """Correct transcript by sending whole chunks to the model.

    Instead of checking each word individually (370 calls), sends ~20 chunks
    with the full vocabulary and lets the model find+fix all errors at once.
    """
    if config is None:
        config = CorrectionConfig()

    t0 = time.time()
    full_text = transcript.get("text", "")
    segments = transcript.get("segments", [])

    if not full_text:
        return transcript, CorrectionReport(
            file_id=file_id, corrections_attempted=0,
            corrections_applied=0, results=[], processing_time_ms=0,
        )

    # Load model
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

    # Build vocab term list for prompt
    term_names = list({t["term"] for t in vocab_terms})
    logger.info("Batch correction: %d vocab terms, %d chars transcript",
                len(term_names), len(full_text))

    # Get OCR hints (if provider available, get hints for whole video)
    ocr_hints = []
    if ocr_provider:
        try:
            ocr_xml = ocr_provider(file_id, 0, 99999)
            if ocr_xml:
                from .ocr_parser import parse_ocr_xml
                frames = parse_ocr_xml(ocr_xml)
                seen = set()
                for frame in frames:
                    text_content = frame.get("text", "")
                    for line in text_content.split("\n"):
                        line = line.strip()
                        if line and line not in seen and len(line) > 3:
                            seen.add(line)
                            ocr_hints.append(line)
        except Exception as e:
            logger.warning("OCR hints fetch failed: %s", e)

    # Chunk the transcript
    chunks = _chunk_transcript(full_text)
    logger.info("Split into %d chunks (avg %d chars)",
                len(chunks), sum(len(c["text"]) for c in chunks) // len(chunks))

    # Process each chunk
    corrected_text = full_text
    all_changes = []
    total_applied = 0
    results = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        logger.info("Processing chunk %d/%d (%d chars, pos %d-%d)...",
                     i + 1, len(chunks), len(chunk_text), chunk["start"], chunk["end"])

        if config.dry_run or model is None:
            logger.info("  Dry-run: skipping inference")
            continue

        # Build prompt and run inference — changes-only format, small output
        prompt = build_batch_prompt(chunk_text, term_names, ocr_hints)
        result_data = run_inference(
            prompt, BATCH_SYSTEM_PROMPT, model, tokenizer,
            max_tokens=256,  # Only need changes list, not full text
        )

        changes = result_data.get("changes", [])
        confidence = result_data.get("confidence", 0.0)

        if confidence < config.confidence_threshold and not changes:
            logger.info("  Chunk %d: skipped (confidence=%.2f, no changes)", i + 1, confidence)
            continue

        if changes:
            logger.info("  Chunk %d: %d changes (confidence=%.2f): %s",
                        i + 1, len(changes), confidence, changes[:5])

            # Apply changes to the full text
            for change in changes:
                if "→" in str(change):
                    parts = str(change).split("→")
                    if len(parts) == 2:
                        old_word = parts[0].strip().strip("'\"")
                        new_word = parts[1].strip().strip("'\"")

                        if old_word and new_word and old_word.lower() != new_word.lower():
                            # Apply with word boundaries
                            pattern = re.compile(
                                r'\b' + re.escape(old_word) + r'\b',
                                re.IGNORECASE,
                            )
                            new_text = pattern.sub(new_word, corrected_text, count=1)
                            if new_text != corrected_text:
                                corrected_text = new_text
                                total_applied += 1
                                all_changes.append(change)
                                logger.info("    Applied: '%s' → '%s'", old_word, new_word)

                                # Create CorrectionResult for report
                                results.append(CorrectionResult(
                                    candidate=CorrectionCandidate(
                                        term=new_word, category="batch",
                                        known_errors=[old_word], error_found=old_word,
                                        char_position=0, timestamp_start=0, timestamp_end=0,
                                        context=change,
                                    ),
                                    corrected_text=new_word,
                                    changes=[change],
                                    confidence=confidence,
                                    need_lip=False,
                                    ocr_hints_used=ocr_hints[:3],
                                    applied=True,
                                ))
        else:
            logger.info("  Chunk %d: no changes needed (confidence=%.2f)", i + 1, confidence)

    # Apply corrections to segments
    corrected_segments = [dict(s) for s in segments]
    if total_applied > 0:
        for change in all_changes:
            if "→" in str(change):
                parts = str(change).split("→")
                if len(parts) == 2:
                    old_word = parts[0].strip().strip("'\"")
                    new_word = parts[1].strip().strip("'\"")
                    pattern = re.compile(r'\b' + re.escape(old_word) + r'\b', re.IGNORECASE)
                    for seg in corrected_segments:
                        seg_new = pattern.sub(new_word, seg.get("text", ""), count=1)
                        if seg_new != seg.get("text", ""):
                            seg["text"] = seg_new

    # Build enhanced transcript
    enhanced = dict(transcript)
    enhanced["text"] = corrected_text
    enhanced["segments"] = corrected_segments

    duration_ms = (time.time() - t0) * 1000

    report = CorrectionReport(
        file_id=file_id,
        corrections_attempted=len(chunks),
        corrections_applied=total_applied,
        results=results,
        processing_time_ms=duration_ms,
    )

    logger.info("Batch correction complete: %d changes applied in %.0fms (%d chunks)",
                total_applied, duration_ms, len(chunks))

    return enhanced, report
