"""Whisper Pass 2 — re-transcribe flagged segments with vocabulary-biased prompt.

After the LLM corrects the transcript (Step 4), this step re-runs Whisper on
only the flagged segments with an `initial_prompt` containing domain vocabulary.
If Whisper agrees with the LLM correction → high confidence, keep it.
If Whisper disagrees → revert to original (conservative).

This provides an acoustic signal to validate LLM corrections.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Singleton Whisper model
_whisper_model = None
_whisper_lock = threading.Lock()

_FFMPEG_TIMEOUT = 30  # seconds per segment extraction


def _get_whisper_model(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """Lazy-initialize faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                logger.info("Loading faster-whisper model: %s (device=%s, compute=%s)",
                            model_size, device, compute_type)
                _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info("Whisper model loaded")
    return _whisper_model


def extract_audio_segment(
    video_url: str,
    start_s: float,
    end_s: float,
    output_path: str,
) -> bool:
    """Extract audio segment from video using FFmpeg.

    Outputs 16kHz mono WAV (what Whisper expects).
    Uses -ss before -i for fast seeking.
    """
    duration = end_s - start_s
    if duration <= 0:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.2f}",
        "-i", video_url,
        "-t", f"{duration:.2f}",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning("FFmpeg audio extraction failed: %s", result.stderr[:200])
            return False
        return Path(output_path).exists() and Path(output_path).stat().st_size > 0
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg audio extraction timed out for [%.1f-%.1f]", start_s, end_s)
        return False
    except Exception as e:
        logger.warning("Audio extraction error: %s", e)
        return False


def build_initial_prompt(
    topic_vocab: List[str] = None,
    custom_vocab: List[str] = None,
    ocr_names: List[str] = None,
    web_vocab: List[str] = None,
    max_chars: int = 896,
) -> str:
    """Build Whisper initial_prompt from all vocab sources.

    Priority (most specific first):
    1. Speaker names from OCR
    2. Custom vocabulary terms
    3. Topic-specific vocab from classification
    4. Web-enriched vocab

    Whisper's initial_prompt limit is ~224 tokens ≈ ~896 chars.
    """
    parts = []

    # Speaker names first (most impactful for name correction)
    if ocr_names:
        names = ", ".join(ocr_names[:10])
        parts.append(f"Speakers: {names}.")

    # Custom vocabulary
    if custom_vocab:
        terms = ", ".join(custom_vocab[:30])
        parts.append(f"Key terms: {terms}.")

    # Topic vocab
    if topic_vocab:
        terms = ", ".join(topic_vocab[:20])
        parts.append(f"Topics: {terms}.")

    # Web vocab
    if web_vocab:
        terms = ", ".join(web_vocab[:15])
        parts.append(f"Related: {terms}.")

    prompt = " ".join(parts)

    # Truncate to max chars
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars].rsplit(" ", 1)[0] + "."

    return prompt


def retranscribe_segment(
    video_url: str,
    start_s: float,
    end_s: float,
    initial_prompt: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Optional[str]:
    """Re-transcribe a single audio segment with faster-whisper.

    Returns the transcribed text, or None on failure.
    """
    model = _get_whisper_model(model_size, device, compute_type)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not extract_audio_segment(video_url, start_s, end_s, tmp_path):
            return None

        segments, info = model.transcribe(
            tmp_path,
            initial_prompt=initial_prompt,
            beam_size=5,
            language="en",
        )

        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip() if text.strip() else None

    except Exception as e:
        logger.warning("Whisper transcription failed for [%.1f-%.1f]: %s", start_s, end_s, e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@dataclass
class WhisperPass2Result:
    """Result of Whisper Pass 2 on a single correction."""
    original_word: str
    llm_correction: str
    whisper_has_correction: bool  # Whisper output contains the LLM correction
    whisper_has_original: bool    # Whisper output contains the original word
    agreement: str                # "agree", "disagree", "unclear"
    segment_start: float = 0.0
    segment_end: float = 0.0
    whisper_text: str = ""


def _normalize(text: str) -> str:
    """Normalize text for comparison — lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def compare_corrections(
    original_text: str,
    llm_corrected_text: str,
    whisper_text: str,
    corrections: List[Tuple[str, str]],  # [(original_word, corrected_word), ...]
) -> List[WhisperPass2Result]:
    """Compare LLM corrections with Whisper Pass 2 output, word by word.

    For each correction X→Y:
    - If Whisper text contains Y (normalized): AGREE
    - If Whisper text contains X but not Y: DISAGREE
    - Otherwise: UNCLEAR (keep LLM correction)
    """
    from rapidfuzz import fuzz

    whisper_norm = _normalize(whisper_text)
    results = []

    for orig_word, corrected_word in corrections:
        orig_norm = _normalize(orig_word)
        corr_norm = _normalize(corrected_word)

        # Check if whisper output contains the corrected word (fuzzy)
        has_correction = (
            corr_norm in whisper_norm or
            fuzz.partial_ratio(corr_norm, whisper_norm) > 85
        )

        # Check if whisper output contains the original word
        has_original = (
            orig_norm in whisper_norm or
            fuzz.partial_ratio(orig_norm, whisper_norm) > 85
        )

        if has_correction:
            agreement = "agree"
        elif has_original and not has_correction:
            agreement = "disagree"
        else:
            agreement = "unclear"

        results.append(WhisperPass2Result(
            original_word=orig_word,
            llm_correction=corrected_word,
            whisper_has_correction=has_correction,
            whisper_has_original=has_original,
            agreement=agreement,
            whisper_text=whisper_text[:200],
        ))

    return results


def _merge_time_ranges(ranges: List[Tuple[float, float]], gap_s: float = 5.0) -> List[Tuple[float, float]]:
    """Merge overlapping or close time ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def run_whisper_pass2(
    video_url: str,
    original_transcript: dict,
    enhanced_transcript: dict,
    report,  # CorrectionReport from Step 4
    initial_prompt: str,
    config,  # CorrectionConfig
) -> Tuple[dict, List[WhisperPass2Result]]:
    """Run Whisper Pass 2 on segments where Step 4 made corrections.

    Returns (possibly_modified_transcript, whisper_results).
    """
    t0 = time.time()
    all_results = []

    if not report.results:
        logger.info("Step 4.5: No corrections to verify — skipping Whisper Pass 2")
        return enhanced_transcript, all_results

    # Collect corrections with their approximate timestamps
    segments = original_transcript.get("segments", [])
    full_text = original_transcript.get("text", "")
    corrections_by_segment = {}  # (start, end) → [(orig, corrected), ...]

    for result in report.results:
        if not result.applied:
            continue

        # Find the segment this correction belongs to
        error = result.candidate.error_found
        corrected = result.corrected_text

        # Use char_position to find timestamp
        char_pos = result.candidate.char_position
        seg_start, seg_end = 0.0, 0.0

        for seg in segments:
            seg_text = seg.get("text", "")
            if error.lower() in seg_text.lower() or corrected.lower() in seg_text.lower():
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                break

        if seg_end <= seg_start:
            continue

        # Add padding
        padding = config.whisper_segment_padding_s
        key = (max(0, seg_start - padding), seg_end + padding)
        if key not in corrections_by_segment:
            corrections_by_segment[key] = []
        corrections_by_segment[key].append((error, corrected))

    if not corrections_by_segment:
        logger.info("Step 4.5: Could not map corrections to segments — skipping")
        return enhanced_transcript, all_results

    # Merge overlapping time ranges
    time_ranges = list(corrections_by_segment.keys())
    merged_ranges = _merge_time_ranges(time_ranges)

    # Limit segments
    merged_ranges = merged_ranges[:config.whisper_max_segments]

    logger.info("Step 4.5: Whisper Pass 2 — %d segments to re-transcribe", len(merged_ranges))

    # Re-transcribe each segment
    reverted_count = 0
    agreed_count = 0
    unclear_count = 0

    for start_s, end_s in merged_ranges:
        logger.info("  Whisper re-transcribing [%.1f-%.1fs]...", start_s, end_s)

        whisper_text = retranscribe_segment(
            video_url, start_s, end_s, initial_prompt,
            model_size=config.whisper_model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        if not whisper_text:
            logger.warning("  Whisper returned no text for [%.1f-%.1f]", start_s, end_s)
            continue

        logger.info("  Whisper output: %s", whisper_text[:150])

        # Find all corrections that fall in this time range
        segment_corrections = []
        for (ts, te), corrs in corrections_by_segment.items():
            if ts >= start_s and te <= end_s + 5:  # Allow some slack
                segment_corrections.extend(corrs)

        if not segment_corrections:
            continue

        # Compare each correction with Whisper output
        results = compare_corrections(
            full_text, enhanced_transcript.get("text", ""),
            whisper_text, segment_corrections,
        )

        for r in results:
            r.segment_start = start_s
            r.segment_end = end_s

            if r.agreement == "agree":
                agreed_count += 1
                logger.info("    AGREE: '%s' → '%s' (Whisper confirms)", r.original_word, r.llm_correction)
            elif r.agreement == "disagree":
                # Revert this correction
                enhanced_text = enhanced_transcript.get("text", "")
                reverted_text = re.sub(
                    r'\b' + re.escape(r.llm_correction) + r'\b',
                    r.original_word,
                    enhanced_text,
                    count=1,
                )
                if reverted_text != enhanced_text:
                    enhanced_transcript["text"] = reverted_text
                    # Also revert in segments
                    for seg in enhanced_transcript.get("segments", []):
                        seg_text = seg.get("text", "")
                        seg_new = re.sub(
                            r'\b' + re.escape(r.llm_correction) + r'\b',
                            r.original_word, seg_text, count=1,
                        )
                        if seg_new != seg_text:
                            seg["text"] = seg_new
                            break
                    reverted_count += 1
                    logger.info("    REVERT: '%s' → '%s' (Whisper disagrees, reverting to '%s')",
                                r.original_word, r.llm_correction, r.original_word)
            else:
                unclear_count += 1
                logger.info("    UNCLEAR: '%s' → '%s' (Whisper ambiguous, keeping LLM)",
                            r.original_word, r.llm_correction)

        all_results.extend(results)

    duration_ms = (time.time() - t0) * 1000
    logger.info("Step 4.5 complete: %d agreed, %d reverted, %d unclear (%.0fms)",
                agreed_count, reverted_count, unclear_count, duration_ms)

    return enhanced_transcript, all_results
