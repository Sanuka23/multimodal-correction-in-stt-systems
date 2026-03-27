"""Whisper Pass 2 — re-transcribe flagged segments with vocabulary-biased prompt.

Extracts audio for only the flagged segments, runs faster-whisper with an
initial_prompt packed with domain vocabulary. The output is compared against
the original transcript by the LLM reconciler (reconciler.py).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Singleton Whisper model
_whisper_model = None
_whisper_lock = threading.Lock()

_FFMPEG_TIMEOUT = 30  # seconds per segment extraction


def _get_whisper_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
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
    """Extract audio segment from video using FFmpeg (16kHz mono WAV)."""
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
        result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
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

    if ocr_names:
        names = ", ".join(ocr_names[:10])
        parts.append(f"Speakers: {names}.")

    if custom_vocab:
        terms = ", ".join(custom_vocab[:30])
        parts.append(f"Key terms: {terms}.")

    if topic_vocab:
        terms = ", ".join(topic_vocab[:20])
        parts.append(f"Topics: {terms}.")

    if web_vocab:
        terms = ", ".join(web_vocab[:15])
        parts.append(f"Related: {terms}.")

    prompt = " ".join(parts)

    if len(prompt) > max_chars:
        prompt = prompt[:max_chars].rsplit(" ", 1)[0] + "."

    return prompt


def retranscribe_segment(
    video_url: str,
    start_s: float,
    end_s: float,
    initial_prompt: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Optional[str]:
    """Re-transcribe a single audio segment with faster-whisper."""
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


def _merge_time_ranges(ranges: List[Tuple[float, float]], gap_s: float = 5.0) -> List[Tuple[float, float]]:
    """Merge overlapping or close time ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [list(sorted_ranges[0])]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def retranscribe_flagged_segments(
    video_url: str,
    flagged_segments: List[Tuple[float, float]],
    initial_prompt: str,
    config,
) -> Dict[Tuple[float, float], str]:
    """Re-transcribe all flagged segments with Whisper.

    Args:
        video_url: URL of the video
        flagged_segments: List of (start_s, end_s) tuples to re-transcribe
        initial_prompt: Vocabulary-packed prompt for Whisper
        config: CorrectionConfig with whisper settings

    Returns:
        Dict mapping (start_s, end_s) → whisper_text
    """
    t0 = time.time()

    if not flagged_segments:
        return {}

    # Add padding and merge overlapping segments
    padding = config.whisper_segment_padding_s
    padded = [(max(0, s - padding), e + padding) for s, e in flagged_segments]
    merged = _merge_time_ranges(padded)
    merged = merged[:config.whisper_max_segments]

    logger.info("Step 4: Whisper Pass 2 — %d segments to re-transcribe (from %d flagged)",
                len(merged), len(flagged_segments))
    logger.info("  initial_prompt (%d chars): %s", len(initial_prompt), initial_prompt[:150])

    results = {}
    for start_s, end_s in merged:
        logger.info("  Whisper [%.1f-%.1fs] (%0.1fs)...", start_s, end_s, end_s - start_s)

        whisper_text = retranscribe_segment(
            video_url, start_s, end_s, initial_prompt,
            model_size=config.whisper_model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        if whisper_text:
            results[(start_s, end_s)] = whisper_text
            logger.info("  → %s", whisper_text[:150])
        else:
            logger.warning("  → (no output)")

    duration_ms = (time.time() - t0) * 1000
    logger.info("Step 4 complete: %d/%d segments transcribed in %.0fms",
                len(results), len(merged), duration_ms)

    return results
