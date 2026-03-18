"""OCR extraction from video files using PaddleOCR PP-OCRv4.

Extracts frames via a single FFmpeg call with fps filter, runs PaddleOCR
on visually unique frames (with unnecessary models disabled for speed),
deduplicates by text similarity, and outputs XML compatible with ocr_parser.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Dict, List, Optional, Set

# Skip slow PaddleOCR model source connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logger = logging.getLogger(__name__)

# Singleton PaddleOCR instance
_ocr_engine = None
_ocr_lock = threading.Lock()


def _get_ocr_engine():
    """Lazy-initialize PaddleOCR singleton with only needed models."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR PP-OCRv4 (lightweight mode)...")
        _ocr_engine = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv4",
            use_doc_orientation_classify=False,  # Not needed for screen captures
            use_doc_unwarping=False,             # Not needed for digital content
            use_textline_orientation=False,       # Screen text is always horizontal
        )
        logger.info("PaddleOCR engine ready (det + rec only)")
    return _ocr_engine


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def _ocr_single_frame(frame, min_confidence: float = 0.5) -> Optional[Dict]:
    """Run PaddleOCR on a single ExtractedFrame.

    Returns dict with timestamp_s, timestamp (MM:SS), and texts list,
    or None if no text found.
    """
    ocr = _get_ocr_engine()
    with _ocr_lock:
        results = ocr.predict(frame.image)

    if not results:
        return None

    texts = []

    # PaddleOCR v3.4 returns list[OCRResult]
    # Each OCRResult has rec_texts (list[str]) and rec_scores (list[float])
    for result_item in results:
        rec_texts = None
        rec_scores = None

        # Try dict-like access (OCRResult supports both)
        try:
            if hasattr(result_item, "get"):
                rec_texts = result_item.get("rec_texts", None)
                rec_scores = result_item.get("rec_scores", None)
            else:
                rec_texts = getattr(result_item, "rec_texts", None)
                rec_scores = getattr(result_item, "rec_scores", None)
        except (AttributeError, TypeError):
            pass

        if rec_texts and rec_scores:
            for text, confidence in zip(rec_texts, rec_scores):
                conf = float(confidence)
                txt = str(text).strip()
                if conf >= min_confidence and txt:
                    texts.append({"text": txt, "confidence": conf})

    if not texts:
        return None

    return {
        "timestamp_s": frame.timestamp_s,
        "timestamp": _format_timestamp(frame.timestamp_s),
        "texts": texts,
    }


def _text_similarity(texts_a: List[str], texts_b: List[str]) -> float:
    """Compute Jaccard similarity between two lists of text strings."""
    words_a: Set[str] = set()
    words_b: Set[str] = set()
    for t in texts_a:
        words_a.update(t.lower().split())
    for t in texts_b:
        words_b.update(t.lower().split())

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _deduplicate_frames(
    ocr_results: List[Dict], threshold: float = 0.9
) -> List[Dict]:
    """Remove consecutive frames with highly similar OCR text."""
    if not ocr_results:
        return []

    deduped = [ocr_results[0]]
    for result in ocr_results[1:]:
        prev_texts = [t["text"] for t in deduped[-1]["texts"]]
        curr_texts = [t["text"] for t in result["texts"]]
        if _text_similarity(prev_texts, curr_texts) < threshold:
            deduped.append(result)

    logger.info(
        "Text dedup: %d → %d frames (removed %d similar)",
        len(ocr_results), len(deduped), len(ocr_results) - len(deduped),
    )
    return deduped


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _frames_to_xml(ocr_results: List[Dict]) -> str:
    """Convert OCR results to <ocr-extraction> XML format."""
    lines = ["<ocr-extraction>"]
    for result in ocr_results:
        ts = result["timestamp"]
        text_content = "\n".join(_escape_xml(t["text"]) for t in result["texts"])
        lines.append(f'  <frame timestamp="{ts}" type="slide">')
        lines.append(f"    <text>{text_content}</text>")
        lines.append("  </frame>")
    lines.append("</ocr-extraction>")
    return "\n".join(lines)


def _extract_ocr_sync(
    video_url: str,
    interval_s: float = 30.0,
    max_frames: int = 100,
    min_confidence: float = 0.5,
    dedup_threshold: float = 0.9,
) -> Optional[str]:
    """Synchronous OCR extraction pipeline.

    1. Single FFmpeg call extracts all frames (fps filter)
    2. Pixel dedup: skip visually identical frames
    3. PaddleOCR on each unique frame (det + rec only, no preprocessing)
    4. Text dedup: merge frames with similar OCR text
    5. Format as XML
    """
    from .video_frames import extract_frames_periodic

    frames = extract_frames_periodic(
        video_url, interval_s=interval_s, max_frames=max_frames
    )
    if not frames:
        logger.warning("No frames extracted from video")
        return None

    logger.info("Running PaddleOCR on %d unique frames...", len(frames))

    ocr_results = []
    for idx, frame in enumerate(frames):
        if idx % 5 == 0:
            logger.info("  OCR frame %d/%d (ts=%.0fs)...",
                        idx + 1, len(frames), frame.timestamp_s)
        result = _ocr_single_frame(frame, min_confidence=min_confidence)
        if result:
            text_preview = result["texts"][0]["text"][:50] if result["texts"] else ""
            logger.debug("  Frame %.0fs: %d texts found ('%s...')",
                         frame.timestamp_s, len(result["texts"]), text_preview)
            ocr_results.append(result)

    if not ocr_results:
        logger.info("OCR found no text in any frame")
        return None

    logger.info("OCR found text in %d / %d frames", len(ocr_results), len(frames))

    deduped = _deduplicate_frames(ocr_results, threshold=dedup_threshold)

    xml = _frames_to_xml(deduped)
    logger.info("OCR extraction complete: %d unique text frames, %d chars of XML",
                len(deduped), len(xml))
    return xml


def extract_ocr_for_segments(
    video_url: str,
    time_ranges: List[tuple],
    padding_s: float = 5.0,
    min_confidence: float = 0.5,
) -> Dict[float, List[str]]:
    """Extract OCR only for specific time ranges (targeted extraction).

    Instead of processing the entire video, only extract frames within
    the given time windows. Much faster for long videos.

    Args:
        video_url: URL or path to video file.
        time_ranges: List of (start_s, end_s) tuples for segments needing OCR.
        padding_s: Extra seconds before/after each range.
        min_confidence: Min PaddleOCR confidence.

    Returns:
        Dict mapping timestamp (float seconds) → list of OCR text strings.
    """
    from .video_frames import extract_frames_at_timestamps

    if not time_ranges:
        return {}

    # Merge overlapping time ranges with padding
    merged = _merge_time_ranges(time_ranges, padding_s)
    logger.info("Targeted OCR: %d ranges → %d merged windows", len(time_ranges), len(merged))

    # Collect timestamps to extract (one frame per 5s within each range)
    timestamps = []
    for start, end in merged:
        t = start
        while t <= end:
            timestamps.append(t)
            t += 5.0  # One frame every 5 seconds within the window

    if not timestamps:
        return {}

    logger.info("Targeted OCR: extracting %d frames (vs ~%d for full video)",
                len(timestamps), int(max(t for t in timestamps) / 30) if timestamps else 0)

    # Extract frames at specific timestamps
    frames = extract_frames_at_timestamps(video_url, timestamps)
    if not frames:
        logger.warning("No frames extracted for targeted OCR")
        return {}

    # Run OCR on each frame
    logger.info("Running PaddleOCR on %d targeted frames...", len(frames))
    results = {}
    for idx, frame in enumerate(frames):
        if idx % 5 == 0:
            logger.info("  Targeted OCR frame %d/%d (ts=%.0fs)...",
                        idx + 1, len(frames), frame.timestamp_s)
        ocr_result = _ocr_single_frame(frame, min_confidence=min_confidence)
        if ocr_result:
            texts = [t["text"] for t in ocr_result["texts"]]
            results[frame.timestamp_s] = texts

    logger.info("Targeted OCR complete: found text in %d/%d frames", len(results), len(frames))
    return results


def _merge_time_ranges(
    ranges: List[tuple], padding: float = 5.0
) -> List[tuple]:
    """Merge overlapping time ranges with padding."""
    if not ranges:
        return []

    # Add padding and sort
    padded = [(max(0, s - padding), e + padding) for s, e in ranges]
    padded.sort(key=lambda x: x[0])

    merged = [padded[0]]
    for start, end in padded[1:]:
        if start <= merged[-1][1]:
            # Overlapping — extend the current range
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


async def extract_ocr_from_video(
    video_url: str,
    interval_s: float = 30.0,
    max_frames: int = 100,
    min_confidence: float = 0.5,
    dedup_threshold: float = 0.9,
) -> Optional[str]:
    """Extract OCR text from a video file using PaddleOCR PP-OCRv4.

    Non-blocking — runs in a thread pool.
    """
    logger.info("Starting PaddleOCR extraction (interval=%.0fs, max=%d frames)", interval_s, max_frames)
    try:
        result = await asyncio.to_thread(
            _extract_ocr_sync,
            video_url,
            interval_s=interval_s,
            max_frames=max_frames,
            min_confidence=min_confidence,
            dedup_threshold=dedup_threshold,
        )
        return result
    except Exception as e:
        logger.error("OCR extraction failed: %s", e, exc_info=True)
        return None
