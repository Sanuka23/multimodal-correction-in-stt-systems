"""Video frame extraction via a single FFmpeg call.

Uses the fps filter to extract ALL frames in one pass, outputting JPEGs
to a temp directory. This is ~50x faster than individual HTTP seeks
because it avoids per-frame connection overhead.

Includes pixel-based deduplication to skip visually identical frames.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 600  # 10 min max for full extraction
_FFPROBE_TIMEOUT = 30


@dataclass
class ExtractedFrame:
    """A single frame extracted from a video."""

    image: np.ndarray  # BGR numpy array (OpenCV format)
    timestamp_s: float


def get_video_duration(video_url: str) -> Optional[float]:
    """Get video duration using ffprobe (reads only the header)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_url,
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError) as e:
        logger.warning("ffprobe failed: %s", e)
    return None


def _compute_image_hash(image: np.ndarray, hash_size: int = 16) -> str:
    """Compute a perceptual hash for deduplication (~1ms per frame)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized > mean_val).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


def extract_frames_periodic(
    video_url: str,
    interval_s: float = 30.0,
    max_frames: int = 100,
) -> List[ExtractedFrame]:
    """Extract visually unique frames using a SINGLE FFmpeg call.

    Uses the fps filter to output all frames at once to a temp directory.
    For a 25-min video at 30s intervals: ~50 frames in one ~30s FFmpeg call
    instead of 50 individual HTTP requests.
    """
    t0 = time.time()

    duration = get_video_duration(video_url)
    if duration is None or duration <= 0:
        logger.error("Could not determine video duration")
        return []

    fps_rate = 1.0 / interval_s
    expected_frames = min(int(duration / interval_s) + 1, max_frames)

    logger.info(
        "Video: %.0fs (%.1f min), extracting 1 frame every %.0fs (~%d frames) via single FFmpeg pass",
        duration, duration / 60, interval_s, expected_frames,
    )

    with tempfile.TemporaryDirectory(prefix="ocr_frames_") as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_url,
            "-vf", f"fps={fps_rate}",
            "-frames:v", str(max_frames),
            "-q:v", "3",
            "-loglevel", "warning",
            output_pattern,
        ]

        logger.info("Running FFmpeg bulk extraction...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error("FFmpeg failed (code %d): %s",
                             result.returncode, result.stderr[:500])
                return []
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out after %ds", _FFMPEG_TIMEOUT)
            return []

        extract_time = time.time() - t0
        logger.info("FFmpeg extraction done in %.1fs", extract_time)

        # Read extracted JPEG files
        frame_files = sorted(
            f for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )

        if not frame_files:
            logger.warning("FFmpeg produced no frames")
            return []

        logger.info("Reading %d extracted frames and deduplicating...", len(frame_files))

        frames: List[ExtractedFrame] = []
        prev_hash: Optional[str] = None
        skipped = 0

        for idx, fname in enumerate(frame_files):
            timestamp_s = idx * interval_s

            filepath = os.path.join(tmpdir, fname)
            image = cv2.imread(filepath)
            if image is None:
                continue

            # Pixel dedup
            img_hash = _compute_image_hash(image)
            if prev_hash is not None and img_hash == prev_hash:
                skipped += 1
                continue

            prev_hash = img_hash
            frames.append(ExtractedFrame(image=image, timestamp_s=timestamp_s))

    total_time = time.time() - t0
    logger.info(
        "Frame extraction complete: %d unique frames (skipped %d dupes from %d total) in %.1fs",
        len(frames), skipped, len(frame_files), total_time,
    )

    return frames


def extract_frames_at_timestamps(
    video_url: str,
    timestamps: List[float],
) -> List[ExtractedFrame]:
    """Extract frames at specific timestamps using individual FFmpeg seeks.

    Used for targeted OCR — extracts only frames at specified times
    instead of the entire video. For small numbers of frames (< 50),
    this is faster than extracting every frame from the full video.
    """
    if not timestamps:
        return []

    t0 = time.time()
    frames = []
    prev_hash = None

    logger.info("Extracting %d frames at specific timestamps...", len(timestamps))

    with tempfile.TemporaryDirectory(prefix="targeted_frames_") as tmpdir:
        for idx, ts in enumerate(sorted(timestamps)):
            output_path = os.path.join(tmpdir, f"frame_{idx:04d}.jpg")

            cmd = [
                "ffmpeg",
                "-ss", str(ts),
                "-i", video_url,
                "-frames:v", "1",
                "-q:v", "3",
                "-loglevel", "quiet",
                "-y",
                output_path,
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg timeout for frame at %.1fs", ts)
                continue

            if not os.path.exists(output_path):
                continue

            image = cv2.imread(output_path)
            if image is None:
                continue

            # Dedup
            img_hash = _compute_image_hash(image)
            if prev_hash is not None and img_hash == prev_hash:
                continue
            prev_hash = img_hash

            frames.append(ExtractedFrame(image=image, timestamp_s=ts))

    total_time = time.time() - t0
    logger.info("Targeted frame extraction: %d frames in %.1fs", len(frames), total_time)
    return frames
