"""Shared utility for extracting frames from a specific video segment.

Uses FFmpeg to extract a small number of frames from a narrow time range,
suitable for AVSR analysis of individual transcript segments.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 60  # 1 min max for a short segment extraction


def extract_segment_frames(
    video_url: str,
    start_s: float,
    end_s: float,
    num_frames: int = 5,
) -> List[np.ndarray]:
    """Extract a few frames from a specific time range using FFmpeg.

    Args:
        video_url: URL or local path to the video file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        num_frames: Number of frames to extract (evenly spaced).

    Returns:
        List of BGR numpy arrays (OpenCV format). May be shorter than
        num_frames if the segment is very short or extraction fails.
    """
    duration = end_s - start_s
    if duration <= 0:
        logger.warning("Invalid segment duration: start=%.2f end=%.2f", start_s, end_s)
        return []

    # Calculate fps to get approximately num_frames from the segment
    # Ensure at least 1 fps to avoid ffmpeg errors
    fps = max(num_frames / duration, 1.0) if duration > 0 else 1.0

    with tempfile.TemporaryDirectory(prefix="avsr_frames_") as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-ss", str(start_s),
            "-t", str(duration),
            "-i", video_url,
            "-vf", f"fps={fps:.4f}",
            "-frames:v", str(num_frames),
            "-q:v", "3",
            "-loglevel", "warning",
            output_pattern,
        ]

        logger.debug(
            "Extracting %d frames from segment [%.2f-%.2f]s",
            num_frames, start_s, end_s,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT,
            )
            if result.returncode != 0:
                logger.warning(
                    "FFmpeg segment extraction failed (code %d): %s",
                    result.returncode, result.stderr[:300],
                )
                return []
        except subprocess.TimeoutExpired:
            logger.warning(
                "FFmpeg segment extraction timed out for [%.2f-%.2f]s",
                start_s, end_s,
            )
            return []

        # Read extracted frames
        frame_files = sorted(
            f for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )

        frames: List[np.ndarray] = []
        for fname in frame_files:
            filepath = os.path.join(tmpdir, fname)
            image = cv2.imread(filepath)
            if image is not None:
                frames.append(image)

    logger.debug("Extracted %d frames from segment [%.2f-%.2f]s", len(frames), start_s, end_s)
    return frames


def extract_video_clip(
    video_url: str,
    start_s: float,
    end_s: float,
    output_dir: str = None,
) -> str:
    """Extract a video clip as an MP4 file for Auto-AVSR processing.

    Auto-AVSR expects 25fps video with yuv420p pixel format.

    Args:
        video_url: URL or local path to the video file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        output_dir: Directory for the temp clip. If None, uses system temp.

    Returns:
        Path to the extracted .mp4 clip file.

    Raises:
        RuntimeError: If FFmpeg extraction fails.
    """
    duration = end_s - start_s
    if duration <= 0:
        raise ValueError(f"Invalid segment: start={start_s}, end={end_s}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="avsr_clip_")

    clip_path = os.path.join(output_dir, "segment.mp4")

    cmd = [
        "ffmpeg",
        "-ss", str(start_s),
        "-i", video_url,
        "-t", str(duration),
        "-r", "25",                # Auto-AVSR expects 25fps
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-loglevel", "warning",
        "-y",
        clip_path,
    ]

    logger.info("Extracting video clip [%.1f-%.1fs] (%.1fs duration)",
                start_s, end_s, duration)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_FFMPEG_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg clip extraction failed (code {result.returncode}): "
                f"{result.stderr[:300]}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"FFmpeg clip extraction timed out for [{start_s}-{end_s}]s"
        )

    if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
        raise RuntimeError(f"FFmpeg produced empty clip for [{start_s}-{end_s}]s")

    logger.info("Clip extracted: %s (%.1f KB)",
                clip_path, os.path.getsize(clip_path) / 1024)
    return clip_path
