"""Full Auto-AVSR lip-reading provider.

Uses mpc001/auto_avsr to perform visual speech recognition on video
segments, producing text transcripts from lip movements.

Setup:
    Run scripts/setup_auto_avsr.sh to download the model and dependencies.
    Or manually:
        git clone https://github.com/mpc001/auto_avsr.git models/auto_avsr
        # Download VSR model weights to models/auto_avsr/
        pip install pytorch-lightning sentencepiece av

The model produces text transcripts from lip movements alone (no audio).
WER: ~19-20% on LRS3 benchmark (visual-only).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from . import AVSRHint

logger = logging.getLogger(__name__)

# Singleton pipeline instance (expensive to load — ~1GB model)
_pipeline = None
_pipeline_lock = threading.Lock()


def _find_auto_avsr_dir() -> Optional[str]:
    """Search for the auto_avsr installation directory."""
    candidates = [
        os.environ.get("AUTO_AVSR_DIR"),
        os.path.join(Path(__file__).parent.parent.parent, "models", "auto_avsr"),
        os.path.join(Path.home(), ".auto_avsr"),
        os.path.join(Path.home(), "auto_avsr"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            # Check if it has the expected structure
            if os.path.exists(os.path.join(path, "pipelines")) or \
               os.path.exists(os.path.join(path, "configs")):
                return path
    return None


def _get_pipeline(model_dir: Optional[str] = None, device: str = "cpu"):
    """Lazy-initialize the Auto-AVSR inference pipeline (singleton)."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        avsr_dir = model_dir or _find_auto_avsr_dir()
        if avsr_dir is None:
            logger.warning(
                "Auto-AVSR directory not found. Run scripts/setup_auto_avsr.sh "
                "or set AUTO_AVSR_DIR environment variable."
            )
            return None

        logger.info("Loading Auto-AVSR from %s (device=%s)...", avsr_dir, device)

        # Add auto_avsr to Python path so its imports work
        if avsr_dir not in sys.path:
            sys.path.insert(0, avsr_dir)

        try:
            from pipelines.pipeline import InferencePipeline

            # Find VSR config file
            config_dir = os.path.join(avsr_dir, "configs")
            vsr_config = None
            for name in ["LRS3_V_WER19.1.ini", "LRS3_V_WER20.3.ini"]:
                candidate = os.path.join(config_dir, name)
                if os.path.exists(candidate):
                    vsr_config = candidate
                    break

            if vsr_config is None:
                # Try any .ini file with "V" (visual) in the name
                for f in os.listdir(config_dir):
                    if f.endswith(".ini") and "_V_" in f:
                        vsr_config = os.path.join(config_dir, f)
                        break

            if vsr_config is None:
                logger.error("No VSR config found in %s", config_dir)
                return None

            logger.info("Using VSR config: %s", vsr_config)
            _pipeline = InferencePipeline(
                vsr_config,
                device=device,
                face_track=True,
                detector="mediapipe",
            )
            logger.info("Auto-AVSR pipeline loaded successfully")
            return _pipeline

        except ImportError as e:
            logger.error(
                "Failed to import Auto-AVSR: %s. "
                "Install dependencies: pip install pytorch-lightning sentencepiece av",
                e,
            )
            return None
        except Exception as e:
            logger.error("Failed to initialize Auto-AVSR pipeline: %s", e, exc_info=True)
            return None


class AutoAVSRProvider:
    """Full lip-reading AVSR provider using Auto-AVSR.

    Produces actual text transcripts from lip movements in video segments.
    Requires the auto_avsr package and model weights to be installed.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        self._model_dir = model_dir
        self._device = device
        self._available = False

        # Try to find and verify installation
        avsr_dir = model_dir or _find_auto_avsr_dir()
        if avsr_dir and os.path.isdir(avsr_dir):
            self._available = True
            logger.info("Auto-AVSR provider initialized (dir=%s)", avsr_dir)
        else:
            logger.info(
                "Auto-AVSR provider initialized (NOT AVAILABLE — "
                "run scripts/setup_auto_avsr.sh to install)"
            )

    def analyze_segment(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[AVSRHint]:
        """Analyze a video segment using full lip-reading model.

        Extracts a video clip for the segment, runs face detection and
        mouth ROI extraction, then performs visual speech recognition
        to produce a text transcript.

        Args:
            video_url: URL or local path to the video.
            start_s: Segment start time in seconds.
            end_s: Segment end time in seconds.

        Returns:
            AVSRHint with lip_transcript (text from lip reading),
            or None if not available or no face detected.
        """
        if not self._available:
            logger.debug("Auto-AVSR not installed, skipping")
            return None

        pipeline = _get_pipeline(self._model_dir, self._device)
        if pipeline is None:
            return None

        tmpdir = None
        try:
            from .mouth_extractor import extract_video_clip

            # Extract video clip for this segment
            tmpdir = tempfile.mkdtemp(prefix="avsr_segment_")
            clip_path = extract_video_clip(
                video_url, start_s, end_s, output_dir=tmpdir
            )

            logger.info(
                "Running Auto-AVSR lip reading on [%.1f-%.1fs]...",
                start_s, end_s,
            )

            # Process landmarks (face detection + alignment)
            landmarks = pipeline.process_landmarks(
                clip_path, landmarks_filename=None
            )

            if landmarks is None:
                logger.info("No face detected in segment [%.1f-%.1fs]", start_s, end_s)
                return AVSRHint(
                    face_detected=False,
                    speaking_confidence=0.0,
                    lip_transcript=None,
                    mode="auto_avsr",
                )

            # Load data (mouth ROI extraction + preprocessing)
            data = pipeline.dataloader.load_data(clip_path, landmarks)

            # Run visual speech recognition inference
            transcript = pipeline.model.infer(data)

            if not transcript or not transcript.strip():
                logger.info(
                    "Auto-AVSR returned empty transcript for [%.1f-%.1fs]",
                    start_s, end_s,
                )
                return AVSRHint(
                    face_detected=True,
                    speaking_confidence=0.3,
                    lip_transcript=None,
                    mode="auto_avsr",
                )

            transcript = transcript.strip()
            logger.info(
                "Auto-AVSR transcript [%.1f-%.1fs]: '%s'",
                start_s, end_s, transcript[:100],
            )

            return AVSRHint(
                face_detected=True,
                speaking_confidence=0.8,
                lip_transcript=transcript,
                mode="auto_avsr",
            )

        except Exception as e:
            logger.warning(
                "Auto-AVSR failed for segment [%.1f-%.1fs]: %s",
                start_s, end_s, e,
            )
            return None

        finally:
            # Clean up temp clip
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass
