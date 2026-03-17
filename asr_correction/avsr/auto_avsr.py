"""Auto-AVSR provider stub.

Full lip-reading model integration. Currently a stub that returns None
unless the auto_avsr package is installed and configured.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import AVSRHint

logger = logging.getLogger(__name__)


class AutoAVSRProvider:
    """Full lip-reading AVSR provider using Auto-AVSR.

    This is a stub implementation. Install the auto_avsr package
    for full visual speech recognition capabilities.
    """

    def __init__(self):
        logger.info(
            "Auto-AVSR provider initialized "
            "(stub — install auto_avsr for full lip reading)"
        )
        self._available = False
        try:
            import auto_avsr  # noqa: F401

            self._available = True
        except ImportError:
            pass

    def analyze_segment(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[AVSRHint]:
        """Analyze a video segment using full lip-reading model.

        Args:
            video_url: URL or local path to the video.
            start_s: Segment start time in seconds.
            end_s: Segment end time in seconds.

        Returns:
            AVSRHint with lip transcript, or None if not available.
        """
        if not self._available:
            logger.debug("Auto-AVSR not installed, skipping")
            return None
        # TODO: Implement full lip-reading inference
        return None
