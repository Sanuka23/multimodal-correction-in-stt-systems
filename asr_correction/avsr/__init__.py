"""AVSR (Audio-Visual Speech Recognition) module.

Provides lip-reading hints to improve ASR correction accuracy.
Two modes:
  - mediapipe: Lightweight face/mouth detection hints (default)
  - auto_avsr: Full lip-reading model for visual transcription
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AVSRHint:
    """Result from AVSR analysis of a video segment."""

    face_detected: bool
    speaking_confidence: float  # 0.0-1.0
    lip_transcript: Optional[str] = None  # Only from full Auto-AVSR mode
    mode: str = "mediapipe"  # "mediapipe" or "auto_avsr"
    active_speaker_id: Optional[float] = None  # Face position ID (nose x-coord)
    num_faces: int = 1  # Number of faces detected in frame

    def to_prompt_hint(self) -> str:
        """Format as a string for the LLM prompt."""
        if not self.face_detected:
            return "No face detected in video segment"

        speaker_info = ""
        if self.num_faces > 1:
            speaker_info = f" [active speaker from {self.num_faces} faces]"

        if self.lip_transcript:
            return (
                f"Visual speech suggests: '{self.lip_transcript}' "
                f"(confidence: {self.speaking_confidence:.2f}){speaker_info}"
            )
        if self.speaking_confidence > 0.5:
            return (
                f"Speaker detected and actively speaking "
                f"(confidence: {self.speaking_confidence:.2f}){speaker_info}"
            )
        return (
            f"Face detected but low speaking activity "
            f"(confidence: {self.speaking_confidence:.2f}){speaker_info}"
        )


class AVSRProvider(Protocol):
    """Protocol for AVSR providers."""

    def analyze_segment(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[AVSRHint]: ...


def get_avsr_provider(mode: str = "mediapipe", model_dir: str = None) -> Optional[AVSRProvider]:
    """Factory function to create the appropriate AVSR provider."""
    if mode == "mediapipe":
        from .mediapipe_hints import MediaPipeHintProvider

        return MediaPipeHintProvider()
    elif mode == "auto_avsr":
        from .auto_avsr import AutoAVSRProvider

        return AutoAVSRProvider(model_dir=model_dir)
    elif mode == "none" or mode is None:
        return None
    else:
        logger.warning("Unknown AVSR mode: %s, using none", mode)
        return None
