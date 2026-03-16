"""OCR integration for ASR correction.

Defines the OCR provider interface and parses ScreenApp's
OCR XML format into usable text hints.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class OCRProvider(Protocol):
    """Interface that ScreenApp implements to provide OCR data.

    The correction module calls get_ocr() when it needs screen content
    for a specific time window around an error candidate.
    """

    def get_ocr(
        self, file_id: str, start_time: float, end_time: float
    ) -> Optional[str]:
        """Request OCR text for a time window.

        Args:
            file_id: ScreenApp file identifier.
            start_time: Window start in seconds.
            end_time: Window end in seconds.

        Returns:
            Raw OCR XML string (ScreenApp's <ocr-extraction> format),
            or None if OCR is not available.
        """
        ...


# Simple callable alternative
OCRCallback = Callable[[str, float, float], Optional[str]]


def parse_ocr_xml(ocr_xml: str) -> List[Dict]:
    """Parse ScreenApp's OCR XML format into structured frames.

    Input format (from systemPromptResponses.OCR.responseText):
        <ocr-extraction>
          <frame timestamp="08:55" type="slide">
            <text>Vanta\\nGet compliant fast...</text>
            <visuals>Description of what's visible</visuals>
          </frame>
        </ocr-extraction>

    Returns list of:
        {"timestamp_s": float, "timestamp": str, "type": str, "text": str}
    """
    frames = []
    frame_pattern = re.compile(
        r'<frame\s+timestamp="([^"]+)"\s+type="([^"]+)">\s*(.*?)\s*</frame>',
        re.DOTALL,
    )
    for match in frame_pattern.finditer(ocr_xml):
        ts_str = match.group(1)
        frame_type = match.group(2)
        content = match.group(3)

        text_match = re.search(r'<text>\s*(.*?)\s*</text>', content, re.DOTALL)
        text = text_match.group(1).strip() if text_match else ""

        # Handle frames without <text> tags
        if not text_match:
            vis_match = re.search(
                r'<visuals>\s*(.*?)\s*</visuals>', content, re.DOTALL
            )
            if not vis_match:
                text = content.strip()

        if text:
            frames.append(
                {
                    "timestamp_s": _parse_timestamp(ts_str),
                    "timestamp": ts_str,
                    "type": frame_type,
                    "text": text,
                }
            )

    return frames


def _parse_timestamp(ts_str: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


# UI noise patterns to filter from OCR output
_NOISE_PATTERNS = [
    r"(?:Un)?[Mm]ute my microphone.*",
    r"Connecting to audio.*",
    r"Phone Call",
    r"Computer Audio",
    r"Join Audio by Computer",
    r"Turn on microphone.*",
    r"Let everyone send messages.*",
    r"When you leave the call.*",
    r"Join later.*",
    r"You can pin a message.*",
]


def extract_hints_from_frames(
    frames: List[Dict],
    center_time: float,
    window: float = 15.0,
    max_hints: int = 15,
) -> List[str]:
    """Extract clean OCR text hints from frames near a timestamp.

    Filters UI noise, deduplicates, and caps at max_hints.
    """
    nearby = [f for f in frames if abs(f["timestamp_s"] - center_time) <= window]
    seen: set = set()
    hints: List[str] = []

    for frame in nearby:
        for line in frame["text"].split("\n"):
            line = line.strip()
            if not line or len(line) <= 1 or line in seen:
                continue
            if any(re.match(pat, line, re.IGNORECASE) for pat in _NOISE_PATTERNS):
                continue
            seen.add(line)
            hints.append(line)
            if len(hints) >= max_hints:
                return hints

    return hints
