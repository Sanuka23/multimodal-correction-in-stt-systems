"""OCR extraction from video files.

Stub implementation — returns None for now.
Future: will use Vertex AI / Gemini to extract visible text from video frames.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def extract_ocr_from_video(video_url: str) -> Optional[str]:
    """Extract OCR text from a video file.

    Stub that returns None. Future implementation will:
    1. Download or stream the video from the URL
    2. Use a multimodal LLM (e.g., Gemini) to extract visible text
    3. Return OCR XML in the standard <ocr-extraction> format:
       <ocr-extraction>
         <frame timestamp="MM:SS" type="slide">
           <text>Extracted text here</text>
         </frame>
       </ocr-extraction>

    Args:
        video_url: Direct URL to the video file.

    Returns:
        OCR XML string, or None if not yet implemented.
    """
    logger.info("OCR extraction stub called for %s — returning None", video_url)
    return None
