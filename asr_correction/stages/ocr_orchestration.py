"""Stage 3 — OCR orchestration helpers.

Smart timestamp computation for OCR (mixes targeted frames at flagged-segment
midpoints with evenly-spaced fallback frames), and a callable OCR-provider
factory for already-extracted hints.

Public functions
----------------
- :func:`compute_ocr_timestamps` — pick frames to OCR
- :func:`create_targeted_ocr_provider` — wrap pre-extracted hints as a provider
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_ocr_timestamps(analyses, duration, config):
    """Compute targeted OCR timestamps from flagged segments.

    Strategy:
    - Extract frame at midpoint of each segment with needs_ocr=True
    - Add evenly-spaced frames to fill remaining budget
    - Cap at config.quick_ocr_num_frames total
    """
    max_frames = config.quick_ocr_num_frames

    # Targeted frames: midpoint of each needs_ocr segment
    targeted_ts = []
    for a in analyses:
        if getattr(a, 'needs_ocr', False) and a.start < a.end:
            midpoint = (a.start + a.end) / 2.0
            targeted_ts.append(midpoint)

    # Deduplicate timestamps that are very close (within 5s)
    targeted_ts.sort()
    deduped = []
    for ts in targeted_ts:
        if not deduped or ts - deduped[-1] > 5.0:
            deduped.append(ts)
    targeted_ts = deduped

    # Reserve up to 2/3 of budget for targeted, rest for evenly-spaced
    max_targeted = min(len(targeted_ts), int(max_frames * 0.67))
    targeted_ts = targeted_ts[:max_targeted]

    # Fill remaining slots with evenly-spaced frames
    remaining = max_frames - len(targeted_ts)
    if remaining > 0 and duration and duration > 0:
        interval = duration / (remaining + 1)
        for i in range(remaining):
            ts = interval * (i + 1)
            # Skip if too close to a targeted frame
            if not any(abs(ts - t) < 5.0 for t in targeted_ts):
                targeted_ts.append(ts)

    targeted_ts.sort()
    return targeted_ts[:max_frames]


def create_targeted_ocr_provider(ocr_hints_by_ts: dict):
    """Create a simple OCR provider from targeted extraction results.

    Returns a callable that matches the OCR provider interface:
    provider(file_id, start_s, end_s) -> OCR XML string
    """
    def provider(file_id, start_s, end_s):
        # Find all OCR results within the time window
        lines = ["<ocr-extraction>"]
        for ts, texts in sorted(ocr_hints_by_ts.items()):
            if start_s <= ts <= end_s:
                text_content = "\n".join(
                    t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    for t in texts
                )
                minutes = int(ts) // 60
                secs = int(ts) % 60
                lines.append(f'  <frame timestamp="{minutes:02d}:{secs:02d}" type="slide">')
                lines.append(f"    <text>{text_content}</text>")
                lines.append("  </frame>")
        lines.append("</ocr-extraction>")
        return "\n".join(lines)

    return provider
