"""Stage 4.5 — AVSR (lip-reading) hint gathering.

Runs the AVSR provider (default: MediaPipe Face Mesh) against AVSR-eligible
candidates to produce speaking-confidence hints (and optional lip
transcripts) that the reconciler can use as additional evidence.

Public function
---------------
- :func:`gather_avsr_hints` — main entry point used by the pipeline
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Categories that warrant a lip-reading look. Adding 'custom' lets users
# force AVSR on team-specific terms like internal product codenames.
AVSR_ELIGIBLE_CATEGORIES = {"person_name", "content_word", "custom"}


def gather_avsr_hints(avsr_provider, flagged, candidates, video_url, config):
    """Step 4.5: Run AVSR on AVSR-eligible candidates and collect hints.

    Quick-win wiring: lazily creates a default provider from `config.avsr_mode`
    when the caller didn't pass one. Caps work at `config.avsr_max_candidates`.
    Drops low-confidence hints below `config.avsr_min_speaking_confidence`.

    Returns:
        list[dict] — one entry per analyzed segment with shape
        {"start": float, "end": float, "hint": str, "confidence": float,
         "lip_transcript": Optional[str]}
    """
    import os
    import time as _time

    if config.dry_run or not video_url:
        return []
    if os.environ.get("ASR_AVSR_ENABLED", "true").lower() == "false":
        logger.info("Step 4.5: AVSR disabled via ASR_AVSR_ENABLED=false")
        return []

    mode = (getattr(config, "avsr_mode", "mediapipe") or "none").lower()
    if mode == "none":
        logger.info("Step 4.5: AVSR mode=none — skipping")
        return []

    # Lazy-create the provider if the caller didn't pass one.
    if avsr_provider is None:
        try:
            from ..avsr import get_avsr_provider
            avsr_provider = get_avsr_provider(mode)
        except Exception as e:
            logger.warning("Step 4.5: Could not init AVSR provider (%s) — %s", mode, e)
            return []
    if avsr_provider is None:
        logger.info("Step 4.5: No AVSR provider available — skipping")
        return []

    # Pick AVSR-eligible candidates: person_name, content_word, custom.
    # When `avsr_run_on_all_flagged` is true, ignore the category gate and
    # process every flagged segment instead.
    run_on_all = bool(getattr(config, "avsr_run_on_all_flagged", False))
    eligible_ts = []
    seen = set()
    for a in flagged:
        if not (a.start < a.end):
            continue
        cats = {getattr(c, "category", "") for c in (a.candidates or [])}
        if run_on_all or (cats & AVSR_ELIGIBLE_CATEGORIES):
            key = (round(a.start, 1), round(a.end, 1))
            if key in seen:
                continue
            seen.add(key)
            eligible_ts.append((a.start, a.end))

    if not eligible_ts:
        logger.info(
            "Step 4.5: No AVSR-eligible segments — skipping (run_on_all=%s)",
            run_on_all,
        )
        return []

    max_n = max(1, int(getattr(config, "avsr_max_candidates", 20)))
    eligible_ts = eligible_ts[:max_n]
    pad = float(getattr(config, "avsr_segment_padding_s", 0.5))
    min_conf = float(getattr(config, "avsr_min_speaking_confidence", 0.55))

    logger.info("=" * 60)
    logger.info("Step 4.5: AVSR (mode=%s) on %d segments (cap=%d)", mode, len(eligible_ts), max_n)

    hints = []
    t0 = _time.time()
    analyzed = 0
    for (s, e) in eligible_ts:
        try:
            hint = avsr_provider.analyze_segment(video_url, max(0.0, s - pad), e + pad)
        except Exception as exc:
            logger.warning("  AVSR failed on [%.2f-%.2f]: %s", s, e, exc)
            continue
        analyzed += 1
        if hint is None or not getattr(hint, "face_detected", False):
            continue

        conf = float(getattr(hint, "speaking_confidence", 0.0) or 0.0)
        if conf < min_conf and not getattr(hint, "lip_transcript", None):
            continue  # gated out — useless to the reconciler

        hints.append({
            "start": s,
            "end": e,
            "hint": hint.to_prompt_hint(),
            "confidence": conf,
            "lip_transcript": getattr(hint, "lip_transcript", None),
        })

    elapsed_ms = int((_time.time() - t0) * 1000)
    logger.info("  Produced %d usable hints (analyzed=%d) in %dms", len(hints), analyzed, elapsed_ms)
    for h in hints[:5]:
        logger.info("    [%.1f-%.1fs] %s", h["start"], h["end"], h["hint"])
    if len(hints) > 5:
        logger.info("    ... and %d more", len(hints) - 5)
    logger.info(
        "AVSR summary status=ok mode=%s analyzed=%d kept=%d dropped=%d latency_ms=%d",
        mode, analyzed, len(hints), max(0, analyzed - len(hints)), elapsed_ms,
    )
    logger.info("=" * 60)
    return hints
