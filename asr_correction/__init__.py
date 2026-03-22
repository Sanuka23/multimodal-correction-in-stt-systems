"""ScreenApp ASR Correction Module.

Two-stage selective correction pipeline:
1. Segment Selector — analyzes transcript + vocab to identify which segments
   need fixing and which modality (OCR/AVSR) each needs.
2. Targeted Correction — only fetches OCR/runs AVSR on flagged segments,
   then corrects with the fine-tuned model.

Usage:
    from asr_correction import correct_transcript

    enhanced, report = correct_transcript(
        transcript=screenapp_output_transcript,
        file_id="abc-123",
        custom_vocabulary=[{"term": "ScreenApp", "category": "product_name"}],
        video_url="http://...",
    )
"""

import logging
import time

from .batch_corrector import correct_transcript_batch
from .config import CorrectionConfig
from .corrector import correct_candidates
from .data_collector import collect_correction_data
from .segment_selector import select_segments
from .types import CorrectionReport
from .vocabulary import load_domain_vocab, merge_vocabularies

__version__ = "3.0.0"

logger = logging.getLogger(__name__)


def correct_transcript(
    transcript: dict,
    file_id: str,
    custom_vocabulary: list = None,
    ocr_provider=None,
    avsr_provider=None,
    video_url: str = None,
    config: CorrectionConfig = None,
    **config_overrides,
) -> tuple:
    """Correct ASR errors in a ScreenApp transcript.

    New two-stage pipeline:
    1. Segment selection (rules + model) → identifies ~5-20% of segments
    2. Targeted OCR/AVSR → only processes flagged time ranges
    3. Correction → runs model on flagged candidates only

    Args:
        transcript: ScreenApp OutputTranscript dict with text, words,
            segments, and meta fields.
        file_id: ScreenApp file ID (for OCR requests and data collection).
        custom_vocabulary: Team/file custom vocabulary list.
        ocr_provider: Pre-existing OCR provider (cached OCR XML).
            If provided, used instead of targeted extraction.
        avsr_provider: AVSR provider for lip-reading analysis.
        video_url: Video URL for targeted OCR/AVSR extraction.
        config: Full CorrectionConfig. If None, uses defaults.
        **config_overrides: Override individual config fields.

    Returns:
        (enhanced_transcript, correction_report) tuple.
    """
    if config is None:
        config = CorrectionConfig(**config_overrides)
    else:
        for key, val in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)

    pipeline_start = time.time()

    # Step 1: Merge vocabularies
    domain_vocab = load_domain_vocab(config.domain_vocab_path)
    vocab_terms = merge_vocabularies(custom_vocabulary, domain_vocab)
    logger.info("Step 1: Merged %d domain + %d custom → %d vocab terms",
                len(domain_vocab), len(custom_vocabulary or []), len(vocab_terms))

    # Step 2: Error detection — LLM Pass 1 (primary) → rule-based (fallback)
    from .model import load_model
    from .llm_detector import detect_errors

    model, tokenizer = None, None
    if not config.dry_run:
        try:
            model, tokenizer = load_model(
                adapter_path=config.adapter_path,
                model_path=config.model_path,
                base_model=config.base_model,
                backend=config.backend,
            )
        except Exception as e:
            logger.error("Model loading failed: %s", e)

    analyses = detect_errors(
        transcript, vocab_terms, model, tokenizer, config=config,
    )

    flagged = [a for a in analyses if a.needs_correction]
    candidates = [c for a in flagged for c in a.candidates]

    if not candidates:
        logger.info("No candidates found — returning original transcript unchanged")
        report = CorrectionReport(
            file_id=file_id,
            corrections_attempted=0,
            corrections_applied=0,
            results=[],
            processing_time_ms=0,
        )
        return transcript, report

    logger.info("Step 2: Found %d candidates in %d/%d segments",
                len(candidates), len(flagged), len(analyses))
    for i, c in enumerate(candidates):
        logger.info("  Candidate %d: '%s' (error='%s', category='%s', ts=%.1f-%.1f)",
                     i + 1, c.term, c.error_found, c.category,
                     c.timestamp_start, c.timestamp_end)

    # Step 3: Targeted OCR extraction (only for flagged segments)
    ocr_needs = [a for a in flagged if a.needs_ocr]
    targeted_ocr_provider = ocr_provider  # Use existing provider if available

    if not ocr_provider and ocr_needs and video_url:
        # No cached OCR — do targeted extraction for flagged segments only
        from .ocr_extractor import extract_ocr_for_segments
        from .ocr_parser import create_ocr_provider_from_hints

        time_ranges = [(a.start, a.end) for a in ocr_needs]
        logger.info("Step 3: Targeted OCR for %d segments (vs full video)", len(ocr_needs))

        ocr_hints_by_ts = extract_ocr_for_segments(
            video_url, time_ranges,
            padding_s=config.ocr_window_seconds,
            min_confidence=config.ocr_min_confidence,
        )

        if ocr_hints_by_ts:
            targeted_ocr_provider = _create_targeted_ocr_provider(ocr_hints_by_ts)
            logger.info("Step 3: Targeted OCR found text at %d timestamps", len(ocr_hints_by_ts))
        else:
            logger.info("Step 3: Targeted OCR found no text")
    elif ocr_provider:
        logger.info("Step 3: Using pre-cached OCR provider")
    else:
        logger.info("Step 3: No video URL or OCR provider — skipping OCR")

    # Step 3.5: Pre-collect AVSR hints for flagged segments (before batch correction)
    lip_hints = {}  # (ts_start, ts_end) → hint_string
    avsr_needs = [a for a in flagged if a.needs_avsr]
    if avsr_provider and video_url and avsr_needs and not config.dry_run:
        avsr_pre = avsr_needs[:config.avsr_max_segments]
        logger.info("Step 3.5: Pre-collecting AVSR hints for %d segments", len(avsr_pre))
        for analysis in avsr_pre:
            try:
                hint = avsr_provider.analyze_segment(
                    video_url, analysis.start, analysis.end
                )
                if hint and hint.face_detected:
                    lip_hints[(analysis.start, analysis.end)] = hint.to_prompt_hint()
            except Exception as e:
                logger.warning("AVSR pre-collection failed for [%.1f-%.1f]: %s",
                             analysis.start, analysis.end, e)
        if lip_hints:
            logger.info("Step 3.5: Collected %d AVSR hints", len(lip_hints))
    else:
        logger.info("Step 3.5: AVSR pre-collection skipped")

    # Step 4: Batch correction — send whole transcript chunks to model
    logger.info("Step 4: Batch correction (whole transcript, not word-by-word)...")
    enhanced, report = correct_transcript_batch(
        transcript, file_id, vocab_terms,
        ocr_provider=targeted_ocr_provider,
        config=config,
        lip_hints=lip_hints,
    )
    logger.info("Step 4 complete: %d corrections applied in %.0fms",
                report.corrections_applied, report.processing_time_ms)

    # Step 5: AVSR pass on flagged segments
    avsr_needs = [a for a in flagged if a.needs_avsr]
    if avsr_provider and video_url and avsr_needs and not config.dry_run:
        from .corrector import _fetch_ocr_hints
        from .model import build_prompt, run_inference

        avsr_needs = avsr_needs[:config.avsr_max_segments]
        logger.info("Step 5: AVSR on %d segments (mode=%s)", len(avsr_needs), config.avsr_mode)

        avsr_applied = 0
        for i, analysis in enumerate(avsr_needs):
            for candidate in analysis.candidates:
                try:
                    hint = avsr_provider.analyze_segment(
                        video_url, candidate.timestamp_start, candidate.timestamp_end
                    )
                    if hint is None:
                        continue

                    lip_hint_str = hint.to_prompt_hint()
                    ocr_hints = _fetch_ocr_hints(candidate, file_id, targeted_ocr_provider, config)

                    prompt = build_prompt(
                        candidate.context, [candidate.term], candidate.category,
                        ocr_hints, lip_hint=lip_hint_str,
                    )
                    new_result = run_inference(
                        prompt, config.system_prompt, model, tokenizer, config.max_tokens
                    )

                    new_confidence = new_result.get("confidence", 0.0)
                    new_changes = new_result.get("changes", [])
                    if new_confidence >= config.confidence_threshold and new_changes:
                        import re
                        pattern = re.compile(
                            r'\b' + re.escape(candidate.error_found) + r'\b',
                            re.IGNORECASE,
                        )
                        text = enhanced.get("text", "")
                        new_text = pattern.sub(candidate.term, text, count=1)
                        if new_text != text:
                            enhanced["text"] = new_text
                            for seg in enhanced.get("segments", []):
                                seg_new = pattern.sub(candidate.term, seg.get("text", ""), count=1)
                                if seg_new != seg.get("text", ""):
                                    seg["text"] = seg_new
                                    break
                            avsr_applied += 1
                except Exception as e:
                    logger.warning("AVSR failed for '%s': %s", candidate.term, e)

        logger.info("Step 5 complete: AVSR applied %d corrections", avsr_applied)
        report.corrections_applied += avsr_applied
    else:
        logger.info("Step 5: AVSR skipped (provider=%s, video=%s, segments=%d)",
                    bool(avsr_provider), bool(video_url), len(avsr_needs) if avsr_needs else 0)

    # Step 6: Collect data for future training
    if config.collect_data and report.results:
        logger.info("Step 6: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )

    # Add selector metadata to report
    report.selector_info = {
        "total_segments": len(analyses),
        "flagged_segments": len(flagged),
        "needs_ocr": len(ocr_needs),
        "needs_avsr": len(avsr_needs) if avsr_needs else 0,
        "candidates_found": len(candidates),
    }

    total_ms = (time.time() - pipeline_start) * 1000
    report.processing_time_ms = total_ms
    logger.info("Pipeline complete in %.0fms (selector → targeted OCR → correction)", total_ms)

    return enhanced, report


def _create_targeted_ocr_provider(ocr_hints_by_ts: dict):
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
