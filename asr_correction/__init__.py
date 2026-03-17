"""ScreenApp ASR Correction Module.

Post-processing step that corrects context-critical terms in
ScreenApp transcripts using a fine-tuned Qwen2.5-7B model,
custom vocabulary, and on-demand OCR hints.

Usage:
    from asr_correction import correct_transcript

    enhanced, report = correct_transcript(
        transcript=screenapp_output_transcript,
        file_id="abc-123",
        custom_vocabulary=[{"term": "ScreenApp", "category": "product_name"}],
        ocr_provider=screenapp_ocr_callback,
    )
"""

import logging

from .config import CorrectionConfig
from .corrector import correct_candidates, identify_candidates
from .data_collector import collect_correction_data
from .types import CorrectionReport
from .vocabulary import load_domain_vocab, merge_vocabularies

__version__ = "1.0.0"

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

    This is the main entry point. Call after customTranscribe() completes,
    before uploading to S3.

    Args:
        transcript: ScreenApp OutputTranscript dict with text, words,
            segments, and meta fields.
        file_id: ScreenApp file ID (for OCR requests and data collection).
        custom_vocabulary: Team/file custom vocabulary list. Can be a list
            of dicts ({"term": ..., "category": ...}) or plain strings.
        ocr_provider: Either a callable(file_id, start_s, end_s) -> OCR XML,
            or an object with get_ocr() method. Can be None.
        config: Full CorrectionConfig. If None, uses defaults.
        **config_overrides: Override individual config fields.

    Returns:
        (enhanced_transcript, correction_report) tuple.
        - enhanced_transcript: dict in OutputTranscript format with
          corrected text and asr_correction metadata in meta.
        - correction_report: CorrectionReport with details of all
          corrections attempted and applied.
    """
    if config is None:
        config = CorrectionConfig(**config_overrides)
    else:
        for key, val in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)

    # Step 1: Merge vocabularies
    domain_vocab = load_domain_vocab(config.domain_vocab_path)
    vocab_terms = merge_vocabularies(custom_vocabulary, domain_vocab)
    logger.info("Step 1: Merged %d domain + %d custom → %d vocab terms",
                len(domain_vocab), len(custom_vocabulary or []), len(vocab_terms))

    # Step 2: Identify correction candidates
    candidates = identify_candidates(transcript, vocab_terms, config)
    logger.info("Step 2: Found %d correction candidates in transcript", len(candidates))

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

    for i, c in enumerate(candidates):
        logger.info("  Candidate %d: '%s' (error='%s', category='%s', pos=%d, ts=%.1f-%.1f)",
                     i + 1, c.term, c.error_found, c.category, c.char_position,
                     c.timestamp_start, c.timestamp_end)

    # Step 3: Pass 1 — Correct candidates (OCR + vocab only)
    logger.info("Step 3: Pass 1 — correction on %d candidates (dry_run=%s)...", len(candidates), config.dry_run)
    enhanced, report = correct_candidates(
        candidates, transcript, file_id, ocr_provider, config
    )
    logger.info("Step 3 complete: %d/%d corrections applied in %.0fms",
                report.corrections_applied, report.corrections_attempted, report.processing_time_ms)

    # Step 4: Pass 2 — AVSR-targeted re-correction for uncertain segments
    if avsr_provider and video_url and not config.dry_run:
        from .corrector import _fetch_ocr_hints
        from .model import build_prompt, load_model, run_inference

        # Find segments where model flagged need_lip or was uncertain
        lip_needed = [
            r for r in report.results
            if r.need_lip or (not r.applied and r.confidence < 0.8)
        ]

        if lip_needed:
            # Limit to max_segments to avoid excessive processing
            lip_needed = lip_needed[:config.avsr_max_segments]
            logger.info("Step 4: Pass 2 — AVSR on %d uncertain segments (mode=%s)",
                        len(lip_needed), config.avsr_mode)

            avsr_applied = 0
            model, tokenizer = load_model(
                adapter_path=config.adapter_path,
                model_path=config.model_path,
                base_model=config.base_model,
                backend=config.backend,
            )

            for i, result in enumerate(lip_needed):
                candidate = result.candidate
                try:
                    hint = avsr_provider.analyze_segment(
                        video_url, candidate.timestamp_start, candidate.timestamp_end
                    )
                    if hint is None:
                        logger.info("  AVSR %d/%d: no hint for '%s'", i + 1, len(lip_needed), candidate.term)
                        continue

                    lip_hint_str = hint.to_prompt_hint()
                    logger.info("  AVSR %d/%d: '%s' → hint='%s'",
                                i + 1, len(lip_needed), candidate.term, lip_hint_str)

                    # Fetch OCR hints again for context
                    ocr_hints = _fetch_ocr_hints(candidate, file_id, ocr_provider, config)

                    # Re-run inference with lip hint
                    prompt = build_prompt(
                        candidate.context,
                        [candidate.term],
                        candidate.category,
                        ocr_hints,
                        lip_hint=lip_hint_str,
                    )
                    new_result = run_inference(
                        prompt, config.system_prompt, model, tokenizer, config.max_tokens
                    )

                    # If now confident with changes, apply
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
                            # Also apply to segments
                            for seg in enhanced.get("segments", []):
                                seg_new = pattern.sub(candidate.term, seg.get("text", ""), count=1)
                                if seg_new != seg.get("text", ""):
                                    seg["text"] = seg_new
                                    break
                            avsr_applied += 1
                            logger.info("  AVSR Pass 2: APPLIED '%s' → '%s' (confidence=%.2f)",
                                        candidate.error_found, candidate.term, new_confidence)
                except Exception as e:
                    logger.warning("  AVSR failed for '%s': %s", candidate.term, e)

            logger.info("Step 4 complete: AVSR applied %d additional corrections", avsr_applied)
            report.corrections_applied += avsr_applied
        else:
            logger.info("Step 4: No segments flagged for AVSR — skipping Pass 2")
    else:
        if not avsr_provider:
            logger.info("Step 4: AVSR provider not configured — skipping Pass 2")

    # Step 5: Collect data for future training
    if config.collect_data and report.results:
        logger.info("Step 5: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )

    return enhanced, report
