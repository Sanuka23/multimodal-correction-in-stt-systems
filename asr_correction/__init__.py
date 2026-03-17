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

    # Step 3: Correct candidates
    logger.info("Step 3: Running correction on %d candidates (dry_run=%s)...", len(candidates), config.dry_run)
    enhanced, report = correct_candidates(
        candidates, transcript, file_id, ocr_provider, config
    )
    logger.info("Step 3 complete: %d/%d corrections applied in %.0fms",
                report.corrections_applied, report.corrections_attempted, report.processing_time_ms)

    # Step 4: Collect data for future training
    if config.collect_data and report.results:
        logger.info("Step 4: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )

    return enhanced, report
