"""POST /asr-correct — ScreenApp-compatible correction endpoint."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import get_jwt_info
from ..database import create_job, complete_job, fail_job, save_correction, get_cached_ocr, cache_ocr_result, update_job_step
from ..services.pipeline_settings import apply_to_config as apply_pipeline_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Correction"])


def _extract_text(transcript) -> str:
    """Extract plain text from a transcript dict."""
    if isinstance(transcript, dict):
        segments = transcript.get("segments", [])
        return " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())
    return str(transcript)


class ASRCorrectionRequest(BaseModel):
    transcript: Dict[str, Any]
    file_id: str
    custom_vocabulary: Union[str, List[str]] = ""
    ocr_xml: Optional[str] = None
    video_url: Optional[str] = None


def _run_in_thread(fn):
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn)


def _make_step_bridge(loop, job_id: str):
    """Return a sync callable that the pipeline (running in a worker thread)
    can call to record pipeline-step events into MongoDB.

    Each call BLOCKS until the MongoDB write completes. This is required to
    preserve ordering — without it, two events for the same step name
    (e.g. running → completed) can race and push duplicate rows because the
    `completed` event's name lookup runs before the `running` event has
    persisted.

    The cost is ~5–10 ms per step write, negligible compared to the rest
    of the pipeline.
    """
    def _bridge(name: str, status: str, duration_ms=None, details=None):
        try:
            future = asyncio.run_coroutine_threadsafe(
                update_job_step(job_id, name, status, duration_ms, details),
                loop,
            )
            # Block — but never longer than 5 s, and never let a slow Mongo
            # write take down the pipeline.
            future.result(timeout=5)
        except Exception as exc:
            logger.debug("step bridge failed for %s/%s: %s", name, status, exc)

    return _bridge


@router.post("/asr-analyze")
async def asr_analyze(
    request: ASRCorrectionRequest,
    user_id: str = Depends(get_jwt_info),
):
    """Analyze a transcript: detect errors + classify meeting topic/field.

    Lightweight — no correction, no OCR, no AVSR. Returns errors and topic info.
    """
    from asr_correction import analyze_transcript, CorrectionConfig

    start_time = time.time()

    transcript_text = _extract_text(request.transcript)
    logger.info(
        "=== ASR Analyze Request ===\n"
        "  file_id:        %s\n"
        "  user_id:        %s\n"
        "  transcript_len: %d chars",
        request.file_id, user_id, len(transcript_text),
    )

    # Parse custom vocabulary
    vocab = request.custom_vocabulary
    if isinstance(vocab, str):
        vocab = [v.strip() for v in vocab.split("\n") if v.strip()]

    config = CorrectionConfig(dry_run=False)

    def _run_analysis():
        return analyze_transcript(
            transcript=request.transcript,
            file_id=request.file_id,
            custom_vocabulary=vocab,
            config=config,
        )

    try:
        result = await _run_in_thread(_run_analysis)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "=== Analyze Complete === file_id=%s | %.0fms | %d errors | field=%s | topic=%s",
            request.file_id, duration_ms,
            len(result["errors"]),
            result["topic_info"]["field"],
            result["topic_info"]["topic"],
        )

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("ASR analysis failed after %.0fms: %s", duration_ms, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR analysis failed: {str(e)}")


@router.post("/asr-correct")
async def asr_correct(
    request: ASRCorrectionRequest,
    user_id: str = Depends(get_jwt_info),
):
    """Enhance a transcript using ASR correction with custom vocabulary and OCR hints.

    This endpoint is compatible with screenapp-backend's enhance-transcript feature.
    """
    from asr_correction import correct_transcript, CorrectionConfig
    from asr_correction.vocabulary import load_domain_vocab, merge_vocabularies

    start_time = time.time()

    # Log incoming request details
    transcript_text = _extract_text(request.transcript)
    segment_count = len(request.transcript.get("segments", []))
    logger.info(
        "=== ASR Correction Request ===\n"
        "  file_id:        %s\n"
        "  user_id:        %s\n"
        "  segments:       %d\n"
        "  transcript_len: %d chars\n"
        "  has_ocr_xml:    %s\n"
        "  has_video_url:  %s\n"
        "  vocab_type:     %s",
        request.file_id,
        user_id,
        segment_count,
        len(transcript_text),
        bool(request.ocr_xml),
        bool(request.video_url),
        type(request.custom_vocabulary).__name__,
    )

    job_id = await create_job(
        "correction",
        file_id=request.file_id,
        input_summary={"file_id": request.file_id, "user_id": user_id},
    )
    logger.info("Created job_id=%s for file_id=%s", job_id, request.file_id)

    await update_job_step(job_id, "request_received", "completed", details={
        "file_id": request.file_id, "segments": segment_count, "transcript_chars": len(transcript_text)
    })

    try:
        # OCR + Whisper Pass 2 use video_url — pass it through to pipeline
        ocr_provider = None
        video_url = request.video_url if request.video_url else None
        if video_url:
            logger.info("Video URL: available (Quick OCR + Whisper Pass 2 will run)")
        else:
            logger.info("Video URL: not provided (OCR + Whisper Pass 2 skipped)")

        # Parse custom vocabulary
        vocab = request.custom_vocabulary
        if isinstance(vocab, str):
            vocab = [v.strip() for v in vocab.split("\n") if v.strip()]
        logger.info("Vocabulary: %d custom terms provided", len(vocab) if vocab else 0)

        config = CorrectionConfig(dry_run=False)
        # Apply operator-controlled per-step toggles (UI: Pipeline Control panel)
        apply_pipeline_settings(config)
        logger.info(
            "Config: dry_run=%s, confidence_threshold=%.2f, backend=%s | "
            "toggles: topic=%s web_vocab=%s validate=%s ocr=%s ocr_vocab=%s "
            "whisper2=%s avsr=%s collect=%s",
            config.dry_run, config.confidence_threshold, config.backend or "auto",
            config.enable_topic_classification,
            config.enable_web_vocab_enrichment,
            config.enable_candidate_validation,
            config.enable_ocr_extraction,
            config.enable_ocr_vocab_extraction,
            config.enable_whisper_pass2,
            config.enable_avsr,
            config.enable_data_collection,
        )

        # Build step bridge — the pipeline emits events for every internal stage,
        # this bridge writes them into MongoDB so the Pipeline Monitor sees them
        # in real time.
        loop = asyncio.get_running_loop()
        step_bridge = _make_step_bridge(loop, job_id)

        # Run the full pipeline in a worker thread.
        logger.info("Starting full correction pipeline...")
        inference_start = time.time()

        def _run_correction():
            return correct_transcript(
                transcript=request.transcript,
                file_id=request.file_id,
                custom_vocabulary=vocab,
                ocr_provider=ocr_provider,
                avsr_provider=None,  # auto-created by pipeline from config.avsr_mode
                video_url=video_url,
                config=config,
                step_callback=step_bridge,
            )

        enhanced, report = await _run_in_thread(_run_correction)
        inference_ms = (time.time() - inference_start) * 1000
        logger.info("ML pipeline completed in %.0fms", inference_ms)

        # Build per-correction results — kept on `ml_inference` for back-compat
        # with any old dashboard rendering that looks at this name. The pipeline
        # already emits `llm_reconciliation`; this is the user-facing summary.
        inference_results = []
        for r in report.results:
            inference_results.append({
                "error": r.candidate.error_found,
                "term": r.candidate.term,
                "status": "APPLIED" if r.applied else "SKIPPED",
                "confidence": round(r.confidence, 2),
                "changes": r.changes,
                "need_lip": r.need_lip,
                "ocr_hints": len(r.ocr_hints_used),
            })

        await update_job_step(job_id, "ml_inference", "completed", duration_ms=inference_ms, details={
            "corrections_attempted": report.corrections_attempted,
            "corrections_applied": report.corrections_applied,
            "results": inference_results,
            "evidence_sources": getattr(report, "evidence_sources", None),
        })

        # Build vocab_used list
        domain_vocab = load_domain_vocab(config.domain_vocab_path)
        vocab_terms = merge_vocabularies(vocab, domain_vocab)
        logger.info("Merged vocabulary: %d total terms (custom + domain)", len(vocab_terms))

        # Serialize per-correction results
        corrections_detail = []
        for r in report.results:
            corrections_detail.append({
                "term": r.candidate.term,
                "category": r.candidate.category,
                "error_found": r.candidate.error_found,
                "context": r.candidate.context,
                "char_position": r.candidate.char_position,
                "timestamp_start": r.candidate.timestamp_start,
                "timestamp_end": r.candidate.timestamp_end,
                "corrected_text": r.corrected_text,
                "changes": r.changes,
                "confidence": r.confidence,
                "need_lip": r.need_lip,
                "ocr_hints_used": r.ocr_hints_used,
                "applied": r.applied,
            })

        # Log correction results summary
        logger.info(
            "=== Correction Results ===\n"
            "  candidates_found:     %d\n"
            "  corrections_attempted: %d\n"
            "  corrections_applied:   %d\n"
            "  processing_time:       %.0fms",
            len(report.results),
            report.corrections_attempted,
            report.corrections_applied,
            report.processing_time_ms,
        )
        for r in report.results:
            status = "APPLIED" if r.applied else "SKIPPED"
            logger.info(
                "  [%s] '%s' → '%s' | confidence=%.2f | changes=%s | ocr_hints=%d",
                status,
                r.candidate.error_found,
                r.candidate.term,
                r.confidence,
                r.changes,
                len(r.ocr_hints_used),
            )

        # Save transcripts to dashboard database for comparison
        await update_job_step(job_id, "apply_corrections", "running")
        apply_step_start = time.time()
        try:
            await save_correction(
                file_id=request.file_id,
                original_text=_extract_text(request.transcript),
                enhanced_text=_extract_text(enhanced),
                corrections_detail=corrections_detail,
                report_summary={
                    "corrections_applied": report.corrections_applied,
                    "corrections_attempted": report.corrections_attempted,
                    "processing_time_ms": report.processing_time_ms,
                },
            )
            logger.info("Saved correction pair to dashboard DB for file_id=%s", request.file_id)
        except Exception as save_err:
            logger.warning("Failed to save correction to dashboard db: %s", save_err)

        apply_duration_ms = (time.time() - apply_step_start) * 1000
        await update_job_step(job_id, "apply_corrections", "completed", duration_ms=apply_duration_ms, details={
            "corrections_applied": report.corrections_applied,
        })

        duration_ms = (time.time() - start_time) * 1000

        await update_job_step(job_id, "complete", "completed", duration_ms=duration_ms, details={
            "total_time_ms": duration_ms,
        })

        await complete_job(job_id, duration_ms, {
            "corrections_applied": report.corrections_applied,
            "corrections_attempted": report.corrections_attempted,
        })

        logger.info(
            "=== Request Complete === file_id=%s | total_time=%.0fms | applied=%d/%d",
            request.file_id, duration_ms, report.corrections_applied, report.corrections_attempted,
        )

        topic_info = getattr(report, "topic_info", {})
        if topic_info:
            logger.info(
                "=== Topic Classification ===\n"
                "  Field:       %s\n"
                "  Topic:       %s\n"
                "  Description: %s\n"
                "  Suggested:   %s",
                topic_info.get("field", "unknown"),
                topic_info.get("topic", "unknown"),
                topic_info.get("description", ""),
                topic_info.get("suggested_vocab", []),
            )

        return {
            "enhanced_transcript": enhanced,
            "correction_report": {
                "file_id": report.file_id,
                "corrections_attempted": report.corrections_attempted,
                "corrections_applied": report.corrections_applied,
                "processing_time_ms": report.processing_time_ms,
                "vocab_used": vocab_terms,
                "corrections": corrections_detail,
                "selector": getattr(report, "selector_info", {}),
                "topic_info": topic_info,
            },
        }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        await fail_job(job_id, str(e), duration_ms)
        logger.error("ASR correction failed after %.0fms: %s", duration_ms, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR correction failed: {str(e)}")
