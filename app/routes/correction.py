"""POST /asr-correct — ScreenApp-compatible correction endpoint."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import get_jwt_info
from ..database import create_job, complete_job, fail_job, save_correction, get_cached_ocr, cache_ocr_result, update_job_step

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
        # Resolve OCR data: cached XML > video URL extraction > none
        ocr_provider = None
        ocr_xml_data = request.ocr_xml

        await update_job_step(job_id, "ocr_extraction", "running")
        ocr_step_start = time.time()

        if ocr_xml_data:
            logger.info("OCR: Using provided ocr_xml (%d chars)", len(ocr_xml_data))
        elif request.video_url:
            # Check dashboard DB cache first
            cached = await get_cached_ocr(request.file_id)
            if cached:
                logger.info("OCR: Using cached OCR from dashboard DB for file_id=%s (%d chars)", request.file_id, len(cached))
                ocr_xml_data = cached
            else:
                logger.info("OCR: No cache found, attempting extraction from video_url")
                from asr_correction.ocr_extractor import extract_ocr_from_video
                from asr_correction.config import CorrectionConfig as _CConfig
                _ocr_cfg = _CConfig()
                try:
                    extracted = await extract_ocr_from_video(
                        request.video_url,
                        interval_s=_ocr_cfg.ocr_frame_interval_s,
                        max_frames=_ocr_cfg.ocr_max_frames,
                        min_confidence=_ocr_cfg.ocr_min_confidence,
                        dedup_threshold=_ocr_cfg.ocr_dedup_threshold,
                    )
                    if extracted:
                        await cache_ocr_result(request.file_id, extracted)
                        ocr_xml_data = extracted
                        logger.info("OCR: Extraction successful, cached result (%d chars)", len(extracted))
                    else:
                        logger.info("OCR: Extraction returned None (stub/unavailable)")
                except Exception as ocr_err:
                    logger.warning("OCR: Extraction failed: %s", ocr_err)
        else:
            logger.info("OCR: No ocr_xml or video_url provided — running without OCR context")

        ocr_duration_ms = (time.time() - ocr_step_start) * 1000

        # Parse OCR XML to extract readable text samples for dashboard display
        ocr_output_samples = []
        if ocr_xml_data:
            try:
                from asr_correction.ocr_parser import parse_ocr_xml
                ocr_frames = parse_ocr_xml(ocr_xml_data)
                for frame in ocr_frames[:20]:  # Show up to 20 frames
                    ocr_output_samples.append({
                        "timestamp": frame.get("timestamp", ""),
                        "text": frame.get("text", "")[:200],  # Truncate long text
                    })
            except Exception:
                pass

        await update_job_step(job_id, "ocr_extraction", "completed", duration_ms=ocr_duration_ms, details={
            "has_ocr": bool(ocr_xml_data),
            "ocr_chars": len(ocr_xml_data) if ocr_xml_data else 0,
            "source": "provided" if request.ocr_xml else ("cached" if ocr_xml_data and not request.ocr_xml else "extracted"),
            "frames_with_text": len(ocr_output_samples),
            "samples": ocr_output_samples,
        })

        # AVSR extraction
        await update_job_step(job_id, "avsr_extraction", "pending" if avsr_provider else "skipped", details={
            "mode": config.avsr_mode if avsr_provider else "disabled",
            "reason": "Waiting for Pass 1 to identify segments" if avsr_provider else "AVSR disabled",
        })

        if ocr_xml_data:
            def ocr_callback(file_id: str, start_time: float, end_time: float):
                return ocr_xml_data
            ocr_provider = ocr_callback

        # Parse custom vocabulary
        await update_job_step(job_id, "vocab_merge", "running")
        vocab_step_start = time.time()
        vocab = request.custom_vocabulary
        if isinstance(vocab, str):
            vocab = [v.strip() for v in vocab.split("\n") if v.strip()]
        logger.info("Vocabulary: %d custom terms provided", len(vocab) if vocab else 0)

        config = CorrectionConfig(dry_run=False)
        logger.info(
            "Config: dry_run=%s, confidence_threshold=%.2f, backend=%s",
            config.dry_run, config.confidence_threshold, config.backend or "auto",
        )

        vocab_duration_ms = (time.time() - vocab_step_start) * 1000
        await update_job_step(job_id, "vocab_merge", "completed", duration_ms=vocab_duration_ms, details={
            "custom_terms": len(vocab) if vocab else 0
        })

        # Initialize AVSR provider
        from asr_correction.avsr import get_avsr_provider
        avsr_provider = get_avsr_provider(config.avsr_mode)
        video_url_for_avsr = request.video_url if avsr_provider else None

        if avsr_provider:
            await update_job_step(job_id, "avsr_extraction", "pending", details={"mode": config.avsr_mode})

        # Run correction in thread pool (blocking ML inference)
        await update_job_step(job_id, "candidate_detection", "running")
        await update_job_step(job_id, "ml_inference", "running")
        logger.info("Starting ML correction pipeline...")
        inference_start = time.time()

        def _run_correction():
            return correct_transcript(
                transcript=request.transcript,
                file_id=request.file_id,
                custom_vocabulary=vocab,
                ocr_provider=ocr_provider,
                avsr_provider=avsr_provider,
                video_url=video_url_for_avsr,
                config=config,
            )

        enhanced, report = await _run_in_thread(_run_correction)
        inference_ms = (time.time() - inference_start) * 1000
        logger.info("ML pipeline completed in %.0fms", inference_ms)

        # Build candidate list for dashboard
        candidate_list = []
        for r in report.results:
            candidate_list.append({
                "term": r.candidate.term,
                "error": r.candidate.error_found,
                "category": r.candidate.category,
                "timestamp": f"{r.candidate.timestamp_start:.0f}-{r.candidate.timestamp_end:.0f}s",
            })

        await update_job_step(job_id, "candidate_detection", "completed", duration_ms=inference_ms, details={
            "candidates_found": len(report.results),
            "candidates": candidate_list,
        })

        # Build per-correction results for dashboard
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
        })

        # AVSR pass 2 status
        lip_needed = [r for r in report.results if r.need_lip or (not r.applied and r.confidence < 0.8)]
        lip_flagged_terms = [{"error": r.candidate.error_found, "term": r.candidate.term,
                              "confidence": round(r.confidence, 2), "need_lip": r.need_lip}
                             for r in report.results if r.need_lip or (not r.applied and r.confidence < 0.8)]

        if avsr_provider and video_url_for_avsr and lip_needed:
            await update_job_step(job_id, "avsr_extraction", "completed", details={
                "mode": config.avsr_mode,
                "segments_analyzed": min(len(lip_needed), config.avsr_max_segments),
                "flagged_segments": lip_flagged_terms[:10],
            })
            await update_job_step(job_id, "avsr_pass2", "completed", details={
                "segments_reanalyzed": min(len(lip_needed), config.avsr_max_segments),
                "flagged_segments": lip_flagged_terms[:10],
            })
        else:
            if not avsr_provider:
                await update_job_step(job_id, "avsr_extraction", "skipped", details={
                    "mode": "disabled", "reason": "AVSR not configured",
                })
            elif not lip_needed:
                await update_job_step(job_id, "avsr_extraction", "skipped", details={
                    "mode": config.avsr_mode, "reason": "No uncertain segments — all corrections confident",
                })
            await update_job_step(job_id, "avsr_pass2", "skipped", details={
                "reason": "No segments needed AVSR" if not lip_needed else "No video URL"
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

        return {
            "enhanced_transcript": enhanced,
            "correction_report": {
                "file_id": report.file_id,
                "corrections_attempted": report.corrections_attempted,
                "corrections_applied": report.corrections_applied,
                "processing_time_ms": report.processing_time_ms,
                "vocab_used": vocab_terms,
                "corrections": corrections_detail,
            },
        }

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        await fail_job(job_id, str(e), duration_ms)
        logger.error("ASR correction failed after %.0fms: %s", duration_ms, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR correction failed: {str(e)}")
