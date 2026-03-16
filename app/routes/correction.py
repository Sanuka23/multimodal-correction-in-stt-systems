"""POST /asr-correct — ScreenApp-compatible correction endpoint."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import get_jwt_info
from ..database import create_job, complete_job, fail_job, save_correction, get_cached_ocr, cache_ocr_result

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
    job_id = await create_job(
        "correction",
        file_id=request.file_id,
        input_summary={"file_id": request.file_id, "user_id": user_id},
    )

    try:
        # Resolve OCR data: cached XML > video URL extraction > none
        ocr_provider = None
        ocr_xml_data = request.ocr_xml

        if not ocr_xml_data and request.video_url:
            # Check dashboard DB cache first
            cached = await get_cached_ocr(request.file_id)
            if cached:
                logger.info("Using cached OCR for file_id=%s", request.file_id)
                ocr_xml_data = cached
            else:
                # Try extracting OCR from video (stub for now)
                from asr_correction.ocr_extractor import extract_ocr_from_video
                try:
                    extracted = await extract_ocr_from_video(request.video_url)
                    if extracted:
                        await cache_ocr_result(request.file_id, extracted)
                        ocr_xml_data = extracted
                except Exception as ocr_err:
                    logger.warning("OCR extraction failed: %s", ocr_err)

        if ocr_xml_data:
            def ocr_callback(file_id: str, start_time: float, end_time: float):
                return ocr_xml_data
            ocr_provider = ocr_callback

        # Parse custom vocabulary
        vocab = request.custom_vocabulary
        if isinstance(vocab, str):
            vocab = [v.strip() for v in vocab.split("\n") if v.strip()]

        config = CorrectionConfig(dry_run=False)

        # Run correction in thread pool (blocking ML inference)
        def _run_correction():
            return correct_transcript(
                transcript=request.transcript,
                file_id=request.file_id,
                custom_vocabulary=vocab,
                ocr_provider=ocr_provider,
                config=config,
            )

        enhanced, report = await _run_in_thread(_run_correction)

        # Build vocab_used list
        domain_vocab = load_domain_vocab(config.domain_vocab_path)
        vocab_terms = merge_vocabularies(vocab, domain_vocab)

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

        # Save transcripts to dashboard database for comparison
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
        except Exception as save_err:
            logger.warning("Failed to save correction to dashboard db: %s", save_err)

        duration_ms = (time.time() - start_time) * 1000
        await complete_job(job_id, duration_ms, {
            "corrections_applied": report.corrections_applied,
            "corrections_attempted": report.corrections_attempted,
        })

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
        logger.error("ASR correction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR correction failed: {str(e)}")
