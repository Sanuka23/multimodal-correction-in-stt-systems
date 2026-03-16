"""Evaluation endpoints — WER/TTER computation."""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..database import create_job, complete_job, fail_job, list_jobs, list_corrections

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])


class EvaluateRequest(BaseModel):
    reference_text: str
    hypothesis_text: str
    target_terms: Optional[List[dict]] = None


@router.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Compare two transcripts and compute WER, CER, and optional TTER."""
    from evaluation.compare import compare_transcripts

    start_time = time.time()
    job_id = await create_job("evaluation", input_summary={
        "ref_length": len(request.reference_text),
        "hyp_length": len(request.hypothesis_text),
        "has_target_terms": request.target_terms is not None,
    })

    try:
        result = compare_transcripts(
            request.reference_text,
            request.hypothesis_text,
            request.target_terms,
        )

        duration_ms = (time.time() - start_time) * 1000
        await complete_job(job_id, duration_ms, {
            "wer": result["wer"],
            "cer": result["cer"],
            "tter": result.get("tter", {}).get("overall_tter") if result.get("tter") else None,
        })

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        await fail_job(job_id, str(e), duration_ms)
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise


@router.get("/evaluations")
async def get_evaluations(limit: int = 50):
    """List past evaluation jobs."""
    return await list_jobs(job_type="evaluation", limit=limit)


@router.get("/corrections")
async def get_corrections(limit: int = 50):
    """List past corrections from dashboard database."""
    return await list_corrections(limit=limit)
