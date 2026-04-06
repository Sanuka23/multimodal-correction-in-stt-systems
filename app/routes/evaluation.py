"""Evaluation endpoints — WER/TTER computation."""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..database import create_job, complete_job, fail_job, list_jobs, list_corrections

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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


# ---------------------------------------------------------------------------
# Evaluation results reader
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON file, returning None if the file is missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _read_csv(path: Path) -> Optional[List[Dict[str, str]]]:
    """Read a CSV file into a list of dicts, returning None if missing."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Convert numeric-looking values
                parsed: Dict[str, Any] = {}
                for k, v in row.items():
                    try:
                        parsed[k] = int(v)
                    except (ValueError, TypeError):
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                rows.append(parsed)
            return rows
    except (csv.Error, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


@router.get("/eval/results")
async def get_eval_results(
    version: str = Query("v2", pattern="^v[12]$", description="Evaluation version: v1 or v2"),
):
    """Read evaluation results from disk for the given version."""

    results_dir = PROJECT_ROOT / ("data/eval_results" if version == "v1" else "data/eval_results_v2")

    summary = _read_json(results_dir / "summary.json")
    per_video = _read_csv(results_dir / "per_video.csv")
    accent_breakdown = _read_csv(results_dir / "accent_breakdown.csv")

    # Shared video eval data (lives outside versioned dirs)
    video_eval_path = PROJECT_ROOT / "data/eval_videos/comparison_results.json"
    manifest_path = PROJECT_ROOT / "data/eval_videos/manifest.json"

    video_eval = _read_json(video_eval_path)
    manifest = _read_json(manifest_path)

    return {
        "version": version,
        "summary": summary,
        "per_video": per_video,
        "accent_breakdown": accent_breakdown,
        "video_eval": video_eval,
        "manifest": manifest,
    }
