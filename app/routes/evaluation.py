"""Evaluation endpoints — WER/TTER computation."""

import csv
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from ..database import create_job, complete_job, fail_job, list_jobs, list_corrections
from ..config import get_settings

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

    # AMI baseline results
    ami_results = _read_json(PROJECT_ROOT / "data/eval_dataset/ami_baseline_results.json")
    # Earnings-22 baseline results
    earnings_results = _read_json(PROJECT_ROOT / "data/eval_dataset/earnings22/baseline_results.json")
    # SlideAVSR baseline results
    slideavsr_results = _read_json(PROJECT_ROOT / "data/eval_dataset/slideavsr/baseline_results.json")
    # AMI v2 (proper corpus with videos)
    ami_v2_results = _read_json(PROJECT_ROOT / "data/eval_dataset/ami_v2/baseline_results.json")

    return {
        "version": version,
        "summary": summary,
        "per_video": per_video,
        "accent_breakdown": accent_breakdown,
        "video_eval": video_eval,
        "manifest": manifest,
        "ami_baseline": ami_results,
        "earnings_baseline": earnings_results,
        "slideavsr_baseline": slideavsr_results,
        "ami_v2_baseline": ami_v2_results,
    }


# ── AMI: Load ground truth vs ScreenApp for comparison ─────────────

@router.get("/ami/compare/{meeting_id}")
async def get_ami_comparison(meeting_id: str):
    """Load AMI ground truth + ScreenApp transcript for side-by-side comparison."""
    gt_file = PROJECT_ROOT / "data" / "eval_dataset" / "transcripts" / f"ami_{meeting_id}_ground_truth.txt"
    sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "transcripts" / f"ami_{meeting_id}_screenapp.json"

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found for {meeting_id}")

    gt_text = gt_file.read_text().strip()
    sa_text = ""
    sa_segments = []
    if sa_file.exists():
        sa_data = json.loads(sa_file.read_text())
        sa_text = sa_data.get("text", "")
        sa_segments = sa_data.get("segments", [])

    return {
        "meeting_id": meeting_id,
        "ground_truth": gt_text,
        "screenapp_text": sa_text,
        "screenapp_segments": sa_segments,
    }


@router.get("/ami/list")
async def list_ami_meetings():
    """List available AMI meetings for comparison."""
    results_path = PROJECT_ROOT / "data" / "eval_dataset" / "ami_baseline_results.json"
    if not results_path.exists():
        return {"meetings": []}
    data = json.loads(results_path.read_text())
    return {"meetings": data}


@router.get("/earnings/compare/{file_id}")
async def get_earnings_comparison(file_id: str):
    """Load Earnings-22 ground truth + ScreenApp transcript for comparison."""
    gt_file = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "transcripts" / f"{file_id}_ground_truth.txt"
    sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "transcripts" / f"{file_id}_screenapp.json"

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found for {file_id}")

    gt_text = gt_file.read_text().strip()
    sa_text = ""
    if sa_file.exists():
        sa_data = json.loads(sa_file.read_text())
        sa_text = sa_data.get("text", "")

    return {"meeting_id": file_id, "ground_truth": gt_text, "screenapp_text": sa_text}


@router.get("/ami_v2/compare/{file_id}")
async def get_ami_v2_comparison(file_id: str):
    """Load AMI v2 ground truth + ScreenApp transcript for comparison."""
    gt_file = PROJECT_ROOT / "data" / "eval_dataset" / "ami_v2" / "transcripts" / f"{file_id}_ground_truth.txt"
    sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "ami_v2" / "transcripts" / f"{file_id}_screenapp.json"

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found for {file_id}")

    gt_text = gt_file.read_text().strip()
    sa_text = ""
    sa_segments = []
    if sa_file.exists():
        sa_data = json.loads(sa_file.read_text())
        sa_text = sa_data.get("text", "")
        sa_segments = sa_data.get("segments", [])

    return {
        "meeting_id": file_id,
        "ground_truth": gt_text,
        "screenapp_text": sa_text,
        "screenapp_segments": sa_segments,
    }


@router.get("/ami_v2/list")
async def list_ami_v2_meetings():
    """List available AMI v2 meetings for comparison."""
    results_path = PROJECT_ROOT / "data" / "eval_dataset" / "ami_v2" / "baseline_results.json"
    if not results_path.exists():
        return {"meetings": []}
    return {"meetings": json.loads(results_path.read_text())}


@router.get("/slideavsr/compare/{file_id}")
async def get_slideavsr_comparison(file_id: str):
    """Load SlideAVSR ground truth + ScreenApp transcript for comparison."""
    gt_file = PROJECT_ROOT / "data" / "eval_dataset" / "slideavsr" / "transcripts" / f"{file_id}_ground_truth.txt"
    sa_file = PROJECT_ROOT / "data" / "eval_dataset" / "slideavsr" / "transcripts" / f"{file_id}_screenapp.json"

    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth not found for {file_id}")

    gt_text = gt_file.read_text().strip()
    sa_text = ""
    sa_segments = []
    if sa_file.exists():
        sa_data = json.loads(sa_file.read_text())
        sa_text = sa_data.get("text", "")
        sa_segments = sa_data.get("segments", [])

    return {
        "meeting_id": file_id,
        "ground_truth": gt_text,
        "screenapp_text": sa_text,
        "screenapp_segments": sa_segments,
    }


@router.get("/slideavsr/list")
async def list_slideavsr_videos():
    """List available SlideAVSR videos for comparison."""
    results_path = PROJECT_ROOT / "data" / "eval_dataset" / "slideavsr" / "baseline_results.json"
    if not results_path.exists():
        return {"meetings": []}
    return {"meetings": json.loads(results_path.read_text())}


@router.get("/earnings/list")
async def list_earnings_calls():
    """List available Earnings-22 calls for comparison."""
    results_path = PROJECT_ROOT / "data" / "eval_dataset" / "earnings22" / "baseline_results.json"
    if not results_path.exists():
        return {"meetings": []}
    data = json.loads(results_path.read_text())
    return {"meetings": data}


# ── Annotate: Fetch file info from ScreenApp DB ────────────────────

@router.get("/annotate/file/{file_id}")
async def get_file_for_annotation(file_id: str):
    """Fetch video URL and transcript for a ScreenApp file.

    Looks up the file in the screenapp DB, returns the video URL
    and fetches the transcript JSON from S3/MinIO.
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_database]

    # Find the file by recordingId (which is the file_id in ScreenApp URLs)
    doc = await db["teamspacefilesystems"].find_one(
        {"recordingId": file_id},
        {"name": 1, "url": 1, "duration": 1, "textData": 1, "recordingId": 1}
    )

    if not doc:
        # Try by _id string match
        doc = await db["teamspacefilesystems"].find_one(
            {"_id": file_id},
            {"name": 1, "url": 1, "duration": 1, "textData": 1, "recordingId": 1}
        )

    if not doc:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    video_url = doc.get("url", "")
    name = doc.get("name", "")
    duration = doc.get("duration", 0)
    text_data = doc.get("textData", {})
    transcript_url = text_data.get("transcriptUrl", "")
    transcript_key = text_data.get("transcriptProviderKey", "")

    # Try to fetch transcript JSON
    transcript = None
    # Try transcript URL first
    if transcript_url:
        try:
            resp = urllib.request.urlopen(transcript_url, timeout=10)
            transcript = json.loads(resp.read())
        except Exception as e:
            logger.warning("Failed to fetch transcript from URL: %s", e)

    # Fallback: try direct MinIO access via provider key
    if not transcript and transcript_key:
        try:
            minio_url = f"http://localhost:9000/gcp.dev.store.screenapp.io/{transcript_key}"
            resp = urllib.request.urlopen(minio_url, timeout=10)
            transcript = json.loads(resp.read())
        except Exception as e:
            logger.warning("Failed to fetch transcript from MinIO: %s", e)

    client.close()

    return {
        "file_id": file_id,
        "name": name,
        "video_url": video_url,
        "duration": duration,
        "transcript": transcript,
        "has_transcript": transcript is not None,
        "transcript_key": transcript_key,
    }
