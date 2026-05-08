"""Dashboard routes — JSON API consumed by the React frontend."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from ..database import list_jobs, get_job_with_steps

router = APIRouter(tags=["Dashboard"])


@router.get("/api/jobs/{job_id}/steps")
async def get_job_steps(job_id: str):
    """Polling endpoint for pipeline step progress."""
    job = await get_job_with_steps(job_id)
    if not job:
        return {"error": "Job not found"}
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "pipeline_steps": job.get("pipeline_steps", []),
    }


@router.get("/api/stats")
async def api_stats():
    """Dashboard stats: correction counts, rates, averages."""
    jobs = await list_jobs(limit=200)
    correction_jobs = [j for j in jobs if j.get("job_type") == "correction"]
    completed = [j for j in correction_jobs if j.get("status") == "completed"]

    total_applied = sum(j.get("result_summary", {}).get("corrections_applied", 0) for j in completed)
    total_attempted = sum(j.get("result_summary", {}).get("corrections_attempted", 0) for j in completed)
    durations = [j["duration_ms"] for j in completed if j.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    confidences = []
    for j in completed:
        conf = j.get("result_summary", {}).get("avg_confidence")
        if conf:
            confidences.append(conf)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return {
        "correction_count": len(correction_jobs),
        "total_applied": total_applied,
        "total_attempted": total_attempted,
        "avg_confidence": round(avg_confidence, 3),
        "success_rate": round(total_applied / total_attempted, 3) if total_attempted > 0 else 0,
        "avg_duration_ms": round(avg_duration, 1),
    }


@router.get("/api/jobs")
async def api_jobs(
    limit: int = Query(20, ge=1, le=200),
    page: int = Query(1, ge=1),
    job_type: Optional[str] = None,
    status: Optional[str] = None,
):
    """Paginated job list for React frontend."""
    all_jobs = await list_jobs(limit=500, job_type=job_type)
    if status:
        all_jobs = [j for j in all_jobs if j.get("status") == status]

    total = len(all_jobs)
    start = (page - 1) * limit
    page_jobs = all_jobs[start:start + limit]

    for j in page_jobs:
        if "_id" in j:
            j["_id"] = str(j["_id"])

    return {
        "jobs": page_jobs,
        "total": total,
        "page": page,
        "limit": limit,
    }


@router.get("/api/jobs/stats")
async def api_jobs_stats():
    """Daily correction counts for last 7 days (for bar chart)."""
    days = []
    for i in range(6, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        day_name = day.strftime("%a").upper()
        days.append({"day": day_name, "date": day.strftime("%Y-%m-%d"), "count": 0})

    jobs = await list_jobs(limit=500)
    for j in jobs:
        created = j.get("created_at")
        if not created:
            continue
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                continue
        date_str = created.strftime("%Y-%m-%d")
        for d in days:
            if d["date"] == date_str:
                d["count"] += 1

    return days


@router.get("/api/health")
async def api_health():
    """Pipeline component health status."""
    try:
        from asr_correction.model import _model_instance
        model_loaded = _model_instance is not None
    except Exception:
        model_loaded = False

    try:
        from asr_correction.config import CorrectionConfig
        config = CorrectionConfig()
        avsr_mode = config.avsr_mode
    except Exception:
        avsr_mode = "mediapipe"

    return {
        "model_loaded": model_loaded,
        "model_name": "Qwen3.5-9B-MLX-4bit",
        "adapter_path": "asr_correction/adapters",
        "ocr_status": "active",
        "ocr_engine": "PaddleOCR v4",
        "avsr_mode": avsr_mode,
        "latency_p50_ms": 161000,
    }
