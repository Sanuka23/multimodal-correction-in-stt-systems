"""Dashboard routes — Jinja2 rendered pages + JSON API for React frontend."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates

from ..database import list_jobs, get_job_with_steps, list_corrections

PROJECT_ROOT = Path(__file__).parent.parent.parent
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard")
async def dashboard_home(request: Request):
    """Overview page with recent jobs and stats."""
    recent_jobs = await list_jobs(limit=20)
    correction_jobs = [j for j in recent_jobs if j["job_type"] == "correction"]
    eval_jobs = [j for j in recent_jobs if j["job_type"] == "evaluation"]

    completed_corrections = [j for j in correction_jobs if j["status"] == "completed"]
    total_corrections = sum(
        j.get("result_summary", {}).get("corrections_applied", 0)
        for j in completed_corrections
    )
    total_attempted = sum(
        j.get("result_summary", {}).get("corrections_attempted", 0)
        for j in completed_corrections
    )
    avg_duration = 0
    completed = [j for j in correction_jobs if j.get("duration_ms")]
    if completed:
        avg_duration = sum(j["duration_ms"] for j in completed) / len(completed)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "recent_jobs": recent_jobs[:10],
        "total_corrections": total_corrections,
        "total_attempted": total_attempted,
        "correction_count": len(correction_jobs),
        "eval_count": len(eval_jobs),
        "avg_duration_ms": round(avg_duration, 1),
    })


@router.get("/dashboard/compare")
async def compare_page(request: Request):
    """Transcript comparison page."""
    return templates.TemplateResponse("compare.html", {"request": request})


@router.get("/dashboard/jobs")
async def jobs_page(request: Request):
    """Full job history table."""
    jobs = await list_jobs(limit=100)
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})


@router.get("/dashboard/pipeline/{job_id}")
async def pipeline_page(request: Request, job_id: str):
    """Pipeline workflow visualization for a specific job."""
    job = await get_job_with_steps(job_id)
    if not job:
        return templates.TemplateResponse("pipeline.html", {
            "request": request, "job": None, "job_id": job_id
        })
    return templates.TemplateResponse("pipeline.html", {
        "request": request, "job": job, "job_id": job_id
    })


@router.get("/api/jobs/{job_id}/steps")
async def get_job_steps(job_id: str):
    """API endpoint for polling pipeline step progress."""
    job = await get_job_with_steps(job_id)
    if not job:
        return {"error": "Job not found"}
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "pipeline_steps": job.get("pipeline_steps", []),
    }


@router.get("/dashboard/training")
async def training_page(request: Request):
    """Training status and trigger page."""
    training_jobs = await list_jobs(job_type="training", limit=10)
    return templates.TemplateResponse("training.html", {
        "request": request,
        "training_jobs": training_jobs,
    })


# =====================================================================
# JSON API Routes for React Frontend
# =====================================================================


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

    # Convert ObjectId to string
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
            except:
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
        "model_name": "Qwen2.5-7B-Instruct-4bit",
        "adapter_path": "asr_correction/adapters",
        "ocr_status": "active",
        "ocr_engine": "PaddleOCR v4",
        "avsr_mode": avsr_mode,
        "latency_p50_ms": 161000,
    }
