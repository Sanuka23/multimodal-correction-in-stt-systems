"""Dashboard routes — Jinja2 rendered pages."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..database import list_jobs, get_job_with_steps

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
