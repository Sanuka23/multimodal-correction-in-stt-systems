"""MongoDB connection using motor (async driver).

Two databases:
  - screenapp (MONGO_DATABASE): shared with screenapp-backend, read-only from here
  - asr_correction_dashboard (MONGO_DASHBOARD_DATABASE): all dashboard data lives here
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_dashboard_db: Optional[AsyncIOMotorDatabase] = None

# Collections in dashboard database
JOBS_COLLECTION = "jobs"
CORRECTIONS_COLLECTION = "corrections"
OCR_CACHE_COLLECTION = "ocr_cache"


async def connect_db() -> None:
    global _client, _dashboard_db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongo_uri)
    _dashboard_db = _client[settings.mongo_dashboard_database]
    logger.info(
        "Connected to MongoDB at %s (dashboard db: %s)",
        settings.mongo_uri,
        settings.mongo_dashboard_database,
    )


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


def get_dashboard_db() -> AsyncIOMotorDatabase:
    if _dashboard_db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return _dashboard_db


# ── Jobs ────────────────────────────────────────────────────────────


async def create_job(job_type: str, file_id: str = None, input_summary: dict = None) -> str:
    db = get_dashboard_db()
    doc = {
        "job_type": job_type,
        "file_id": file_id,
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "completed_at": None,
        "duration_ms": None,
        "input_summary": input_summary or {},
        "result_summary": {},
        "error": None,
        "pipeline_steps": [],
    }
    result = await db[JOBS_COLLECTION].insert_one(doc)
    return str(result.inserted_id)


async def complete_job(job_id: str, duration_ms: float, result_summary: dict = None) -> None:
    from bson import ObjectId
    db = get_dashboard_db()
    await db[JOBS_COLLECTION].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc),
            "duration_ms": duration_ms,
            "result_summary": result_summary or {},
        }},
    )


async def fail_job(job_id: str, error: str, duration_ms: float = None) -> None:
    from bson import ObjectId
    db = get_dashboard_db()
    await db[JOBS_COLLECTION].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc),
            "duration_ms": duration_ms,
            "error": error,
        }},
    )


async def list_jobs(job_type: str = None, limit: int = 50) -> list:
    db = get_dashboard_db()
    query = {}
    if job_type:
        query["job_type"] = job_type
    cursor = db[JOBS_COLLECTION].find(query).sort("created_at", -1).limit(limit)
    jobs = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        jobs.append(doc)
    return jobs


# ── Corrections (transcript storage) ───────────────────────────────


async def save_correction(file_id: str, original_text: str, enhanced_text: str,
                          corrections_detail: list, report_summary: dict) -> str:
    db = get_dashboard_db()
    doc = {
        "file_id": file_id,
        "original_text": original_text,
        "enhanced_text": enhanced_text,
        "corrections": corrections_detail,
        "corrections_applied": report_summary.get("corrections_applied", 0),
        "corrections_attempted": report_summary.get("corrections_attempted", 0),
        "processing_time_ms": report_summary.get("processing_time_ms", 0),
        "created_at": datetime.now(timezone.utc),
    }
    result = await db[CORRECTIONS_COLLECTION].insert_one(doc)
    return str(result.inserted_id)


# ── OCR Cache ───────────────────────────────────────────────────────


async def get_cached_ocr(file_id: str) -> Optional[str]:
    """Check dashboard DB for cached OCR extraction."""
    db = get_dashboard_db()
    doc = await db[OCR_CACHE_COLLECTION].find_one({"file_id": file_id})
    if doc and doc.get("ocr_xml"):
        return doc["ocr_xml"]
    return None


async def cache_ocr_result(file_id: str, ocr_xml: str) -> None:
    """Cache OCR extraction result in dashboard DB."""
    db = get_dashboard_db()
    await db[OCR_CACHE_COLLECTION].update_one(
        {"file_id": file_id},
        {"$set": {
            "file_id": file_id,
            "ocr_xml": ocr_xml,
            "created_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


async def update_job_step(job_id: str, step_name: str, status: str,
                          duration_ms: float = None, details: dict = None) -> None:
    """Update a pipeline step within a job. Creates the step if it doesn't exist."""
    from bson import ObjectId
    db = get_dashboard_db()
    step = {
        "name": step_name,
        "status": status,  # "pending", "running", "completed", "failed", "skipped"
        "duration_ms": duration_ms,
        "details": details or {},
        "updated_at": datetime.now(timezone.utc),
    }
    # Use $set on the specific array element matched by name, or $push if new
    result = await db[JOBS_COLLECTION].update_one(
        {"_id": ObjectId(job_id), "pipeline_steps.name": step_name},
        {"$set": {"pipeline_steps.$": step}},
    )
    if result.matched_count == 0:
        # Step doesn't exist yet, push it
        await db[JOBS_COLLECTION].update_one(
            {"_id": ObjectId(job_id)},
            {"$push": {"pipeline_steps": step}},
        )


async def get_job_with_steps(job_id: str) -> dict:
    """Get a job document including its pipeline_steps array.

    Converts all datetime objects to ISO strings for JSON serialization.
    """
    from bson import ObjectId
    db = get_dashboard_db()
    doc = await db[JOBS_COLLECTION].find_one({"_id": ObjectId(job_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
        # Convert datetime objects to strings for JSON/Jinja2 tojson compatibility
        for step in doc.get("pipeline_steps", []):
            if isinstance(step.get("updated_at"), datetime):
                step["updated_at"] = step["updated_at"].isoformat()
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        if isinstance(doc.get("completed_at"), datetime):
            doc["completed_at"] = doc["completed_at"].isoformat()
    return doc


async def list_corrections(limit: int = 50) -> list:
    db = get_dashboard_db()
    cursor = db[CORRECTIONS_COLLECTION].find({}).sort("created_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results
