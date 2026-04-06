"""Training endpoints — trigger LoRA fine-tuning and check status."""

import functools
import itertools
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..database import create_job, complete_job, fail_job, list_jobs

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Known datasets — id -> (display name, path relative to PROJECT_ROOT, list of JSONL files)
_KNOWN_DATASETS = {
    "collected_data": {
        "name": "Collected Data (Main)",
        "path": "data/collected_data",
        "files": ["train.jsonl", "valid.jsonl"],
    },
    "training_pairs": {
        "name": "Training Pairs (ScreenApp)",
        "path": "data/training_pairs",
        "files": ["screenapp_final.jsonl"],
    },
    "hard_negatives": {
        "name": "Hard Negatives",
        "path": "data/hard_negatives",
        "files": ["hard_negatives.jsonl", "train.jsonl", "valid.jsonl"],
    },
    "live_corrections": {
        "name": "Live Corrections",
        "path": "asr_correction/collected_data",
        "files": ["corrections.jsonl"],
    },
}

# Track the current training thread
_training_lock = threading.Lock()
_training_thread: Optional[threading.Thread] = None


class TrainRequest(BaseModel):
    iterations: int = 1000
    batch_size: int = 2
    learning_rate: float = 2e-5
    data_dir: str = "./data/collected_data"


def _run_training(job_id: str, config_kwargs: dict):
    """Run training in a background thread."""
    import asyncio
    from training.train_lora import TrainingConfig, train_with_mlx

    start_time = time.time()
    try:
        config = TrainingConfig(**config_kwargs)
        result = train_with_mlx(config)
        duration_ms = (time.time() - start_time) * 1000

        # Update job in MongoDB (need event loop for async)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(complete_job(job_id, duration_ms, {
            "duration_seconds": result.get("duration_seconds"),
            "iterations": config_kwargs.get("iterations", 1000),
        }))
        loop.close()
        logger.info("Training completed in %.1f minutes", duration_ms / 60000)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fail_job(job_id, str(e), duration_ms))
        loop.close()
        logger.error("Training failed: %s", e, exc_info=True)


@router.post("/train")
async def start_training(request: TrainRequest):
    """Trigger LoRA fine-tuning in the background."""
    global _training_thread

    with _training_lock:
        if _training_thread and _training_thread.is_alive():
            return {"status": "already_running", "message": "Training is already in progress"}

    job_id = await create_job("training", input_summary={
        "iterations": request.iterations,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
    })

    config_kwargs = {
        "iterations": request.iterations,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "data_dir": Path(request.data_dir),
    }

    _training_thread = threading.Thread(
        target=_run_training,
        args=(job_id, config_kwargs),
        daemon=True,
    )
    _training_thread.start()

    return {"job_id": job_id, "status": "started"}


@router.get("/training/status")
async def training_status():
    """Get the latest training job status."""
    jobs = await list_jobs(job_type="training", limit=1)
    if not jobs:
        return {"status": "no_training_jobs"}

    latest = jobs[0]
    is_running = _training_thread and _training_thread.is_alive()
    latest["is_running"] = is_running
    return latest


# ---------------------------------------------------------------------------
# Dataset browsing helpers
# ---------------------------------------------------------------------------

def _count_lines_cached(file_path: Path) -> int:
    """Count lines in a file, with a short-lived cache."""
    return _count_lines_impl(str(file_path), int(time.time() // 60))


@functools.lru_cache(maxsize=64)
def _count_lines_impl(file_path_str: str, _cache_key: int) -> int:
    """Count lines in a file.  ``_cache_key`` changes every 60 s to expire the cache."""
    p = Path(file_path_str)
    if not p.exists():
        return 0
    count = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


@router.get("/training/datasets")
async def list_datasets():
    """Return a list of available training datasets with stats."""
    datasets = []
    for ds_id, ds_info in _KNOWN_DATASETS.items():
        ds_path = PROJECT_ROOT / ds_info["path"]
        files = []
        total = 0
        for fname in ds_info["files"]:
            fpath = ds_path / fname
            count = _count_lines_cached(fpath)
            files.append({"name": fname, "count": count})
            total += count

        entry: dict = {
            "id": ds_id,
            "name": ds_info["name"],
            "path": ds_info["path"],
            "files": files,
            "total": total,
        }

        # For collected_data, enrich with metadata.json
        if ds_id == "collected_data":
            meta_path = ds_path / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    entry["metadata"] = {
                        "accents": meta.get("accents", {}),
                        "positive": meta.get("positive_count"),
                        "negative": meta.get("negative_count"),
                    }
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read metadata.json: %s", exc)

        datasets.append(entry)

    return {"datasets": datasets}


@router.get("/training/data")
async def browse_dataset(
    dataset: str = Query(..., description="Dataset ID"),
    file: str = Query(..., description="JSONL file name"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=200, description="Items per page"),
):
    """Paginated JSONL reader for a training dataset."""

    # Validate dataset ID
    if dataset not in _KNOWN_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

    ds_info = _KNOWN_DATASETS[dataset]

    # Validate file name (prevent path traversal)
    if file not in ds_info["files"]:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file}' not in dataset '{dataset}'. Available: {ds_info['files']}",
        )

    file_path = (PROJECT_ROOT / ds_info["path"] / file).resolve()

    # Extra safety: ensure resolved path is still under PROJECT_ROOT
    if not str(file_path).startswith(str(PROJECT_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {ds_info['path']}/{file}")

    total = _count_lines_cached(file_path)
    skip = (page - 1) * limit
    entries = []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            # Skip to the right page without loading everything
            sliced = itertools.islice(f, skip, skip + limit)
            for idx, line in enumerate(sliced):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = {"_raw": line}

                # Extract metadata and messages
                metadata = obj.get("metadata", {})
                messages = obj.get("messages", [])

                entries.append({
                    "index": skip + idx,
                    "metadata": {
                        "source": metadata.get("source"),
                        "accent": metadata.get("accent"),
                        "term": metadata.get("term"),
                        "category": metadata.get("category"),
                        "error_found": metadata.get("error_found"),
                        "applied": metadata.get("applied"),
                        "is_negative": metadata.get("is_negative"),
                        "has_ocr": metadata.get("has_ocr"),
                    },
                    "messages": messages,
                })
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Error reading file: {exc}")

    return {
        "entries": entries,
        "total": total,
        "page": page,
        "limit": limit,
        "dataset": dataset,
        "file": file,
    }
