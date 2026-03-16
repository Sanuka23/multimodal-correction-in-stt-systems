"""Training endpoints — trigger LoRA fine-tuning and check status."""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..database import create_job, complete_job, fail_job, list_jobs

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])

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
