"""GET/POST /api/pipeline/settings — operator-controlled per-step toggles."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.pipeline_settings import get_settings, update_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline Settings"])


class SettingsPatch(BaseModel):
    enable_topic_classification: bool | None = None
    enable_web_vocab_enrichment: bool | None = None
    enable_candidate_validation: bool | None = None
    enable_ocr_extraction: bool | None = None
    enable_ocr_vocab_extraction: bool | None = None
    enable_whisper_pass2: bool | None = None
    enable_avsr: bool | None = None
    enable_data_collection: bool | None = None
    avsr_mode: str | None = None
    avsr_run_on_all_flagged: bool | None = None
    avsr_min_speaking_confidence: float | None = None


@router.get("/api/pipeline/settings")
async def get_pipeline_settings() -> Dict[str, Any]:
    return get_settings()


@router.post("/api/pipeline/settings")
async def post_pipeline_settings(patch: SettingsPatch) -> Dict[str, Any]:
    body = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not body:
        raise HTTPException(status_code=400, detail="No settings to update")
    if "avsr_mode" in body and body["avsr_mode"] not in {"none", "mediapipe", "auto_avsr"}:
        raise HTTPException(status_code=400, detail="avsr_mode must be one of none|mediapipe|auto_avsr")
    return update_settings(body)
