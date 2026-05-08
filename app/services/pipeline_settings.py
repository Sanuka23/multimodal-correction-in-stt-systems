"""Pipeline settings — operator-controlled per-step toggles.

Persisted to a small JSON file so the FastAPI process and the CLI
share the same source of truth. Defaults: every stage enabled.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "enable_topic_classification": True,
    "enable_web_vocab_enrichment": True,
    "enable_candidate_validation": True,
    "enable_ocr_extraction": True,
    "enable_ocr_vocab_extraction": True,
    "enable_whisper_pass2": True,
    "enable_avsr": True,
    "enable_data_collection": True,
    # AVSR sub-settings
    "avsr_mode": "mediapipe",                 # none | mediapipe | auto_avsr
    "avsr_run_on_all_flagged": False,         # bypass category gate
    "avsr_min_speaking_confidence": 0.55,
}

_LOCK = threading.Lock()
_PATH = Path(
    os.environ.get(
        "ASR_PIPELINE_SETTINGS_PATH",
        Path(__file__).resolve().parents[2] / "data" / "pipeline_settings.json",
    )
)


def _ensure_dir() -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Dict[str, Any]:
    """Return current settings (defaults filled in)."""
    with _LOCK:
        try:
            if _PATH.exists():
                with open(_PATH) as fh:
                    stored = json.load(fh) or {}
            else:
                stored = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed reading %s — falling back to defaults: %s", _PATH, exc)
            stored = {}
    merged = {**_DEFAULT_SETTINGS, **stored}
    return merged


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `patch` into stored settings; return the merged result."""
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be a dict")
    # Whitelist — only known keys are persisted.
    clean = {k: v for k, v in patch.items() if k in _DEFAULT_SETTINGS}
    with _LOCK:
        try:
            if _PATH.exists():
                with open(_PATH) as fh:
                    stored = json.load(fh) or {}
            else:
                stored = {}
        except (OSError, json.JSONDecodeError):
            stored = {}
        stored.update(clean)
        _ensure_dir()
        with open(_PATH, "w") as fh:
            json.dump(stored, fh, indent=2, sort_keys=True)
    logger.info("Pipeline settings updated: %s", clean)
    return {**_DEFAULT_SETTINGS, **stored}


def apply_to_config(config) -> None:
    """Mutate a `CorrectionConfig` in place using current settings."""
    s = get_settings()
    for key, val in s.items():
        if hasattr(config, key):
            setattr(config, key, val)
