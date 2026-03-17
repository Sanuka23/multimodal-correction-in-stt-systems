"""Configuration for the ASR correction module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_MODULE_DIR = Path(__file__).parent


@dataclass
class CorrectionConfig:
    """All tunable parameters for the correction pipeline."""

    # Model
    adapter_path: Optional[str] = None
    model_path: Optional[str] = None
    base_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    confidence_threshold: float = 0.7
    max_tokens: int = 512

    # Vocabulary
    domain_vocab_path: Optional[str] = None
    custom_vocabulary: Optional[List[dict]] = None

    # OCR (hint extraction — used by corrector.py)
    ocr_window_seconds: float = 15.0
    max_ocr_hints: int = 15

    # OCR Extraction (used by ocr_extractor.py)
    # 30s interval — slides/dashboards rarely change faster in meetings
    # For a 25-min video: ~50 frames → after pixel dedup ~20-30 unique
    ocr_frame_interval_s: float = 30.0
    ocr_max_frames: int = 100
    ocr_min_confidence: float = 0.5
    ocr_dedup_threshold: float = 0.9

    # Context
    context_window_chars: int = 80
    min_context_length: int = 10

    # Data collection
    collect_data: bool = True
    data_output_dir: Optional[str] = None

    # Runtime
    dry_run: bool = False
    backend: Optional[str] = None

    # System prompt (must match training data format)
    system_prompt: str = (
        "You are an ASR transcript correction model. Given a noisy ASR transcript "
        "segment and context signals, detect errors in context-critical terms and "
        "output the corrected transcript with changes noted."
    )

    def __post_init__(self):
        """Resolve paths relative to module directory and apply env overrides."""
        # Adapter path
        if self.adapter_path is None:
            env_val = os.environ.get("ASR_CORRECTION_ADAPTER_PATH")
            if env_val:
                self.adapter_path = env_val
            else:
                default = str(_MODULE_DIR / "adapters")
                if Path(default).exists():
                    self.adapter_path = default

        # Model path (local weights)
        if self.model_path is None:
            env_val = os.environ.get("ASR_CORRECTION_MODEL_PATH")
            if env_val:
                self.model_path = env_val
            else:
                default = str(_MODULE_DIR / "model_weights")
                if Path(default).exists():
                    self.model_path = default

        # Domain vocab
        if self.domain_vocab_path is None:
            env_val = os.environ.get("ASR_CORRECTION_VOCAB_PATH")
            if env_val:
                self.domain_vocab_path = env_val
            else:
                default = str(_MODULE_DIR / "domain_vocab.json")
                if Path(default).exists():
                    self.domain_vocab_path = default

        # Data output
        if self.data_output_dir is None:
            env_val = os.environ.get("ASR_CORRECTION_DATA_DIR")
            if env_val:
                self.data_output_dir = env_val
            else:
                self.data_output_dir = str(_MODULE_DIR / "collected_data")
