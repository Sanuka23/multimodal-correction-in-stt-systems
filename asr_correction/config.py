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
    base_model: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
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

    # AVSR
    avsr_mode: str = "mediapipe"  # "mediapipe", "auto_avsr", "none"
    avsr_confidence_threshold: float = 0.5
    avsr_max_segments: int = 20  # Max segments to analyze in Pass 2
    avsr_model_dir: Optional[str] = None  # Path to auto_avsr repo (for auto_avsr mode)

    # Quick OCR (Step 3 — fast whole-video scan)
    quick_ocr_num_frames: int = 10

    # Whisper Pass 2 (Step 4 — re-transcribe flagged segments with vocab prompt)
    whisper_model_size: str = "small"  # "tiny", "base", "small", "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_max_segments: int = 20
    whisper_segment_padding_s: float = 2.0

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
        # Adapter path — prompt-only mode (no adapters)
        # LoRA fine-tune experiments:
        #   v1 (1000 iters): too aggressive (Kimi→Mimi, Sheldon→Tim)
        #   v2 (v1+100-200 hard neg iters): too passive (0 corrections)
        # Base Qwen3.5-9B + validation step gives best balance
        if self.adapter_path is None:
            env_val = os.environ.get("ASR_CORRECTION_ADAPTER_PATH")
            if env_val:
                self.adapter_path = env_val

        # Model path — disabled, download fresh from HuggingFace
        # Old Qwen2.5 local weights are incompatible with Qwen3.5
        if self.model_path is None:
            env_val = os.environ.get("ASR_CORRECTION_MODEL_PATH")
            if env_val:
                self.model_path = env_val
            # else: don't auto-load old local weights

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
