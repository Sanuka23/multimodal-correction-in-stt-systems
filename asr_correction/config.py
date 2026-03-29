"""Configuration for the ASR correction module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_MODULE_DIR = Path(__file__).parent


@dataclass
class CorrectionConfig:
    """All tunable parameters for the v2 Whisper reconciliation pipeline."""

    # Model (Qwen3.5-9B for LLM detection + reconciliation)
    adapter_path: Optional[str] = None
    model_path: Optional[str] = None
    base_model: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    confidence_threshold: float = 0.7
    max_tokens: int = 512

    # Vocabulary
    domain_vocab_path: Optional[str] = None

    # Context
    context_window_chars: int = 80
    min_context_length: int = 10

    # Quick OCR (Step 3 — extract frames from video for screen text)
    quick_ocr_num_frames: int = 10

    # Whisper Pass 2 (Step 4 — re-transcribe flagged segments)
    whisper_model_size: str = "small"  # "tiny", "base", "small", "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_max_segments: int = 20
    whisper_segment_padding_s: float = 2.0

    # Data collection
    collect_data: bool = True
    data_output_dir: Optional[str] = None

    # Runtime
    dry_run: bool = False
    backend: Optional[str] = None

    # System prompt
    system_prompt: str = (
        "You are an ASR transcript correction model. Given a noisy ASR transcript "
        "segment and context signals, detect errors in context-critical terms and "
        "output the corrected transcript with changes noted."
    )

    def __post_init__(self):
        """Resolve paths relative to module directory and apply env overrides."""
        # Adapter path (prompt-only mode by default — no LoRA adapters)
        if self.adapter_path is None:
            env_val = os.environ.get("ASR_CORRECTION_ADAPTER_PATH")
            if env_val:
                self.adapter_path = env_val

        # Model path
        if self.model_path is None:
            env_val = os.environ.get("ASR_CORRECTION_MODEL_PATH")
            if env_val:
                self.model_path = env_val

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
