"""Model loading and inference for ASR correction.

Supports MLX (Apple Silicon) and transformers (Linux) backends.
Model is loaded as a singleton and reused across calls.
"""

from __future__ import annotations

import json
import logging
import platform
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Singleton model state
_model_instance = None
_tokenizer_instance = None
_backend: Optional[str] = None


def detect_backend() -> str:
    """Auto-detect whether to use MLX or transformers."""
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import transformers  # noqa: F401
        return "transformers"
    except ImportError:
        pass
    return "none"


def load_model(
    adapter_path: Optional[str] = None,
    model_path: Optional[str] = None,
    base_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    backend: Optional[str] = None,
) -> Tuple:
    """Load model with LoRA adapters. Cached as singleton.

    Args:
        adapter_path: Path to LoRA adapter directory.
        model_path: Path to local model weights. If provided, loads
            from this directory instead of downloading from HuggingFace.
        base_model: HuggingFace model ID (fallback if model_path not set).
        backend: "mlx", "transformers", or None for auto-detect.

    Returns:
        (model, tokenizer) tuple. Both may be None if no backend available.
    """
    global _model_instance, _tokenizer_instance, _backend

    if _model_instance is not None:
        return _model_instance, _tokenizer_instance

    _backend = backend or detect_backend()

    # Resolve adapter path
    if adapter_path and not Path(adapter_path).exists():
        logger.warning("Adapter path not found: %s. Running without adapters.", adapter_path)
        adapter_path = None

    # Use local model weights if available, otherwise fall back to HuggingFace ID
    model_source = model_path if model_path and Path(model_path).exists() else base_model

    if _backend == "mlx":
        from mlx_lm import load

        logger.info("Loading MLX model from %s (adapters: %s)", model_source, adapter_path)
        _model_instance, _tokenizer_instance = load(
            model_source, adapter_path=adapter_path
        )
    elif _backend == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading transformers model from %s", model_source)
        _tokenizer_instance = AutoTokenizer.from_pretrained(model_source)
        base = AutoModelForCausalLM.from_pretrained(
            model_source, device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel

            _model_instance = PeftModel.from_pretrained(base, adapter_path)
        else:
            _model_instance = base
    else:
        logger.warning("No ML backend available. Corrections will use dry-run mode.")
        return None, None

    return _model_instance, _tokenizer_instance


def build_prompt(
    asr_text: str,
    vocab: list,
    category: str,
    ocr_hints: Optional[list] = None,
) -> str:
    """Build the user prompt matching training data format exactly."""
    prompt = (
        "Correct this ASR transcript segment using the provided context.\n"
        "IMPORTANT: Only change words that are clearly wrong. If a word is "
        "already correct, do NOT change it. Use OCR screen text to verify "
        "the correct spelling of domain terms.\n\n"
        f"ASR transcript: {asr_text}\n"
        f"Custom vocabulary: {json.dumps(vocab)}\n"
        f"Category: {category}\n"
    )

    if ocr_hints:
        prompt += (
            f"Screen text (OCR from slides/UI visible during this segment):\n"
            f"  {chr(10).join('- ' + h for h in ocr_hints)}\n"
            "Use these screen terms to verify correct spellings.\n"
        )
    else:
        prompt += "OCR hints: none available\n"

    prompt += "Lip reading hint: null"
    return prompt


def run_inference(
    prompt: str,
    system_prompt: str,
    model=None,
    tokenizer=None,
    max_tokens: int = 512,
) -> dict:
    """Run inference and parse JSON response.

    Returns dict with: corrected, changes, confidence, need_lip
    """
    import time as _time

    if model is None or tokenizer is None:
        logger.warning("Model/tokenizer is None — returning fallback response")
        return _fallback_response()

    logger.info("Running inference (backend=%s, max_tokens=%d)", _backend, max_tokens)
    logger.debug("Prompt: %s", prompt[:200])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    t0 = _time.time()

    if _backend == "mlx":
        from mlx_lm import generate

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=max_tokens
        )
    elif _backend == "transformers":
        import torch

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    else:
        return _fallback_response()

    inference_ms = (_time.time() - t0) * 1000
    logger.info("Inference completed in %.0fms", inference_ms)
    logger.info("Raw model response: %s", response[:500] if response else "(empty)")

    result = _parse_response(response)
    logger.info("Parsed result: confidence=%.2f, changes=%s, need_lip=%s",
                result.get("confidence", 0), result.get("changes", []), result.get("need_lip", False))
    return result


def _parse_response(response: str) -> dict:
    """Parse model JSON output. Handles malformed JSON gracefully."""
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "corrected": parsed.get("corrected"),
                "changes": parsed.get("changes", []),
                "confidence": float(parsed.get("confidence", 0.0)),
                "need_lip": bool(parsed.get("need_lip", False)),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return _fallback_response(raw=response)


def _fallback_response(raw: str = "") -> dict:
    result = {
        "corrected": None,
        "changes": [],
        "confidence": 0.0,
        "need_lip": False,
    }
    if raw:
        result["raw_response"] = raw
    return result


def unload_model():
    """Explicitly unload model to free memory."""
    global _model_instance, _tokenizer_instance, _backend
    _model_instance = None
    _tokenizer_instance = None
    _backend = None
