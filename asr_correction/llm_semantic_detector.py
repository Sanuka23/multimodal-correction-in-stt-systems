"""Phase 4 — Channel B: LLM semantic error detection.

Sends a meeting transcript to the local MLX LLM and asks it to identify
words that look semantically out of place or are likely mis-transcriptions
of proper nouns, technical terms, or business jargon. The LLM returns
structured JSON with {original, target, confidence, reason}. The caller
applies corrections above a strict confidence threshold.

This is the Phase 4 "LLM-only" tier of the design: no hard evidence is
required — the LLM reasons from context alone. Because this is risky,
the default confidence threshold is very high (0.90) and the system
prompt is explicit about the bias toward precision over recall.

Spec: docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.1 Channel B
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .dynamic_detector import DynamicCandidate
from .model import load_model, run_inference_raw

logger = logging.getLogger(__name__)

# Chunk size in characters. Keep under the model's attention window.
_CHUNK_CHAR_SIZE = 4000
_CHUNK_OVERLAP = 200

_SYSTEM_PROMPT = (
    "You are an ASR error detector for meeting transcripts. Your job is to "
    "find words that are likely mis-transcriptions of proper nouns, technical "
    "terms, product names, person names, or domain-specific jargon. "
    "You only flag words that are CLEARLY wrong given the surrounding context. "
    "You NEVER flag common English words, filler words, or words that might "
    "simply be uncommon but correctly spelled. You prefer high precision over "
    "high recall: if you are not sure, return an empty list. "
    "Keep 'reason' fields to at most 10 words — long explanations waste tokens "
    "and risk truncation. "
    "You output only valid JSON. Your response MUST be a single JSON array."
)

_USER_PROMPT_TEMPLATE = """\
Below is a transcript chunk from a meeting about: {topic_context}

Identify words that are likely ASR mis-transcriptions of specific domain
vocabulary (proper nouns, technical terms, product names, business terms).
For each suspected error, return a JSON object with:

  "original":   the exact word as it appears in the transcript
  "target":     your best guess at the correct word
  "confidence": float 0.0-1.0 (how sure you are this is an error)
  "reason":     short explanation (max 10 words)

Only include corrections you are at least 85% confident about. The target
must be a real term that fits the meeting topic. If nothing looks wrong,
return [].

OUTPUT FORMAT (JSON array only, no prose):
[
  {{"original": "Kime", "target": "Kimi K2", "confidence": 0.95, "reason": "AI model name"}},
  ...
]

TRANSCRIPT CHUNK:
\"\"\"
{chunk}
\"\"\"

JSON:"""


@dataclass
class LlmProposal:
    """One correction proposed by the LLM."""
    original: str
    target: str
    confidence: float
    reason: str
    chunk_index: int


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of up to _CHUNK_CHAR_SIZE chars.

    Tries to break on sentence boundaries where possible. Small inputs
    are returned as a single chunk.
    """
    text = text.strip()
    if len(text) <= _CHUNK_CHAR_SIZE:
        return [text] if text else []

    chunks: list[str] = []
    i = 0
    while i < len(text):
        end = min(i + _CHUNK_CHAR_SIZE, len(text))
        # Try to break on a sentence boundary inside the last 500 chars.
        if end < len(text):
            window = text[max(i, end - 500):end]
            sep = max(
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
            )
            if sep > 0:
                end = max(i, end - 500) + sep + 2
        chunks.append(text[i:end])
        if end >= len(text):
            break
        i = max(0, end - _CHUNK_OVERLAP)
    return chunks


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]", re.DOTALL)


def _parse_llm_response(raw: str) -> list[dict]:
    """Extract and parse the first JSON array from raw LLM output.

    Robust to:
    - Markdown code fences around the JSON
    - Preceding prose or chain-of-thought
    - Trailing commentary after the array
    - Truncated responses (recovers complete objects from an incomplete array)
    Returns [] on any parse failure (logged as a warning).
    """
    if not raw:
        return []

    # Strip common code-fence wrappers
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try direct parse first
    for candidate in (cleaned, _extract_first_array(cleaned)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

    # Truncation recovery: scan from the first `[` and extract every
    # complete top-level object we can parse, then wrap them in a fresh array.
    recovered = _recover_truncated_objects(cleaned)
    if recovered:
        return recovered

    logger.warning("LLM returned unparseable response: %r", raw[:200])
    return []


def _recover_truncated_objects(text: str) -> list[dict]:
    """Scan `text` for complete top-level `{...}` objects and parse each.

    Useful when the model's final object was cut off by max_tokens. We drop
    the incomplete tail and keep whatever full objects we found.
    """
    start = text.find("[")
    if start == -1:
        return []
    out: list[dict] = []
    i = start + 1
    n = len(text)
    while i < n:
        # Skip whitespace, commas
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n or text[i] != "{":
            break
        # Find matching close brace, honoring strings.
        depth = 0
        j = i
        in_str = False
        escape = False
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j + 1])
                            if isinstance(obj, dict):
                                out.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            # Unclosed object → truncation boundary, stop.
            break
    return out


def _extract_first_array(text: str) -> str | None:
    """Find the first bracketed JSON array in `text` using a simple
    depth-counted scan (the regex approach can miss nested structures)."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def detect_candidates_llm_semantic(
    transcript: dict,
    *,
    max_chunks: int = 4,
    model_handles: tuple | None = None,
    max_tokens: int = 2048,
    topic_context: str = "",
) -> list[LlmProposal]:
    """Run Channel B detection on a ScreenApp transcript.

    Args:
        transcript: ScreenApp transcript dict with `text` and `segments`.
        max_chunks: cap on number of chunks sent to the LLM per transcript.
            Keeps latency bounded on long meetings.
        model_handles: pre-loaded (model, tokenizer) tuple. If None, the
            model is loaded via load_model() on the first call.
        max_tokens: per-chunk response token budget.
        topic_context: meeting topic description to help the LLM understand
            the domain (e.g. "machine learning research paper presentation
            about image segmentation"). Empty string if unavailable.

    Returns:
        List of LlmProposal. The caller filters by confidence.
    """
    full_text = (transcript.get("text") or "").strip()
    if not full_text:
        return []

    model, tokenizer = model_handles or load_model()
    if model is None or tokenizer is None:
        logger.warning("No LLM backend available; Channel B skipped.")
        return []

    if not topic_context:
        topic_context = "(topic unknown — general meeting)"

    chunks = _chunk_text(full_text)[:max_chunks]
    proposals: list[LlmProposal] = []

    for idx, chunk in enumerate(chunks):
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            chunk=chunk, topic_context=topic_context,
        )
        logger.info("Channel B chunk %d/%d (%d chars)", idx + 1, len(chunks), len(chunk))
        try:
            raw = run_inference_raw(
                user_prompt, _SYSTEM_PROMPT, model, tokenizer,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("Channel B LLM call failed on chunk %d: %s", idx + 1, e)
            continue

        parsed = _parse_llm_response(raw)
        for item in parsed:
            original = str(item.get("original", "")).strip()
            target = str(item.get("target", "")).strip()
            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            reason = str(item.get("reason", "")).strip()
            if not original or not target:
                continue
            if original.lower() == target.lower():
                continue
            # Sanity: don't accept wildly different lengths.
            if len(target) > len(original) * 4:
                continue
            proposals.append(LlmProposal(
                original=original,
                target=target,
                confidence=max(0.0, min(1.0, conf)),
                reason=reason,
                chunk_index=idx,
            ))

    logger.info("Channel B: %d proposals across %d chunks", len(proposals), len(chunks))
    return proposals
