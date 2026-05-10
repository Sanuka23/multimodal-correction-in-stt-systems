"""Stage 2.5 — Meeting topic classification.

Asks the LLM to classify the meeting's field, topic, and produce a
1-2 sentence description plus a 10-20 term suggested vocabulary list.

The output is consumed downstream by:
- Web vocab enrichment (uses field + topic + description as search seeds)
- Whisper Pass 2 (suggested_vocab feeds the initial_prompt)
- Reconciler (meeting context line at the top of the prompt)

Public function
---------------
- :func:`classify_topic` — main entry point used by the pipeline
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify_topic(transcript, vocab_terms, candidates, model, tokenizer):
    """Classify the meeting topic/field using the LLM.

    Returns a dict with keys: field, topic, description, suggested_vocab.
    Falls back to {"field":"unknown", ...} if the model is unavailable
    or the JSON parse fails.
    """
    import json as _json
    import re as _re
    from ..model import run_inference_raw

    topic_info = {"field": "unknown", "topic": "unknown", "description": "", "suggested_vocab": []}

    if model is None or tokenizer is None:
        logger.warning("[topic] No model available — skipping topic classification")
        return topic_info

    # Build transcript text
    transcript_text = transcript.get("text", "")
    if not transcript_text:
        segments = transcript.get("segments", [])
        transcript_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", ""))

    vocab_list = [t["term"] for t in vocab_terms[:30]]
    error_summary = ", ".join(
        f"'{c.error_found}'→'{c.term}'" for c in candidates[:15]
    )

    classify_prompt = (
        "Analyze this meeting transcript and classify it.\n\n"
        f"Transcript: {transcript_text[:3000]}\n\n"
    )
    if vocab_list:
        classify_prompt += f"Vocabulary terms: {_json.dumps(vocab_list)}\n\n"
    if error_summary:
        classify_prompt += f"ASR errors detected: {error_summary}\n\n"

    classify_prompt += (
        "Respond with JSON:\n"
        "{\n"
        '  "field": "broad field (tech, business, healthcare, education, finance, entertainment, etc.)",\n'
        '  "topic": "specific topic (e.g. AWS cloud deployment, marketing campaign planning)",\n'
        '  "description": "1-2 sentence summary of what this meeting discusses",\n'
        '  "suggested_vocab": ["term1", "term2", ...]\n'
        "}\n"
        "The suggested_vocab should be 10-20 domain-specific words likely discussed in this type of meeting."
    )

    system_prompt = "You are a meeting transcript analyzer. Classify the meeting by field and topic, and suggest domain-specific vocabulary."

    try:
        raw_response = run_inference_raw(
            classify_prompt, system_prompt, model, tokenizer, max_tokens=512,
        )
        json_match = _re.search(r"\{.*\}", raw_response, _re.DOTALL)
        if json_match:
            parsed = _json.loads(json_match.group())
            topic_info = {
                "field": parsed.get("field", "unknown"),
                "topic": parsed.get("topic", "unknown"),
                "description": parsed.get("description", ""),
                "suggested_vocab": parsed.get("suggested_vocab", []),
            }
    except Exception as e:
        logger.warning("[topic] Topic classification failed: %s", e)

    logger.info("=" * 60)
    logger.info("Step 2.5: TOPIC CLASSIFICATION")
    logger.info("  Field:       %s", topic_info["field"])
    logger.info("  Topic:       %s", topic_info["topic"])
    logger.info("  Description: %s", topic_info["description"])
    logger.info("  Suggested vocab: %s", topic_info["suggested_vocab"])
    logger.info("=" * 60)

    return topic_info
