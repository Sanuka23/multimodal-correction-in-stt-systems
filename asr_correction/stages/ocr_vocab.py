"""Stage 3.5 — OCR vocabulary extraction.

LLM-driven entity mining over the raw OCR text snippets produced by
:mod:`asr_correction.ocr_extractor`. Pulls out proper nouns of type
*person*, *product*, *company*, or *tool*, which then become the
**protected terms** the reconciler is forbidden from modifying.

Public function
---------------
- :func:`extract_vocab_from_ocr` — main entry point used by the pipeline
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def extract_vocab_from_ocr(ocr_hints, model, tokenizer):
    """Stage 3.5 entry point — extract proper nouns from OCR text.

    Uses the LLM to identify high-value terms from raw screen text.
    These become protected ground truth for the reconciler.
    """
    from ..model import run_inference_raw

    if not ocr_hints or not model:
        return []

    t0 = time.time()

    # Deduplicate and evenly sample across ALL OCR lines (not just first 80)
    unique = list(dict.fromkeys(ocr_hints))
    if len(unique) > 150:
        step = max(1, len(unique) // 150)
        unique = unique[::step][:150]
    sample = "\n".join(unique)

    prompt = (
        "Extract person names, product/tool names, and company names from this screen text.\n"
        "This text was read from a video meeting screen using OCR.\n\n"
        f"Screen text:\n{sample}\n\n"
        "Rules:\n"
        "- Only extract proper nouns (names, products, companies)\n"
        "- Ignore UI text like 'Stop presenting', 'Add people', 'Search'\n"
        "- Ignore URLs, email addresses, and random characters\n"
        "- Include full names for people (e.g. 'Avindi De Silva' → extract 'Avindi')\n\n"
        'Respond with JSON array: [{"term": "Avindi", "type": "person"}, '
        '{"term": "ChartMogul", "type": "product"}, {"term": "Stripe", "type": "company"}]\n'
        "Only include terms you are confident about."
    )

    system = "You extract structured data from OCR text. Output only valid JSON."

    try:
        raw = run_inference_raw(prompt, system, model, tokenizer, max_tokens=512)

        import json as _json
        import re as _re

        # Parse JSON array from response
        match = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if match:
            parsed = _json.loads(match.group())
            if isinstance(parsed, list):
                # Validate and deduplicate
                seen = set()
                valid = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    term = item.get("term", "").strip()
                    term_type = item.get("type", "").strip().lower()
                    if not term or len(term) < 2 or term.lower() in seen:
                        continue
                    if term_type not in ("person", "product", "company", "tool"):
                        continue
                    seen.add(term.lower())
                    valid.append({"term": term, "type": term_type})

                duration_ms = (time.time() - t0) * 1000
                logger.info("============================================================")
                logger.info("Step 3.5: OCR VOCAB EXTRACTION")
                logger.info("  Extracted %d terms from OCR in %.0fms:", len(valid), duration_ms)
                for v in valid[:15]:
                    logger.info("    [%s] %s", v["type"], v["term"])
                if len(valid) > 15:
                    logger.info("    ... and %d more", len(valid) - 15)
                logger.info("============================================================")
                return valid

    except Exception as e:
        logger.warning("Step 3.5: OCR vocab extraction failed — %s", e)

    return []
