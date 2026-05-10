"""Stage 2.6 — Web vocabulary enrichment.

Searches DuckDuckGo with the meeting's classified topic, extracts domain
glossary terms via the LLM, and runs a quality gate before returning.

Public functions
----------------
- :func:`enrich_vocab_from_web` — main entry point used by the pipeline
- :func:`parse_web_vocab_json` — tolerant JSON parser (also used by tests)
- :func:`truncate_at_sentence` — sentence-boundary truncation helper
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Categories the LLM is allowed to assign to extracted terms.
ALLOWED_WEB_VOCAB_CATEGORIES = {
    "product_name", "tech_term", "person_name",
    "company_name", "domain_term", "tech_acronym",
}


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text on the last sentence boundary at or before max_chars.

    Avoids the JSON-confusion problem where mid-sentence truncation feeds the
    LLM a half-thought that it tries to "complete" inside the JSON array.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer the last sentence-end punctuation; fall back to the last whitespace.
    for delim in (". ", "! ", "? ", "; ", "\n"):
        idx = cut.rfind(delim)
        if idx >= max_chars * 0.6:  # don't lose more than ~40 % of context
            return cut[: idx + 1]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0 else cut


def parse_web_vocab_json(raw: str):
    """Best-effort JSON-array extraction. Returns list[dict] or None."""
    import json as _json
    import re as _re

    if not raw:
        return None
    # Strip optional Markdown fences
    raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=_re.MULTILINE)

    # Prefer a fully-bracketed array if present.
    match = _re.search(r"\[.*\]", raw, _re.DOTALL)
    snippet = match.group() if match else None

    if snippet is None:
        # Truncated / unclosed array — find the first '[' and try to seal it
        # at the last complete '}' we can see.
        lb = raw.find("[")
        rb = raw.rfind("}")
        if lb >= 0 and rb > lb:
            snippet = raw[lb : rb + 1] + "]"

    if snippet is None:
        return None

    try:
        parsed = _json.loads(snippet)
        return parsed if isinstance(parsed, list) else None
    except _json.JSONDecodeError:
        # Tolerant repair: trim to the last complete object inside the snippet.
        last_obj = snippet.rfind("}")
        if last_obj > 0:
            try:
                return _json.loads(snippet[: last_obj + 1] + "]")
            except _json.JSONDecodeError:
                return None
    return None


def enrich_vocab_from_web(topic_info, model, tokenizer):
    """Stage 2.6 entry point — DuckDuckGo + LLM extraction + quality gate.

    Honours `ASR_WEB_VOCAB_ENABLED=false` env kill switch (demo safety).
    Sentence-boundary truncation of snippets (cleaner LLM input).
    One self-repair retry when the LLM returns malformed JSON.
    Quality gate: drops obviously bad terms before returning.
    Structured single-line summary log for observability.
    """
    import os
    import time as _time
    from ..model import run_inference_raw

    if os.environ.get("ASR_WEB_VOCAB_ENABLED", "true").lower() == "false":
        logger.info("Step 2.6: Disabled via ASR_WEB_VOCAB_ENABLED=false")
        return []

    description = topic_info.get("description", "") if topic_info else ""
    field = topic_info.get("field", "") if topic_info else ""
    topic = topic_info.get("topic", "") if topic_info else ""

    if not description or model is None or tokenizer is None:
        logger.info("Step 2.6: Skipped — no description or no model")
        logger.info("WEB_VOCAB summary status=skipped reason=no_description_or_model")
        return []

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info("Step 2.6: WEB VOCAB ENRICHMENT")

    # ── Search ──
    snippets = []
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("  ddgs not installed — run `pip install ddgs>=4.0`")
        logger.info("WEB_VOCAB summary status=skipped reason=ddgs_missing")
        logger.info("=" * 60)
        return []

    queries = [
        f"{field} {topic} terminology glossary",
        f"{description[:100]} technical terms",
    ]
    for q in queries:
        logger.info("  Searching: %s", q[:80])
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=5))
            for r in results:
                body = r.get("body", "")
                if body:
                    snippets.append(body)
        except Exception as e:
            logger.warning("  Search failed: %s", e)

    if not snippets:
        logger.info("  No search results found")
        logger.info("WEB_VOCAB summary status=empty queries=%d", len(queries))
        logger.info("=" * 60)
        return []

    web_context = truncate_at_sentence(" ".join(snippets), 2500)
    logger.info("  Got %d chars of web context from %d snippets",
                len(web_context), len(snippets))

    # ── Extract via LLM (with one repair retry) ──
    extract_prompt = (
        f"This meeting is about: {description}\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n\n"
        f"Web search context about this domain:\n{web_context}\n\n"
        "List domain-specific terms that ASR (speech-to-text) commonly gets wrong.\n\n"
        "Return ONLY a compact JSON array on a single line:\n"
        '[{"term":"Groq","category":"company_name"},{"term":"Kubernetes","category":"tech_term"},...]\n\n'
        "Categories: product_name, tech_term, person_name, company_name, domain_term, tech_acronym\n"
        "Give 10-20 terms. Compact single-line JSON, no newlines."
    )
    system_prompt = (
        "You extract domain-specific vocabulary. Return only a compact JSON array."
    )

    raw_response = ""
    parsed = None
    try:
        raw_response = run_inference_raw(
            extract_prompt, system_prompt, model, tokenizer, max_tokens=1024,
        )
        parsed = parse_web_vocab_json(raw_response)
    except Exception as e:
        logger.warning("  First extraction call failed: %s", e)

    # One repair retry — feed the malformed output back and ask for valid JSON.
    if parsed is None and raw_response:
        logger.info("  Retrying with self-repair…")
        repair_prompt = (
            "The previous response was not valid JSON. Return the SAME content "
            "as a single-line, well-formed JSON array of "
            '{"term":"…","category":"…"} objects, nothing else.\n\n'
            "Previous response:\n" + raw_response[:1500]
        )
        try:
            repaired = run_inference_raw(
                repair_prompt, system_prompt, model, tokenizer, max_tokens=1024,
            )
            parsed = parse_web_vocab_json(repaired)
        except Exception as e:
            logger.warning("  Repair call failed: %s", e)

    if parsed is None:
        logger.warning("  Could not parse web-vocab JSON after retry")
        logger.info("  Raw (truncated): %s", (raw_response or "")[:240])
        logger.info("WEB_VOCAB summary status=parse_failed snippets=%d", len(snippets))
        logger.info("=" * 60)
        return []

    # ── Quality gate ──
    accepted, rejected = [], []
    seen = set()
    for v in parsed:
        if not isinstance(v, dict):
            continue
        term = (v.get("term") or "").strip()
        category = (v.get("category") or "").strip().lower()
        if not term:
            rejected.append(("empty_term", v))
            continue
        if len(term) < 3 or len(term) > 40:
            rejected.append(("bad_length", term))
            continue
        if term.lower() in seen:
            rejected.append(("duplicate", term))
            continue
        if category and category not in ALLOWED_WEB_VOCAB_CATEGORIES:
            rejected.append(("bad_category", term))
            continue
        seen.add(term.lower())
        accepted.append({"term": term, "category": category or "domain_term"})

    elapsed_ms = int((_time.time() - t0) * 1000)
    logger.info("  Accepted %d / %d terms in %dms", len(accepted), len(parsed), elapsed_ms)
    for v in accepted[:15]:
        logger.info("    [%s] %s", v["category"], v["term"])
    if len(accepted) > 15:
        logger.info("    ... and %d more", len(accepted) - 15)
    if rejected:
        logger.info("  Rejected %d (sample): %s", len(rejected), rejected[:5])
    logger.info(
        "WEB_VOCAB summary status=ok accepted=%d rejected=%d snippets=%d latency_ms=%d",
        len(accepted), len(rejected), len(snippets), elapsed_ms,
    )
    logger.info("=" * 60)
    return accepted
