"""Stage 2.7 — Per-candidate validation.

Two-pass validation that refines correction targets after Stage 2 detection:

Pass 1 — Cross-chunk target pooling (free, no web calls):
  The LLM detector processes the transcript in chunks and may propose
  DIFFERENT targets for the SAME entity across chunks (e.g.
  "Post-Sog → Post-SOC" in chunk 5 but "Post-Talk → Post-Hog" in chunk 6).
  Pass 1 collects all proposed targets, then for each candidate uses
  Jaro-Winkler similarity to pick the best phonetic match in the pool.

Pass 2 — Web search fallback (for remaining unresolved candidates):
  For up to 8 candidates still not in vocab, DuckDuckGo searches the
  error word + topic + "product software", regex-extracts PascalCase
  entities from results, overrides if similarity > 0.75 AND beats the
  original proposal.

Public function
---------------
- :func:`validate_candidates_via_web` — main entry point used by the pipeline
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_candidates_via_web(candidates, vocab_terms, topic_info):
    """Stage 2.7 entry point. Mutates `candidates` in place."""
    import jellyfish

    if not candidates:
        return

    vocab_set = {t["term"].lower() for t in vocab_terms}
    topic = topic_info.get("topic", "") if topic_info else ""
    field = topic_info.get("field", "") if topic_info else ""

    logger.info("=" * 60)
    logger.info("Step 2.7: CANDIDATE VALIDATION")

    # ── Pass 1: Cross-chunk target pooling ──
    # Collect all unique proposed targets across all candidates.
    target_pool = {}  # lowercased → original form
    for c in candidates:
        t = c.term.strip()
        if t and t.lower() not in vocab_set:
            target_pool[t.lower()] = t

    overrides = 0
    for c in candidates:
        error_word = c.error_found.strip()
        proposed = c.term.strip()
        if not error_word or error_word.lower() == proposed.lower():
            continue
        if proposed.lower() in vocab_set:
            continue  # Already in vocab — no need to validate

        error_clean = error_word.lower().replace("-", "").replace("'", "")
        proposed_clean = proposed.lower().replace("-", "").replace("'", "")
        proposed_sim = jellyfish.jaro_winkler_similarity(error_clean, proposed_clean)

        # Check if any OTHER target in the pool is a better match
        best_alt = None
        best_alt_sim = 0.0
        for pool_low, pool_orig in target_pool.items():
            if pool_low == proposed.lower():
                continue  # Same as current proposal
            if pool_low == error_clean:
                continue  # Same as error word
            pool_clean = pool_low.replace("-", "").replace("'", "")
            sim = jellyfish.jaro_winkler_similarity(error_clean, pool_clean)
            if sim > best_alt_sim and sim > 0.75:
                best_alt_sim = sim
                best_alt = pool_orig

        if best_alt and best_alt_sim > proposed_sim:
            logger.info("  CROSS-CHUNK OVERRIDE: '%s' → '%s' (was '%s', sim=%.2f vs %.2f)",
                        error_word, best_alt, proposed, best_alt_sim, proposed_sim)
            c.term = best_alt
            overrides += 1

    if overrides:
        logger.info("  Pass 1: %d candidates overridden from cross-chunk target pool", overrides)

    # ── Pass 2: Web search for remaining unresolved ──
    needs_web = []
    for c in candidates:
        if (c.term.lower() not in vocab_set
                and c.error_found.lower() != c.term.lower()
                and c.term.lower() not in target_pool):
            needs_web.append(c)

    if needs_web:
        needs_web = needs_web[:8]  # Cap at 8 web searches
        logger.info("  Pass 2: Web search for %d unresolved candidates", len(needs_web))
        web_overrides = 0
        try:
            from ddgs import DDGS
            import re
            for c in needs_web:
                error_word = c.error_found.strip()
                proposed = c.term.strip()
                error_nohyph = error_word.replace("-", " ")
                query = f'{error_nohyph} {field} {topic} product software'
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=3))
                except Exception:
                    continue
                # Extract PascalCase/camelCase entity words from results
                cw_set = set()
                for r in results:
                    for w in re.findall(r'\b[A-Z][a-zA-Z0-9]+\b',
                                        f"{r.get('title','')} {r.get('body','')}"):
                        if len(w) >= 4:
                            cw_set.add(w)
                error_clean = error_word.lower().replace("-", "")
                proposed_sim = jellyfish.jaro_winkler_similarity(
                    error_clean, proposed.lower().replace("-", ""))
                best_w, best_s = None, 0.0
                for cw in cw_set:
                    s = jellyfish.jaro_winkler_similarity(error_clean, cw.lower().replace("-", ""))
                    if s > best_s and s > 0.75 and cw.lower() != error_clean:
                        best_s, best_w = s, cw
                if best_w and best_s > proposed_sim:
                    logger.info("  WEB OVERRIDE: '%s' → '%s' (was '%s', sim=%.2f)",
                                error_word, best_w, proposed, best_s)
                    c.term = best_w
                    web_overrides += 1
            logger.info("  Pass 2: %d web overrides", web_overrides)
        except ImportError:
            logger.warning("  ddgs not installed — skipping web search")

    logger.info("=" * 60)
