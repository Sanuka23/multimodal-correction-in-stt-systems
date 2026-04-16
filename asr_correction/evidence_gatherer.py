"""Evidence gathering for dynamic correction candidates.

Phase 2: Tier 1 — OCR direct match (candidate already carries OCR token).
Phase 3: Tier 2 — Wikipedia entity lookup for rare-word candidates.

Spec: docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jellyfish

from .dynamic_detector import DynamicCandidate
from .wikipedia_client import opensearch as wiki_opensearch

EvidenceTier = Literal["hard", "llm_only"]
EvidenceSource = Literal["ocr", "wikipedia", "web", "none"]

# Source priors for the hard-evidence confidence formula (spec §4.5).
_SOURCE_PRIOR = {
    "ocr": 1.00,
    "wikipedia": 0.85,
    "web": 0.70,
}


@dataclass
class Evidence:
    """Result of evidence gathering for one candidate."""
    tier: EvidenceTier
    source: EvidenceSource
    target: str | None        # the replacement word (None for llm_only)
    confidence: float         # 0.0 – 1.0
    phonetic_similarity: float
    topic_relevance: float
    evidence_snippet: str     # human-readable for auditability


def _phonetic_similarity(candidate_word: str, target: str) -> float:
    """Combined spelling + phonetic similarity.

    Delegates to the detector's _phonetic_sim so that the same length
    guards and metaphone/direct blending apply consistently across the
    detection and evidence stages.
    """
    from .dynamic_detector import _phonetic_sim
    return _phonetic_sim(candidate_word, target)


def _hard_evidence_confidence(
    source: EvidenceSource,
    phonetic_sim: float,
    topic_relevance: float,
) -> float:
    """Spec §4.5 hard-evidence formula:
        conf = 0.5*source_prior + 0.3*phonetic_sim + 0.2*topic_relevance
    """
    prior = _SOURCE_PRIOR.get(source, 0.0)
    return 0.5 * prior + 0.3 * phonetic_sim + 0.2 * topic_relevance


def gather_evidence_ocr(
    candidate: DynamicCandidate,
    phonetic_threshold: float = 0.78,
) -> Evidence:
    """Tier 1 evidence: use the OCR token directly as the target.

    The candidate already carries the OCR token and the suspected transcript
    word. The confidence is computed from the phonetic similarity between
    them plus a fixed topic relevance of 1.0 (the OCR text IS the topic
    context — it is literally on screen at the relevant moment).

    For candidates with signal="ocr_concat" (multi-word concatenation match,
    e.g. "Dense CRF" → "DenseCRF"), confidence is always 1.0 because the
    match is an exact character-for-character concatenation. These cases
    bypass the phonetic threshold entirely.

    If the phonetic similarity is below `phonetic_threshold`, the candidate
    is returned as `tier="llm_only"` with `target=None` so the reconciliation
    stage can decide whether to trust its own reasoning. In Phase 2 this
    LLM-only path is not wired in and such candidates are effectively dropped.
    """
    # Exact concatenation match — unambiguous, skip phonetic gate.
    if candidate.signal == "ocr_concat":
        return Evidence(
            tier="hard",
            source="ocr",
            target=candidate.ocr_token,
            confidence=1.0,
            phonetic_similarity=1.0,
            topic_relevance=1.0,
            evidence_snippet=(
                f"OCR concatenation match at {candidate.ocr_timestamp:.0f}s: "
                f"'{candidate.multiword_phrase}' → '{candidate.ocr_token}'"
            ),
        )

    ocr = candidate.ocr_token
    suspect = candidate.nearest_transcript_word

    sim = _phonetic_similarity(suspect, ocr)

    if sim < phonetic_threshold:
        return Evidence(
            tier="llm_only",
            source="none",
            target=None,
            confidence=0.0,
            phonetic_similarity=sim,
            topic_relevance=0.0,
            evidence_snippet=f"phonetic similarity too low ({sim:.2f}) for OCR match",
        )

    # The topic relevance gate is trivially 1.0 here: the OCR token is
    # temporally aligned to the transcript span, which is exactly the
    # topic context we would otherwise try to verify.
    topic_relevance = 1.0
    conf = _hard_evidence_confidence("ocr", sim, topic_relevance)

    return Evidence(
        tier="hard",
        source="ocr",
        target=ocr,
        confidence=conf,
        phonetic_similarity=sim,
        topic_relevance=topic_relevance,
        evidence_snippet=(
            f"OCR match at {candidate.ocr_timestamp:.0f}s: "
            f"'{suspect}' ~ '{ocr}' (phonetic {sim:.2f})"
        ),
    )


def gather_evidence_wikipedia(
    candidate: DynamicCandidate,
    phonetic_threshold: float = 0.82,
    topic_keywords: set[str] | None = None,
) -> Evidence:
    """Tier 2 evidence: look up the suspect word in Wikipedia.

    Uses the MediaWiki opensearch API, which does fuzzy matching on
    article titles. For each returned title, compute phonetic similarity
    to the suspect and pick the best match above `phonetic_threshold`.

    If `topic_keywords` are provided (set of lowercased topic words from
    the meeting's topic classification), the topic_relevance factor is
    boosted to 1.0 when the Wikipedia title overlaps with the topic, or
    reduced to 0.5 when it doesn't. This reduces false positives from
    unrelated Wikipedia matches.

    Returns an Evidence object with:
    - `tier="hard"` if a title passes the phonetic bar and confidence >= threshold.
    - `tier="llm_only"` otherwise.
    """
    suspect = candidate.nearest_transcript_word.strip(".,!?;:'\"()[]{}")
    if not suspect or len(suspect) < 5:
        return Evidence(
            tier="llm_only", source="none", target=None,
            confidence=0.0, phonetic_similarity=0.0, topic_relevance=0.0,
            evidence_snippet="suspect too short for Wikipedia lookup",
        )

    titles = wiki_opensearch(suspect, limit=5)
    if not titles:
        return Evidence(
            tier="llm_only", source="none", target=None,
            confidence=0.0, phonetic_similarity=0.0, topic_relevance=0.0,
            evidence_snippet=f"no Wikipedia results for '{suspect}'",
        )

    # Only accept single-word Wikipedia titles. Multi-word titles are
    # overwhelmingly noise — Wikipedia's opensearch returns things like
    # "Transferable belief model" or "Hartley Shawcross" that match the
    # first letters of the query but are semantically unrelated. Real
    # technical/business mis-transcriptions almost always resolve to
    # single-token entity names (Cloudflare, Kubernetes, PointRend).
    best_title: str | None = None
    best_sim = 0.0
    for title in titles:
        if not title or " " in title.strip():
            continue
        primary = title.strip()
        sim = _phonetic_similarity(suspect, primary)
        if sim > best_sim:
            best_sim = sim
            best_title = primary

    if best_title is None or best_sim < phonetic_threshold:
        return Evidence(
            tier="llm_only", source="none", target=None,
            confidence=0.0, phonetic_similarity=best_sim, topic_relevance=0.0,
            evidence_snippet=(
                f"no Wikipedia match above threshold for '{suspect}' "
                f"(best={best_title!r} sim={best_sim:.2f})"
            ),
        )

    # Reject self-matches: if the best title is essentially the same
    # word as the suspect (exact match or high Levenshtein ratio on the
    # raw strings), there is nothing to correct.
    title_first = best_title.split()[0].lower().strip(".,!?;:'\"()[]{}")
    suspect_low = suspect.lower()
    if title_first == suspect_low:
        return Evidence(
            tier="llm_only", source="none", target=None,
            confidence=0.0, phonetic_similarity=best_sim, topic_relevance=0.0,
            evidence_snippet=f"Wikipedia best match is the suspect itself",
        )
    # Also reject when the Wikipedia title is a prefix/suffix of the suspect
    # or vice versa — most "destocking → Defrocking" noise falls into this
    # bucket because the first 3–4 letters match.
    if (title_first.startswith(suspect_low[:4])
            and not title_first.startswith(suspect_low[:6])):
        return Evidence(
            tier="llm_only", source="none", target=None,
            confidence=0.0, phonetic_similarity=best_sim, topic_relevance=0.0,
            evidence_snippet=(
                f"Wikipedia prefix-only overlap rejected: "
                f"'{suspect}' vs '{best_title}'"
            ),
        )

    # Compute topic relevance: if we have topic keywords, check if the
    # Wikipedia title overlaps with the meeting topic.
    if topic_keywords:
        title_words = {w.lower() for w in best_title.split() if len(w) > 2}
        overlap = title_words & topic_keywords
        topic_relevance = 1.0 if overlap else 0.5
    else:
        topic_relevance = 0.8

    conf = _hard_evidence_confidence("wikipedia", best_sim, topic_relevance)
    return Evidence(
        tier="hard",
        source="wikipedia",
        target=best_title,
        confidence=conf,
        phonetic_similarity=best_sim,
        topic_relevance=topic_relevance,
        evidence_snippet=(
            f"Wikipedia: '{suspect}' → '{best_title}' (phonetic {best_sim:.2f}, "
            f"topic_rel={topic_relevance:.1f})"
        ),
    )
