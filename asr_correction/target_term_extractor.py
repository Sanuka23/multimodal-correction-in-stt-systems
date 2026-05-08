"""Extract target terms from a ground-truth transcript for TTER evaluation.

Target terms are words whose misrecognition is semantically significant:
person names, organization / product names, technical jargon, and acronyms.
See docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.6.
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from wordfreq import zipf_frequency

from .technical_keywords import TECHNICAL_KEYWORDS

# Zipf frequency threshold for the "rare-word" target-term rule.
# Words at or below this threshold are rare enough to be considered
# domain-relevant (proper nouns, jargon, technical terms).
_RARE_WORD_ZIPF_MAX = 3.5

# Minimum length for the rare-word rule (Rule 5). Short rare words
# (< 4 chars) tend to be initialisms and are already caught by Rule 3.
_RARE_WORD_MIN_LEN = 4

# NER labels that typically correspond to target terms.
_ENTITY_LABELS = {"PERSON", "ORG", "PRODUCT", "WORK_OF_ART", "GPE"}

# Function-word acronyms to exclude from Rule 3.
_FUNCTION_WORD_CAPS = {"THE", "AND", "FOR", "YOU", "BUT", "NOT", "ALL", "ANY", "CAN"}

# Acronym regex: 2+ caps, optional digits with optional hyphen.
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:-?\d+)?\b")

# Words tagged PROPN by spaCy that are still too common to count as target terms.
_COMMON_PROPN_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "english", "american", "european",
}


@lru_cache(maxsize=1)
def _get_nlp():
    """Load spaCy model once per process."""
    return spacy.load("en_core_web_sm")


def _is_all_caps_dominant(text: str) -> bool:
    """Detect transcripts stored mostly in uppercase (e.g. SlideAVSR ground truth).

    Returns True if the uppercase-to-letter ratio exceeds 0.6. When True, the
    acronym rule is disabled (it would otherwise match every word) and the
    text is lowercased before spaCy processing.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters) > 0.6


def extract_target_terms(text: str) -> set[str]:
    """Return the set of target terms in `text`.

    Applied rules (union):
    1. spaCy NER entities with label in {PERSON, ORG, PRODUCT, WORK_OF_ART, GPE}
    2. Proper nouns (POS=PROPN) not in the common-word exclusion list
    3. Acronyms matching \\b[A-Z]{2,}(?:-?\\d+)?\\b, excluding function words.
       Skipped for all-caps dominant text where every word would match.
    4. Technical / business keywords from TECHNICAL_KEYWORDS (case-insensitive)
    """
    if not text.strip():
        return set()

    all_caps = _is_all_caps_dominant(text)
    # All-caps input (e.g. SlideAVSR ground truth) confuses spaCy's NER and
    # POS tagger; lowercase it before feeding the pipeline. Recall drops on
    # proper nouns but precision stays sane.
    processing_text = text.lower() if all_caps else text

    nlp = _get_nlp()
    doc = nlp(processing_text)
    terms: set[str] = set()

    # Rule 1: named entities
    for ent in doc.ents:
        if ent.label_ in _ENTITY_LABELS:
            terms.add(ent.text)

    # Rule 2: proper nouns not in common exclusion list
    for token in doc:
        if token.pos_ == "PROPN" and token.text.lower() not in _COMMON_PROPN_WORDS:
            terms.add(token.text)

    # Rule 3: acronyms (text-level regex, not spaCy-dependent).
    # Skip for all-caps dominant text.
    if not all_caps:
        for match in _ACRONYM_RE.findall(text):
            if match not in _FUNCTION_WORD_CAPS:
                terms.add(match)

    # Rule 4: technical keywords (case-insensitive match, preserve as-written form)
    lower_text_tokens = {t.text for t in doc if t.text.lower() in TECHNICAL_KEYWORDS}
    terms.update(lower_text_tokens)

    # Rule 5: rare-word heuristic via wordfreq. Catches domain-relevant
    # vocabulary that NER misses, especially on all-caps ground truth
    # where spaCy's entity recognizer degrades. A word counts if:
    #  - alphabetic and at least _RARE_WORD_MIN_LEN characters
    #  - wordfreq Zipf score at or below _RARE_WORD_ZIPF_MAX
    #  - not a month / day / other known non-term
    for token in doc:
        t = token.text
        if not t.isalpha() or len(t) < _RARE_WORD_MIN_LEN:
            continue
        t_low = t.lower()
        if t_low in _COMMON_PROPN_WORDS:
            continue
        if zipf_frequency(t_low, "en") <= _RARE_WORD_ZIPF_MAX:
            terms.add(t)

    return terms
