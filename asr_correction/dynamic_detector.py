"""Dynamic candidate detection for Phase 2 (Channel C — OCR cross-reference).

Produces candidate spans from a ScreenApp transcript by cross-referencing
OCR frames: any entity-like token visible on the screen that does not
appear in the nearby transcript becomes a correction candidate.

Spec: docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.1
Phase 2 scope: Channel C only. Channels A (Whisper confidence) and B (LLM
semantic check) are deferred because Whisper's probability distribution in
ScreenApp's output is too compressed for a static threshold to be useful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import jellyfish
from wordfreq import zipf_frequency

# Entity-token extraction uses a word-level tokenizer instead of regexes
# over capitalization because we need richer shape analysis (see
# _is_distinctive_entity).

# Common English words that look like entities but are not.
_COMMON_WORDS = frozenset({
    "the", "and", "for", "you", "but", "not", "all", "any", "can", "have",
    "this", "that", "with", "from", "will", "been", "what", "when", "where",
    "how", "why", "which", "who", "our", "your", "their", "these", "those",
    "about", "after", "again", "also", "as", "at", "be", "because", "before",
    "being", "by", "do", "does", "doing", "each", "few", "further", "had",
    "has", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "if", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "now", "of", "off", "on", "once", "only",
    "or", "other", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "them", "themselves", "then", "there", "they",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "while", "would", "yes",
    # Common acronyms that are not target terms
    "am", "pm", "usa", "uk",
})

# Function-word-looking all-caps tokens to exclude
_FUNCTION_WORD_CAPS = frozenset({"THE", "AND", "FOR", "YOU", "BUT", "NOT", "ALL", "ANY", "CAN"})

# Phonetic similarity threshold for "word is already in transcript" check.
# Above this similarity, we consider the OCR token to already be present
# in the transcript, so no correction is proposed.
_TRANSCRIPT_MATCH_THRESHOLD = 0.88

# Minimum similarity required to pair an OCR token with a transcript word
# as a suspected mis-transcription. Uses Levenshtein ratio (see _phonetic_sim),
# which is stricter than Jaro-Winkler and rejects unrelated words that
# coincidentally share a few characters.
_SUSPECT_MIN_SIMILARITY = 0.70

# Minimum length for both sides of a phonetic comparison. Shorter words
# produce spurious metaphone matches (e.g. "to" ↔ "Tuan") because the
# phonetic codes collapse to 1–2 characters that score high on
# Jaro-Winkler due to their common prefixes.
_MIN_WORD_LENGTH = 5

# Common-English threshold on the Zipf frequency scale (1–7). Words at or
# above this value are too common to plausibly be a mis-transcription of
# a proper noun — e.g. "very" (6.00), "image" (4.94), "tiny" (4.62),
# "noisy" (3.67). Mis-transcribed proper nouns almost always surface as
# uncommon words (zipf < 3.5) because Whisper emits unusual spellings
# when uncertain. This threshold trades a small amount of recall
# (real but common-sounding mis-transcriptions) for much higher
# precision — a requirement for the TTER evaluation.
_SUSPECT_MAX_ZIPF = 3.5

# Function words and high-frequency short tokens that should never be
# flagged as mis-transcriptions even if phonetics disagree — they are
# the kind of words ASR gets right and OCR would never show on a slide.
_SUSPECT_BLOCKLIST = frozenset({
    "the", "and", "for", "you", "but", "not", "all", "any", "can", "have",
    "this", "that", "with", "from", "will", "been", "what", "when",
    "about", "after", "again", "into", "just", "more", "most", "over",
    "some", "such", "than", "then", "they", "were", "your", "their",
    "would", "could", "should", "an", "am", "is", "it", "to", "in",
    "on", "at", "of", "or", "we", "me", "my", "he", "be", "do", "so",
    "up", "if", "us", "no", "as",
})


@dataclass
class DynamicCandidate:
    """A correction candidate surfaced by dynamic detection."""
    signal: str                    # "ocr_mismatch" | "ocr_concat"
    ocr_token: str                 # the entity token seen on screen
    ocr_timestamp: float           # when the OCR frame was captured
    nearest_transcript_word: str = ""   # the transcript word being replaced
    nearest_transcript_start: float = 0.0
    segment_id: int = -1
    whisper_prob: float = 0.0
    # For signal="ocr_concat", holds the full multi-word phrase to replace
    # (e.g. "Dense CRF" when the OCR token is "DenseCRF"). Empty for the
    # single-word ocr_mismatch case.
    multiword_phrase: str = ""


def _is_distinctive_entity(token: str) -> bool:
    """Return True if `token` has distinctive proper-noun / acronym shape.

    Acceptable shapes (for Phase 2 — OCR-only correction):
    - All-caps acronym of length 2+ (e.g. CVPR, GPU, MAGNET, CNN).
    - camelCase or PascalCase with an internal uppercase letter after a
      lowercase letter (e.g. VinAi, MagNet, GLNet, PyTorch, QueryMaintenance).
    - Contains a digit (e.g. GPT-4, Qwen3, ES2002a).

    Single-capitalized common words like "Image", "High", "Every", "State"
    are rejected. This trades recall on single-word proper nouns (author
    first names, place names) for much tighter precision on technical
    terms — which is Phase 2's target class.
    """
    if len(token) < 2:
        return False

    # Rule 1: all-caps acronym
    if token.isupper() and len(token) >= 2:
        return True

    # Rule 3: contains digit
    if any(c.isdigit() for c in token):
        return True

    # Rule 2: camelCase / PascalCase — an uppercase letter appears after
    # a lowercase letter at some internal position.
    for i in range(1, len(token)):
        if token[i].isupper() and token[i - 1].islower():
            return True

    return False


def extract_entity_tokens(ocr_text: str) -> list[str]:
    """Extract entity-like tokens from raw OCR text.

    Tokenizes the OCR text on non-word characters, then keeps only tokens
    whose shape is distinctive (see _is_distinctive_entity). Deduplicates
    while preserving first-seen order.
    """
    if not ocr_text:
        return []

    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9]*", ocr_text):
        if raw in seen:
            continue
        if raw.upper() in _FUNCTION_WORD_CAPS:
            continue
        if raw.lower() in _COMMON_WORDS:
            continue
        if not _is_distinctive_entity(raw):
            continue
        seen.add(raw)
        tokens.append(raw)
    return tokens


def _transcript_word_iter(transcript: dict) -> Iterable[dict]:
    """Yield every word dict across all segments of a ScreenApp transcript."""
    for seg in transcript.get("segments", []):
        for word in seg.get("words", []):
            yield word


def _transcript_words_in_window(
    transcript: dict,
    center_s: float,
    window_s: float = 10.0,
) -> list[dict]:
    """All transcript word entries within ±window_s of `center_s`."""
    lo, hi = center_s - window_s, center_s + window_s
    return [
        w for w in _transcript_word_iter(transcript)
        if lo <= w.get("start", 0.0) <= hi
    ]


def _levenshtein_ratio(a: str, b: str) -> float:
    """1 - edit_distance / max(len). Strict similarity."""
    if not a or not b:
        return 0.0
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    return 1.0 - jellyfish.levenshtein_distance(a, b) / m


def _phonetic_sim(a: str, b: str) -> float:
    """Levenshtein-ratio similarity, using both raw strings and Metaphone codes.

    Returns the max of:
      - Levenshtein ratio on the raw lowercased strings
      - Levenshtein ratio on the strings' Metaphone codes

    Levenshtein ratio is much stricter than Jaro-Winkler: it requires the
    words to differ by at most a handful of character edits, rather than
    merely sharing a common prefix. This rejects coincidentally-similar
    but semantically unrelated pairs like "magnet"/"semantic" while still
    catching homophone errors ("Kime"/"Kimi") and spelling variants
    ("Cloudware"/"Cloudflare").

    Short inputs (< _MIN_WORD_LENGTH characters) short-circuit to exact
    match only — Metaphone degenerates on short strings.
    """
    if not a or not b:
        return 0.0
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    if not a_low or not b_low:
        return 0.0
    if len(a_low) < _MIN_WORD_LENGTH or len(b_low) < _MIN_WORD_LENGTH:
        return 1.0 if a_low == b_low else 0.0

    direct = _levenshtein_ratio(a_low, b_low)
    code_a = jellyfish.metaphone(a_low) or a_low
    code_b = jellyfish.metaphone(b_low) or b_low
    metaphone_sim = _levenshtein_ratio(code_a, code_b)
    return max(direct, metaphone_sim)


def _is_token_in_transcript_window(
    token: str,
    transcript_window: list[dict],
    threshold: float = _TRANSCRIPT_MATCH_THRESHOLD,
) -> bool:
    """Return True if any word in the window phonetically matches `token`."""
    token_lower = token.lower()
    for w in transcript_window:
        wt = w.get("word", "").strip().lower()
        if not wt:
            continue
        if wt == token_lower:
            return True
        if _phonetic_sim(token, wt) >= threshold:
            return True
    return False


def _find_nearest_suspect_word(
    token: str,
    transcript_window: list[dict],
) -> tuple[dict, float] | None:
    """Pick the transcript word in the window most likely to be a
    mis-transcription of `token`.

    Returns (word_dict, similarity) for the best candidate, or None if no
    transcript word is phonetically close enough. A candidate must satisfy:

    - Word length >= _MIN_WORD_LENGTH (short words produce false matches).
    - Word not in _SUSPECT_BLOCKLIST (common function words).
    - Phonetic similarity to `token` >= _SUSPECT_MIN_SIMILARITY and
      < _TRANSCRIPT_MATCH_THRESHOLD (above the upper bound means the token
      is already in the transcript, so no correction is needed).

    There is NO fallback to lowest-probability words — if nothing meets
    the phonetic bar, we return None and drop the candidate entirely.
    """
    if not transcript_window:
        return None

    best: tuple[dict, float] | None = None
    best_sim = 0.0
    for w in transcript_window:
        wt = w.get("word", "").strip()
        wt_clean = wt.lower().strip(".,!?;:'\"")
        if not wt_clean:
            continue
        if len(wt_clean) < _MIN_WORD_LENGTH:
            continue
        if wt_clean in _SUSPECT_BLOCKLIST:
            continue
        # Reject common English words — they are almost never mis-transcribed
        # proper nouns. See _SUSPECT_MAX_ZIPF docstring.
        if zipf_frequency(wt_clean, "en") >= _SUSPECT_MAX_ZIPF:
            continue

        sim = _phonetic_sim(token, wt_clean)
        if sim < _SUSPECT_MIN_SIMILARITY or sim >= _TRANSCRIPT_MATCH_THRESHOLD:
            continue
        if sim > best_sim:
            best_sim = sim
            best = (w, sim)

    return best


def _find_multiword_concat_match(
    token: str,
    transcript_window: list[dict],
) -> dict | None:
    """Check whether `token` equals the concatenation of 2–3 adjacent
    transcript words (case-insensitive, punctuation-stripped).

    Returns the FIRST word of the match so the caller can record the start
    time and later replace the whole span. If no concatenation match is
    found, returns None.

    This catches the "tokenization split" class of errors where Whisper
    correctly hears a compound name like "DenseCRF" or "PointRend" but
    outputs it as two words ("Dense CRF" / "Point Rend").

    Precision guard: at least ONE of the concatenated transcript words
    must be a rare word (Zipf <= 3.5). Otherwise the match is almost
    certainly a PaddleOCR missing-space artifact like "WHICHTHE" or
    "REALITYFOR" — not a real entity that needs correction.
    """
    token_low = token.lower()
    if len(token_low) < _MIN_WORD_LENGTH:
        return None

    words = [
        (w, (w.get("word", "").strip().lower().strip(".,!?;:'\"")))
        for w in transcript_window
    ]
    words = [(w, c) for (w, c) in words if c]

    # Try 2-word and 3-word concatenations
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            parts = [c for _, c in words[i:i + n]]
            concat = "".join(parts)
            if concat != token_low:
                continue

            # Require at least one rare word in the match.
            has_rare = any(
                zipf_frequency(p, "en") <= _SUSPECT_MAX_ZIPF
                for p in parts
            )
            if not has_rare:
                continue
            return words[i][0]
    return None


# Zipf threshold for the rare-word detection channel. Words at or below
# this value are considered suspicious (possible mis-transcribed proper
# nouns or domain-specific vocabulary). The threshold is tighter than
# _SUSPECT_MAX_ZIPF because detection is unsupervised here — we don't
# have OCR to confirm the context, so we only want to investigate words
# that are genuinely rare.
_RARE_DETECTION_ZIPF_MAX = 2.0

# Minimum transcript word length for rare-word detection.
_RARE_DETECTION_MIN_LEN = 5


def detect_candidates_rare_words(
    transcript: dict,
    max_candidates: int = 40,
) -> list[DynamicCandidate]:
    """Produce candidates by scanning the transcript for CAPITALIZED rare words.

    Whisper capitalizes words it believes are proper nouns. If a word is
    capitalized AND rare (low Zipf), it is very likely a mis-transcribed
    proper noun — exactly what Phase 3 wants to investigate. Lowercase
    rare words like "destocking" or "biomorphic" are correctly
    transcribed uncommon English vocabulary and are NOT investigated.

    Sentence-initial capitals are excluded by checking the word's
    position — if it is the first word of a segment OR the previous
    word ends in sentence-terminating punctuation, the capitalization
    is not meaningful and the word is skipped.

    Criteria:
    - Original (un-normalized) token starts with an uppercase letter
    - Mid-sentence position (not sentence-initial)
    - Length >= _RARE_DETECTION_MIN_LEN
    - Lowercased form's Zipf <= _RARE_DETECTION_ZIPF_MAX
    - Not in _SUSPECT_BLOCKLIST
    """
    out: list[DynamicCandidate] = []
    seen: set[str] = set()
    _SENT_END = {".", "!", "?"}

    for seg in transcript.get("segments", []):
        prev_word_raw = ""
        for w in seg.get("words", []):
            raw = w.get("word", "").strip()
            clean = raw.lower().strip(".,!?;:'\"()[]{}")

            # Reject too-short, empty, or blocklisted words early.
            if not clean or len(clean) < _RARE_DETECTION_MIN_LEN:
                prev_word_raw = raw
                continue
            if clean in _SUSPECT_BLOCKLIST:
                prev_word_raw = raw
                continue
            if clean in seen:
                prev_word_raw = raw
                continue

            # Require the original token to be capitalized (first letter
            # uppercase). "POITRANT" and "Cloudware" pass; "destocking" fails.
            first_alpha = next((c for c in raw if c.isalpha()), "")
            if not first_alpha.isupper():
                prev_word_raw = raw
                continue

            # Reject sentence-initial capitals: if the previous token ends
            # in a sentence terminator, or this is the first word in the
            # segment, the capitalization is not meaningful.
            is_sentence_start = (
                not prev_word_raw
                or prev_word_raw.strip().endswith(tuple(_SENT_END))
            )
            if is_sentence_start:
                prev_word_raw = raw
                continue

            if zipf_frequency(clean, "en") > _RARE_DETECTION_ZIPF_MAX:
                prev_word_raw = raw
                continue

            seen.add(clean)
            out.append(DynamicCandidate(
                signal="rare_word",
                ocr_token="",
                ocr_timestamp=w.get("start", 0.0),
                nearest_transcript_word=raw,
                nearest_transcript_start=w.get("start", 0.0),
                whisper_prob=w.get("probability", 1.0),
            ))
            prev_word_raw = raw
            if len(out) >= max_candidates:
                return out
    return out


def detect_candidates_channel_c(
    transcript: dict,
    ocr_frames: list[dict],
    window_s: float = 10.0,
) -> list[DynamicCandidate]:
    """Cross-reference OCR frames against the transcript to find candidates.

    Args:
        transcript: ScreenApp transcript dict with `segments[*].words[*]`.
        ocr_frames: list of `{timestamp_s, text}` dicts from asr_correction.ocr_parser.
        window_s: temporal window around each OCR frame to search in the transcript.

    Returns:
        Deduped list of DynamicCandidate. Each has either:
        - signal="ocr_missed" — the entity token is on screen but not in the
          transcript window (the nearest low-confidence word is the suspect).
        - signal="ocr_mismatch" — the entity token phonetically matches some
          transcript word but they differ enough to be a suspected error.

    Dedup: a token that appears in multiple OCR frames is kept only once,
    linked to its first transcript suspect.
    """
    out: list[DynamicCandidate] = []
    seen_tokens: set[str] = set()

    for frame in ocr_frames:
        ts = frame.get("timestamp_s", 0.0)
        text = frame.get("text", "")
        tokens = extract_entity_tokens(text)
        if not tokens:
            continue

        window = _transcript_words_in_window(transcript, ts, window_s)
        if not window:
            continue

        for token in tokens:
            if token.lower() in seen_tokens:
                continue

            if _is_token_in_transcript_window(token, window):
                continue  # Already present — no correction needed

            # First try multi-word concatenation match — catches tokenization
            # splits like "Dense CRF" → "DenseCRF".
            concat_match = _find_multiword_concat_match(token, window)
            if concat_match is not None:
                # Reconstruct the matched phrase for replacement.
                window_lower = [
                    (w.get("word", "").strip().lower().strip(".,!?;:'\""))
                    for w in window
                ]
                start_idx = window.index(concat_match)
                # Find how many words form the match.
                for n in (2, 3):
                    if start_idx + n > len(window):
                        continue
                    concat = "".join(window_lower[start_idx:start_idx + n])
                    if concat == token.lower():
                        phrase_words = [
                            window[start_idx + k].get("word", "").strip()
                            for k in range(n)
                        ]
                        phrase = " ".join(phrase_words)
                        out.append(DynamicCandidate(
                            signal="ocr_concat",
                            ocr_token=token,
                            ocr_timestamp=ts,
                            nearest_transcript_word=phrase_words[0],
                            nearest_transcript_start=concat_match.get("start", 0.0),
                            whisper_prob=concat_match.get("probability", 1.0),
                            multiword_phrase=phrase,
                        ))
                        seen_tokens.add(token.lower())
                        break
                continue

            result = _find_nearest_suspect_word(token, window)
            if result is None:
                continue

            suspect, sim = result
            cand = DynamicCandidate(
                signal="ocr_mismatch",
                ocr_token=token,
                ocr_timestamp=ts,
                nearest_transcript_word=suspect.get("word", "").strip(),
                nearest_transcript_start=suspect.get("start", 0.0),
                whisper_prob=suspect.get("probability", 1.0),
            )
            out.append(cand)
            seen_tokens.add(token.lower())

    return out
