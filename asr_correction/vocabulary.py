"""Vocabulary management for ASR correction.

Merges ScreenApp's per-team customVocabulary with the domain vocabulary
shipped in domain_vocab.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_domain_vocab(path: Optional[str] = None) -> dict:
    """Load the domain vocabulary file.

    Falls back to domain_vocab.json in the module directory.
    """
    if path is None:
        path = str(Path(__file__).parent / "domain_vocab.json")
    with open(path) as f:
        return json.load(f)


def merge_vocabularies(
    custom_vocab: Optional[list],
    domain_vocab: dict,
) -> List[dict]:
    """Merge ScreenApp customVocabulary with domain vocab.

    Args:
        custom_vocab: From ScreenApp team.fileOptions.customVocabulary.
            Can be a list of dicts with term/category/known_errors,
            or a plain list of strings.
        domain_vocab: From domain_vocab.json with "terms" list.

    Returns:
        Unified list of term dicts:
        [{"term": str, "category": str, "known_errors": [str]}]
    """
    terms: Dict[str, dict] = {}

    # Load domain vocab terms
    for term_info in domain_vocab.get("terms", []):
        key = term_info["term"].lower()
        if key not in terms:
            terms[key] = {
                "term": term_info["term"],
                "category": term_info.get("category", "unknown"),
                "known_errors": list(term_info.get("known_errors", [])),
            }
        else:
            # Merge known_errors from duplicate entries
            existing = set(terms[key]["known_errors"])
            for err in term_info.get("known_errors", []):
                if err not in existing:
                    terms[key]["known_errors"].append(err)

    # Overlay custom vocabulary (higher priority)
    if custom_vocab:
        for entry in custom_vocab:
            if isinstance(entry, str):
                entry = {"term": entry, "category": "custom"}
            key = entry.get("term", "").lower()
            if not key:
                continue
            if key in terms:
                if entry.get("category"):
                    terms[key]["category"] = entry["category"]
                existing = set(terms[key]["known_errors"])
                for err in entry.get("known_errors", []):
                    if err not in existing:
                        terms[key]["known_errors"].append(err)
            else:
                terms[key] = {
                    "term": entry["term"],
                    "category": entry.get("category", "custom"),
                    "known_errors": list(entry.get("known_errors", [])),
                }

    return list(terms.values())
