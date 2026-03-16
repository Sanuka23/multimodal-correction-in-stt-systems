"""Target Term Error Rate (TTER) computation.

TTER = (target term errors) / (total target term occurrences)
"""

import re


def normalize(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def find_occurrences(text: str, term: str) -> list:
    """Find all occurrences of a term in text, return list of (start, end) char positions."""
    norm_text = normalize(text)
    norm_term = normalize(term)
    positions = []
    start = 0
    while True:
        idx = norm_text.find(norm_term, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(norm_term)))
        start = idx + 1
    return positions


def get_context(text: str, pos: int, window: int = 60) -> str:
    """Extract surrounding context from text at character position."""
    norm = normalize(text)
    start = max(0, pos - window)
    end = min(len(norm), pos + window)
    return "..." + norm[start:end] + "..."


def check_term_in_hypothesis(ref_text: str, hyp_text: str, term: str,
                              known_errors: list, ref_positions: list) -> list:
    """Check each reference occurrence against hypothesis transcript."""
    norm_ref = normalize(ref_text)
    norm_hyp = normalize(hyp_text)
    norm_term = normalize(term)

    ref_len = len(norm_ref)
    hyp_len = len(norm_hyp)

    results = []
    for ref_start, ref_end in ref_positions:
        ratio = hyp_len / max(ref_len, 1)
        hyp_approx_start = max(0, int(ref_start * ratio) - 120)
        hyp_approx_end = min(hyp_len, int(ref_end * ratio) + 120)
        hyp_region = norm_hyp[hyp_approx_start:hyp_approx_end]

        ref_context = get_context(ref_text, ref_start, 50)

        if norm_term in hyp_region:
            results.append({
                "term": term,
                "status": "correct",
                "ref_context": ref_context,
                "hyp_context": "..." + hyp_region[:120] + "...",
                "found_as": term,
            })
        else:
            found_error = None
            for err in known_errors:
                if normalize(err) in hyp_region:
                    found_error = err
                    break

            if found_error:
                results.append({
                    "term": term,
                    "status": "error",
                    "error_type": "known_substitution",
                    "ref_context": ref_context,
                    "hyp_context": "..." + hyp_region[:120] + "...",
                    "found_as": found_error,
                })
            else:
                results.append({
                    "term": term,
                    "status": "error",
                    "error_type": "missing_or_unknown",
                    "ref_context": ref_context,
                    "hyp_context": "..." + hyp_region[:120] + "...",
                    "found_as": None,
                })

    return results


def compute_tter(ref_text: str, hyp_text: str, terms: list) -> dict:
    """Compute TTER between reference and hypothesis text.

    Args:
        ref_text: Reference (ground truth) transcript text.
        hyp_text: Hypothesis (ASR output) transcript text.
        terms: List of dicts with 'term', 'category', and optional 'known_errors'.

    Returns:
        Dict with overall_tter, total_occurrences, total_errors, term_results.
    """
    term_summaries = []

    for term_info in terms:
        term = term_info["term"]
        category = term_info.get("category", "unknown")
        known_errors = term_info.get("known_errors", [])

        ref_positions = find_occurrences(ref_text, term)
        total_occurrences = len(ref_positions)

        if total_occurrences == 0:
            continue

        results = check_term_in_hypothesis(ref_text, hyp_text, term, known_errors, ref_positions)

        correct = sum(1 for r in results if r["status"] == "correct")
        errors = sum(1 for r in results if r["status"] == "error")

        term_tter = errors / total_occurrences * 100 if total_occurrences > 0 else 0

        term_summaries.append({
            "term": term,
            "category": category,
            "occurrences": total_occurrences,
            "correct": correct,
            "errors": errors,
            "tter": round(term_tter, 1),
            "error_details": [r for r in results if r["status"] == "error"],
        })

    total_occ = sum(t["occurrences"] for t in term_summaries)
    total_err = sum(t["errors"] for t in term_summaries)
    total_correct = sum(t["correct"] for t in term_summaries)
    overall_tter = total_err / total_occ * 100 if total_occ > 0 else 0

    return {
        "overall_tter": round(overall_tter, 2),
        "total_occurrences": total_occ,
        "total_correct": total_correct,
        "total_errors": total_err,
        "terms_tracked": len(term_summaries),
        "term_results": sorted(term_summaries, key=lambda x: x["tter"], reverse=True),
    }
