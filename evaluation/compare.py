"""Compare two transcripts with WER + optional TTER."""

from .wer import WERCalculator
from .tter import compute_tter


def _build_diff(wer_result) -> list:
    """Build word-level diff from aligned reference/hypothesis.

    Returns a list of dicts: {type, ref_word, hyp_word} where type is
    'equal', 'substitute', 'delete', or 'insert'.
    """
    import jiwer

    ref_text = wer_result.reference
    hyp_text = wer_result.hypothesis

    if not ref_text and not hyp_text:
        return []

    try:
        word_output = jiwer.process_words(ref_text, hyp_text)
    except Exception:
        return []

    diff = []
    for chunk in word_output.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                diff.append({
                    "type": "equal",
                    "ref_word": word_output.references[0][chunk.ref_start_idx + i],
                    "hyp_word": word_output.hypotheses[0][chunk.hyp_start_idx + i],
                })
        elif chunk.type == "substitute":
            ref_words = [word_output.references[0][chunk.ref_start_idx + i]
                         for i in range(chunk.ref_end_idx - chunk.ref_start_idx)]
            hyp_words = [word_output.hypotheses[0][chunk.hyp_start_idx + i]
                         for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx)]
            max_len = max(len(ref_words), len(hyp_words))
            for i in range(max_len):
                diff.append({
                    "type": "substitute",
                    "ref_word": ref_words[i] if i < len(ref_words) else "",
                    "hyp_word": hyp_words[i] if i < len(hyp_words) else "",
                })
        elif chunk.type == "delete":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                diff.append({
                    "type": "delete",
                    "ref_word": word_output.references[0][chunk.ref_start_idx + i],
                    "hyp_word": "",
                })
        elif chunk.type == "insert":
            for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                diff.append({
                    "type": "insert",
                    "ref_word": "",
                    "hyp_word": word_output.hypotheses[0][chunk.hyp_start_idx + i],
                })

    return diff


def compare_transcripts(reference: str, hypothesis: str, target_terms: list = None) -> dict:
    """Run full comparison between reference and hypothesis transcripts.

    Args:
        reference: Ground truth text.
        hypothesis: ASR output text.
        target_terms: Optional list of target term dicts for TTER computation.

    Returns:
        Dict with WER, CER, diff, and optionally TTER results.
    """
    calculator = WERCalculator(normalize=True)
    wer_result = calculator.compute(reference, hypothesis)

    result = {
        "wer": round(wer_result.wer * 100, 2),
        "cer": round(wer_result.cer * 100, 2),
        "substitutions": wer_result.substitutions,
        "insertions": wer_result.insertions,
        "deletions": wer_result.deletions,
        "hits": wer_result.hits,
        "ref_word_count": wer_result.ref_word_count,
        "hyp_word_count": wer_result.hyp_word_count,
        "diff": _build_diff(wer_result),
    }

    if target_terms:
        tter_result = compute_tter(reference, hypothesis, target_terms)
        result["tter"] = tter_result

    return result
