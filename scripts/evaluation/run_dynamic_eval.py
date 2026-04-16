#!/usr/bin/env python3
"""Phase 2 + Phase 3 — dynamic-track TTER evaluation.

Runs on all three datasets (SlideAVSR, AMI v2, Earnings-22) with:
  - Phase 2 detection: Channel C (OCR cross-reference) — only fires on
    SlideAVSR where we have cached OCR.
  - Phase 3 detection: rare-word transcript scan — fires on every dataset.
  - Phase 2 evidence: Tier 1 (OCR direct match).
  - Phase 3 evidence: Tier 2 (Wikipedia opensearch).

Writes per-file CSV + summary JSON to data/eval_results_v3/.

Run with the conda Python:
    /opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_dynamic_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HARD_EVIDENCE_THRESHOLD = 0.70
LLM_CONFIDENCE_THRESHOLD = 0.90
LLM_PHONETIC_GATE = 0.70

OUT_DIR = PROJECT_ROOT / "data/eval_results_v3"
_PUNCT_STRIP = ".,!?;:'\"()[]{}"


@dataclass
class FileResult:
    dataset: str
    file_id: str
    gt_words: int
    sa_words: int
    target_term_count: int
    target_total_in_ref: int
    baseline_wer: float
    baseline_tter: float
    corrected_wer: float
    corrected_tter: float
    candidates_ocr: int
    candidates_rare: int
    candidates_llm: int
    corrections_applied: int
    corrections_skipped_low_conf: int
    applied_details: list[dict]


def _normalize_for_metric(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_corrections_to_text(raw_text: str, corrections: list[dict]) -> str:
    """Apply string-level substitutions to the transcript text.

    Handles both single-word substitutions and multi-word concatenation
    phrases (signal="ocr_concat"). Strips trailing punctuation from the
    original to avoid \\b word-boundary failures.
    """
    out = raw_text
    for c in corrections:
        orig_clean = c["original"].strip(_PUNCT_STRIP)
        if not orig_clean:
            continue
        if " " in orig_clean:
            parts = [re.escape(p) for p in orig_clean.split()]
            pattern = r"\b" + r"\s+".join(parts) + r"\b"
        else:
            pattern = r"\b" + re.escape(orig_clean) + r"\b"
        out = re.sub(pattern, c["target"], out, flags=re.IGNORECASE)
    return out


def _load_ocr_frames_slideavsr(file_id: str) -> list[dict]:
    from asr_correction.ocr_parser import parse_ocr_xml
    xml_path = PROJECT_ROOT / f"data/eval_dataset/slideavsr/ocr_cache/{file_id}.xml"
    if not xml_path.exists():
        return []
    return parse_ocr_xml(xml_path.read_text())


def _load_entries() -> list[dict]:
    """Discover ground-truth / screenapp pairs across all three datasets."""
    entries: list[dict] = []

    for ds, root in [
        ("slideavsr", PROJECT_ROOT / "data/eval_dataset/slideavsr/transcripts"),
        ("ami_v2", PROJECT_ROOT / "data/eval_dataset/ami_v2/transcripts"),
        ("earnings22", PROJECT_ROOT / "data/eval_dataset/earnings22/transcripts"),
    ]:
        if not root.exists():
            continue
        for gt_path in sorted(root.glob("*_ground_truth.txt")):
            file_id = gt_path.name.removesuffix("_ground_truth.txt")
            sa_path = root / f"{file_id}_screenapp.json"
            if not sa_path.exists():
                continue
            entries.append({
                "dataset": ds,
                "file_id": file_id,
                "gt_file": gt_path,
                "sa_file": sa_path,
            })
    return entries


def _evaluate_one(
    entry: dict,
    use_wikipedia: bool = True,
    use_llm: bool = False,
    model_handles=None,
) -> FileResult:
    import jiwer
    from asr_correction.target_term_extractor import extract_target_terms
    from asr_correction.tter import compute_tter
    from asr_correction.dynamic_detector import (
        detect_candidates_channel_c,
        detect_candidates_rare_words,
        _phonetic_sim,
    )
    from asr_correction.evidence_gatherer import (
        gather_evidence_ocr,
        gather_evidence_wikipedia,
    )
    if use_llm:
        from asr_correction.llm_semantic_detector import detect_candidates_llm_semantic

    gt_text = entry["gt_file"].read_text()
    sa_data = json.loads(entry["sa_file"].read_text())
    sa_text = sa_data.get("text", "")

    # Baseline metrics
    ref_norm = _normalize_for_metric(gt_text)
    baseline_hyp_norm = _normalize_for_metric(sa_text)
    target_terms = extract_target_terms(gt_text)
    baseline_wer = round(jiwer.wer(ref_norm, baseline_hyp_norm) * 100, 2)
    baseline_tter_full = compute_tter(ref_norm, baseline_hyp_norm, target_terms)
    baseline_tter = round(baseline_tter_full["tter"] * 100, 2)

    applied: list[dict] = []
    skipped = 0
    seen_originals: set[str] = set()

    # --- Phase 2 detection: Channel C (OCR) — SlideAVSR only ---
    ocr_candidates: list = []
    if entry["dataset"] == "slideavsr":
        ocr_frames = _load_ocr_frames_slideavsr(entry["file_id"])
        if ocr_frames:
            ocr_candidates = detect_candidates_channel_c(sa_data, ocr_frames)
            for cand in ocr_candidates:
                ev = gather_evidence_ocr(cand)
                if ev.tier == "hard" and ev.confidence >= HARD_EVIDENCE_THRESHOLD and ev.target:
                    original = (cand.multiword_phrase
                                if cand.signal == "ocr_concat" and cand.multiword_phrase
                                else cand.nearest_transcript_word)
                    key = original.strip(_PUNCT_STRIP).lower()
                    if key in seen_originals:
                        continue
                    seen_originals.add(key)
                    applied.append({
                        "original": original,
                        "target": ev.target,
                        "signal": cand.signal,
                        "source": "ocr",
                        "confidence": round(ev.confidence, 3),
                        "phonetic_similarity": round(ev.phonetic_similarity, 3),
                        "snippet": ev.evidence_snippet,
                    })
                else:
                    skipped += 1

    # --- Phase 3 detection: rare-word scan + Wikipedia ---
    rare_candidates: list = []
    if use_wikipedia:
        rare_candidates = detect_candidates_rare_words(sa_data, max_candidates=40)
        for cand in rare_candidates:
            key = cand.nearest_transcript_word.strip(_PUNCT_STRIP).lower()
            if key in seen_originals:
                continue
            ev = gather_evidence_wikipedia(cand)
            if ev.tier == "hard" and ev.confidence >= HARD_EVIDENCE_THRESHOLD and ev.target:
                seen_originals.add(key)
                applied.append({
                    "original": cand.nearest_transcript_word,
                    "target": ev.target,
                    "signal": "rare_word",
                    "source": "wikipedia",
                    "confidence": round(ev.confidence, 3),
                    "phonetic_similarity": round(ev.phonetic_similarity, 3),
                    "snippet": ev.evidence_snippet,
                })
            else:
                skipped += 1

    # --- Phase 4 detection: LLM semantic check (Channel B) ---
    llm_proposals: list = []
    if use_llm:
        # Build topic context from the first 500 chars of the transcript
        # (a rough proxy for the full topic classification that the real
        # pipeline does in Step 2.5).
        first_text = (sa_data.get("text") or "")[:500]
        topic_ctx = f"Transcript excerpt: {first_text[:200]}..."
        try:
            llm_proposals = detect_candidates_llm_semantic(
                sa_data, max_chunks=2, model_handles=model_handles,
                topic_context=topic_ctx,
            )
        except Exception as e:
            print(f"    WARN: LLM detection failed: {e}")
        for p in llm_proposals:
            key = p.original.strip(_PUNCT_STRIP).lower()
            if key in seen_originals:
                continue
            # Strict filters for LLM-only tier:
            # 1. confidence >= threshold
            # 2. original and target both present and single-word
            # 3. phonetic similarity between original and target above gate
            if p.confidence < LLM_CONFIDENCE_THRESHOLD:
                skipped += 1
                continue
            if " " in p.original or " " in p.target:
                skipped += 1
                continue
            phon = _phonetic_sim(p.original, p.target)
            if phon < LLM_PHONETIC_GATE:
                skipped += 1
                continue
            seen_originals.add(key)
            applied.append({
                "original": p.original,
                "target": p.target,
                "signal": "llm_semantic",
                "source": "llm",
                "confidence": round(p.confidence, 3),
                "phonetic_similarity": round(phon, 3),
                "snippet": f"LLM: {p.reason[:80]}",
            })

    # Apply corrections
    corrected_text = _apply_corrections_to_text(sa_text, applied)
    corrected_hyp_norm = _normalize_for_metric(corrected_text)
    corrected_wer = round(jiwer.wer(ref_norm, corrected_hyp_norm) * 100, 2)
    corrected_tter = round(
        compute_tter(ref_norm, corrected_hyp_norm, target_terms)["tter"] * 100, 2
    )

    return FileResult(
        dataset=entry["dataset"],
        file_id=entry["file_id"],
        gt_words=len(ref_norm.split()),
        sa_words=len(baseline_hyp_norm.split()),
        target_term_count=len(target_terms),
        target_total_in_ref=baseline_tter_full["target_total"],
        baseline_wer=baseline_wer,
        baseline_tter=baseline_tter,
        corrected_wer=corrected_wer,
        corrected_tter=corrected_tter,
        candidates_ocr=len(ocr_candidates),
        candidates_rare=len(rare_candidates),
        candidates_llm=len(llm_proposals),
        corrections_applied=len(applied),
        corrections_skipped_low_conf=skipped,
        applied_details=applied,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset: slideavsr | ami_v2 | earnings22")
    parser.add_argument("--no-wikipedia", action="store_true",
                        help="Disable Wikipedia lookup (Phase 2 only mode)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable Phase 4 LLM semantic detection (local MLX model)")
    args = parser.parse_args()

    entries = _load_entries()
    if args.dataset:
        entries = [e for e in entries if e["dataset"] == args.dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Dynamic-track evaluation: {len(entries)} files ===")
    print(f"    use_wikipedia = {not args.no_wikipedia}")
    print(f"    use_llm       = {args.llm}\n")

    model_handles = None
    if args.llm:
        print("Loading MLX model for Channel B ...")
        from asr_correction.model import load_model
        t_load = time.time()
        model_handles = load_model()
        print(f"  loaded in {time.time() - t_load:.1f}s\n")

    results: list[FileResult] = []
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['dataset']}/{entry['file_id']} ...", flush=True)
        r = _evaluate_one(
            entry,
            use_wikipedia=not args.no_wikipedia,
            use_llm=args.llm,
            model_handles=model_handles,
        )
        results.append(r)
        delta = round(r.baseline_tter - r.corrected_tter, 2)
        marker = "✓" if delta > 0.01 else ("✗" if delta < -0.01 else "-")
        print(
            f"    baseline TTER {r.baseline_tter:5.1f}%  →  "
            f"corrected TTER {r.corrected_tter:5.1f}%  "
            f"(Δ {delta:+.2f}) {marker}  "
            f"ocr={r.candidates_ocr} rare={r.candidates_rare} "
            f"llm={r.candidates_llm} applied={r.corrections_applied}"
        )
        for a in r.applied_details[:3]:
            orig_short = a['original'][:25]
            tgt_short = a['target'][:30]
            print(f"        [{a['source']}] '{orig_short}' → '{tgt_short}'  conf={a['confidence']:.2f}")
    elapsed = time.time() - t0

    # Write CSV
    csv_path = OUT_DIR / "dynamic_per_file.csv"
    fieldnames = [
        "dataset", "file_id", "gt_words", "sa_words",
        "target_term_count", "target_total_in_ref",
        "baseline_wer", "baseline_tter", "corrected_wer", "corrected_tter",
        "tter_delta", "wer_delta",
        "candidates_ocr", "candidates_rare", "candidates_llm",
        "corrections_applied", "corrections_skipped_low_conf",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "dataset": r.dataset,
                "file_id": r.file_id,
                "gt_words": r.gt_words,
                "sa_words": r.sa_words,
                "target_term_count": r.target_term_count,
                "target_total_in_ref": r.target_total_in_ref,
                "baseline_wer": r.baseline_wer,
                "baseline_tter": r.baseline_tter,
                "corrected_wer": r.corrected_wer,
                "corrected_tter": r.corrected_tter,
                "tter_delta": round(r.baseline_tter - r.corrected_tter, 2),
                "wer_delta": round(r.baseline_wer - r.corrected_wer, 2),
                "candidates_ocr": r.candidates_ocr,
                "candidates_rare": r.candidates_rare,
                "candidates_llm": r.candidates_llm,
                "corrections_applied": r.corrections_applied,
                "corrections_skipped_low_conf": r.corrections_skipped_low_conf,
            })

    (OUT_DIR / "dynamic_applied_corrections.json").write_text(json.dumps(
        [{"dataset": r.dataset, "file_id": r.file_id, "applied": r.applied_details}
         for r in results],
        indent=2,
    ))

    # Aggregates per dataset
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "hard_evidence_threshold": HARD_EVIDENCE_THRESHOLD,
        "total_files": len(results),
        "per_dataset": {},
    }
    for ds in sorted({r.dataset for r in results}):
        rows = [r for r in results if r.dataset == ds]
        n = len(rows)
        summary["per_dataset"][ds] = {
            "files": n,
            "avg_baseline_wer_pct": round(sum(r.baseline_wer for r in rows) / n, 2),
            "avg_corrected_wer_pct": round(sum(r.corrected_wer for r in rows) / n, 2),
            "avg_wer_delta_pct": round(
                sum(r.baseline_wer - r.corrected_wer for r in rows) / n, 2),
            "avg_baseline_tter_pct": round(sum(r.baseline_tter for r in rows) / n, 2),
            "avg_corrected_tter_pct": round(sum(r.corrected_tter for r in rows) / n, 2),
            "avg_tter_delta_pct": round(
                sum(r.baseline_tter - r.corrected_tter for r in rows) / n, 2),
            "total_candidates_ocr": sum(r.candidates_ocr for r in rows),
            "total_candidates_rare": sum(r.candidates_rare for r in rows),
            "total_candidates_llm": sum(r.candidates_llm for r in rows),
            "total_corrections_applied": sum(r.corrections_applied for r in rows),
        }
    (OUT_DIR / "dynamic_summary.json").write_text(json.dumps(summary, indent=2))

    # Pretty summary
    print(f"\n{'='*78}")
    print("PHASE 3 SUMMARY")
    print(f"{'='*78}")
    print(f"{'Dataset':<12} {'Files':>5} {'BaseWER':>8} {'CorrWER':>8} "
          f"{'BaseTTER':>9} {'CorrTTER':>9} {'ΔTTER':>8} {'Applied':>8}")
    print(f"{'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for ds, s in summary["per_dataset"].items():
        print(f"{ds:<12} {s['files']:>5} "
              f"{s['avg_baseline_wer_pct']:>7.1f}% {s['avg_corrected_wer_pct']:>7.1f}% "
              f"{s['avg_baseline_tter_pct']:>8.1f}% {s['avg_corrected_tter_pct']:>8.1f}% "
              f"{s['avg_tter_delta_pct']:>+7.2f} {s['total_corrections_applied']:>8}")
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {OUT_DIR / 'dynamic_summary.json'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
