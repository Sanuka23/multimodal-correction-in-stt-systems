#!/usr/bin/env python3
"""Baseline TTER evaluation for SlideAVSR, AMI v2, and Earnings-22.

Reads ground-truth and ScreenApp baseline transcripts from
data/eval_dataset/{slideavsr,ami_v2,earnings22}/, extracts target terms,
computes TTER, and writes per-file rows to data/eval_results_v3/per_file.csv
plus a summary to data/eval_results_v3/summary.json.

Run with the conda Python:
    /opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_tter_eval.py
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


@dataclass
class EvalEntry:
    """One ground-truth / hypothesis pair to evaluate."""
    dataset: str
    file_id: str
    gt_text: str
    sa_text: str


def _load_pairs(root: Path, dataset: str) -> list[EvalEntry]:
    """Load (ground_truth, screenapp) pairs from a transcripts directory."""
    if not root.exists():
        return []
    out: list[EvalEntry] = []
    for gt_path in sorted(root.glob("*_ground_truth.txt")):
        file_id = gt_path.name.removesuffix("_ground_truth.txt")
        sa_path = root / f"{file_id}_screenapp.json"
        if not sa_path.exists():
            continue
        sa_data = json.loads(sa_path.read_text())
        out.append(EvalEntry(
            dataset=dataset,
            file_id=file_id,
            gt_text=gt_path.read_text(),
            sa_text=sa_data.get("text", ""),
        ))
    return out


def load_all_entries() -> list[EvalEntry]:
    return (
        _load_pairs(PROJECT_ROOT / "data/eval_dataset/slideavsr/transcripts", "slideavsr")
        + _load_pairs(PROJECT_ROOT / "data/eval_dataset/ami_v2/transcripts", "ami_v2")
        + _load_pairs(PROJECT_ROOT / "data/eval_dataset/earnings22/transcripts", "earnings22")
    )


def _normalize_for_metric(text: str) -> str:
    """Lowercase and collapse whitespace for WER/TTER input.

    Does NOT strip punctuation-like symbols heuristically — jiwer.process_words
    handles whitespace tokenization. Mirrors the lightweight normalization used
    in existing upload scripts so the numbers here are comparable to what is
    already in the dashboard.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_one(entry: EvalEntry) -> dict:
    """Compute baseline WER + TTER for a single entry."""
    import jiwer
    from asr_correction.target_term_extractor import extract_target_terms
    from asr_correction.tter import compute_tter

    ref_norm = _normalize_for_metric(entry.gt_text)
    hyp_norm = _normalize_for_metric(entry.sa_text)
    wer = round(jiwer.wer(ref_norm, hyp_norm) * 100, 2)

    # Target terms come from the UNNORMALIZED ground truth so the extractor
    # sees real casing and punctuation (helps spaCy NER and the all-caps
    # heuristic in the extractor).
    target_terms = extract_target_terms(entry.gt_text)

    # TTER uses the normalized strings so that case / punctuation differences
    # do not inflate the error count.
    tter_result = compute_tter(ref_norm, hyp_norm, target_terms)

    return {
        "dataset": entry.dataset,
        "file_id": entry.file_id,
        "gt_words": len(ref_norm.split()),
        "sa_words": len(hyp_norm.split()),
        "target_term_count": len(target_terms),
        "target_total_in_ref": tter_result["target_total"],
        "wer_pct": wer,
        "tter_pct": round(tter_result["tter"] * 100, 2),
        "substitutions": tter_result["substitutions"],
        "deletions": tter_result["deletions"],
        "insertions": tter_result["insertions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true",
                        help="Just list discovered entries and exit")
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset: slideavsr | ami_v2 | earnings22")
    args = parser.parse_args()

    entries = load_all_entries()
    if args.dataset:
        entries = [e for e in entries if e.dataset == args.dataset]

    print(f"Discovered {len(entries)} evaluation entries:")
    by_dataset: dict[str, int] = {}
    for e in entries:
        by_dataset[e.dataset] = by_dataset.get(e.dataset, 0) + 1
    for ds, n in sorted(by_dataset.items()):
        print(f"  {ds}: {n} files")

    if args.list_only:
        return 0

    out_dir = PROJECT_ROOT / "data/eval_results_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_file.csv"
    summary_path = out_dir / "summary.json"

    rows: list[dict] = []
    t_start = time.time()
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry.dataset}/{entry.file_id} ...", flush=True)
        row = _compute_one(entry)
        rows.append(row)
        print(
            f"    WER {row['wer_pct']:5.1f}%   "
            f"TTER {row['tter_pct']:5.1f}%   "
            f"targets={row['target_term_count']}   "
            f"target_in_ref={row['target_total_in_ref']}"
        )
    elapsed = time.time() - t_start

    # Write CSV
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Write summary (per-dataset aggregates)
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "total_files": len(rows),
        "per_dataset": {},
    }
    for ds in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        n = len(ds_rows)
        summary["per_dataset"][ds] = {
            "files": n,
            "avg_wer_pct": round(sum(r["wer_pct"] for r in ds_rows) / n, 2),
            "avg_tter_pct": round(sum(r["tter_pct"] for r in ds_rows) / n, 2),
            "total_target_terms": sum(r["target_term_count"] for r in ds_rows),
            "total_target_positions": sum(r["target_total_in_ref"] for r in ds_rows),
        }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Final table
    print(f"\n{'='*72}")
    print(f"{'Dataset':<12} {'Files':>6} {'Avg WER':>10} {'Avg TTER':>10} {'Target Terms':>14}")
    print(f"{'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*14}")
    for ds, s in summary["per_dataset"].items():
        print(f"{ds:<12} {s['files']:>6} "
              f"{s['avg_wer_pct']:>9.1f}% {s['avg_tter_pct']:>9.1f}% "
              f"{s['total_target_terms']:>14}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
