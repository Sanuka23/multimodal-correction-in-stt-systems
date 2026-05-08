# v3 Dynamic-Evidence Evaluation — FROZEN

**Status:** Frozen. Do not modify the corresponding eval results.
**Date:** 2026-05-09
**Frozen at branch tip:** prior to commit that retired `run_dynamic_eval.py`

## What was the v3 dynamic-evidence eval?

A standalone evaluation harness that reproduced the production correction
pipeline's per-candidate evidence aggregation logic in a self-contained
script, separate from the FastAPI service.

It produced:

- `data/eval_results_v3/dynamic_summary.json`
- `data/eval_results_v3/dynamic_per_file.csv`
- `data/eval_results_v3/dynamic_applied_corrections.json`

These are the bench numbers cited in the FYP dissertation
(SlideAVSR +0.43 pp TTER, AMI v2 0.00 pp, Earnings22 -0.03 pp).

## Why retired?

The harness lived in `scripts/evaluation/run_dynamic_eval.py` and depended on
three legacy modules in `asr_correction/`:

- `dynamic_detector.py`
- `evidence_gatherer.py`
- `llm_semantic_detector.py`

All three were superseded by the v4.1 production pipeline
(`asr_correction/__init__.py :: correct_transcript`), which now emits
the same evidence channels via per-step events.

Re-running the bench against v4.1 would require building a parallel harness
that calls `correct_transcript()` over the dataset directly. That work is a
separate project (see `docs/PLAN_web_vocab_and_avsr.md` future work) and is
intentionally out of scope for the dissertation.

## To resurrect this harness

The script and its three module dependencies can be recovered via git:

```bash
git log --diff-filter=D --summary -- scripts/evaluation/run_dynamic_eval.py
# Find the deletion commit hash, then:
git show <hash>:scripts/evaluation/run_dynamic_eval.py > scripts/evaluation/run_dynamic_eval.py
git show <hash>:asr_correction/dynamic_detector.py     > asr_correction/dynamic_detector.py
git show <hash>:asr_correction/evidence_gatherer.py    > asr_correction/evidence_gatherer.py
git show <hash>:asr_correction/llm_semantic_detector.py > asr_correction/llm_semantic_detector.py
```

The frozen result files in `data/eval_results_v3/` remain untouched.
