# Phase 1 — Measurement Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the target term extractor, TTER metric, and baseline evaluation runner so we can measure how wrong the current ScreenApp transcripts are on domain-relevant terms — before touching the correction pipeline.

**Architecture:** Three self-contained modules added under `asr_correction/` plus one evaluation runner under `scripts/evaluation/`. Each module has a single responsibility, is unit-tested with `pytest`, and is pure-Python (no pipeline dependency). The runner composes them to compute baseline TTER for every file in SlideAVSR, AMI v2, and Earnings-22, producing a single CSV + summary JSON that the dashboard will later read.

**Tech Stack:** Python 3, `spacy` (NER, `en_core_web_sm`), `jiwer.process_words` (alignment), `jellyfish` (phonetic similarity), `pytest`, existing ScreenApp eval datasets under `data/eval_dataset/`.

**Source spec:** `docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md` — sections 4.6 (Target Term Extractor), 4.7 (TTER Metric), 6.1–6.3 (Evaluation Plan).

**Python interpreter:** all commands use the conda Python at `/opt/homebrew/Caskroom/miniforge/base/bin/python3` because the system Python lacks pipeline deps (per prior session memory).

---

## File Structure

Files to create:

| Path | Responsibility |
|------|----------------|
| `asr_correction/target_term_extractor.py` | Given ground-truth text, return a set of target terms using NER + proper noun + acronym + keyword rules |
| `asr_correction/tter.py` | Given reference, hypothesis, and target terms, return TTER + per-edit counts |
| `asr_correction/technical_keywords.py` | Static curated list of technical / business keywords used by the extractor |
| `tests/__init__.py` | Empty, marks tests dir as a package |
| `tests/test_target_term_extractor.py` | Unit tests for extractor rules |
| `tests/test_tter.py` | Unit tests for TTER metric on hand-crafted examples |
| `scripts/evaluation/run_tter_eval.py` | End-to-end runner: walks each dataset, computes baseline TTER, writes CSV + JSON |
| `data/eval_results_v3/.gitkeep` | Ensures the output directory exists |
| `pytest.ini` | Minimal pytest config pinning the test path and Python |

Files to modify: **none in Phase 1.** The correction pipeline is untouched.

---

## Task 0: Environment Setup

**Files:**
- Create: `pytest.ini`
- Verify: conda Python has `spacy` + `en_core_web_sm` installed

- [ ] **Step 1: Install spaCy + English model**

The conda Python already has `jiwer`, `jellyfish`, `faster-whisper`, etc. but is missing `spacy`. Install it and the small English model:

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pip install 'spacy>=3.7,<4.0'
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m spacy download en_core_web_sm
```

Expected: both commands complete without errors. Verify:

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(nlp('Laura is the project manager.').ents)"
```

Expected output: `(Laura,)` (or similar — a non-empty tuple of entity spans).

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
```

- [ ] **Step 3: Create empty `tests/__init__.py`**

```python
```

(empty file)

- [ ] **Step 4: Verify pytest discovery**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest --collect-only 2>&1 | tail -5
```

Expected: `no tests ran in X.XXs` or `collected 0 items` — confirms pytest is wired up and ready.

- [ ] **Step 5: Commit**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
git add pytest.ini tests/__init__.py
git commit -m "chore: add pytest config and tests/ directory"
```

---

## Task 1: Technical Keywords Module

**Files:**
- Create: `asr_correction/technical_keywords.py`

A static module that exports a frozenset of domain keywords used by the target term extractor as Rule 4. Kept separate so it can grow without touching the extractor logic.

- [ ] **Step 1: Write `technical_keywords.py`**

```python
"""Curated technical and business terms for target term extraction.

These are common-English words that would NOT be caught by NER or proper-noun
rules but are still domain-relevant and should count as target terms for TTER.

Lower-cased; the extractor compares case-insensitively.
"""

TECHNICAL_KEYWORDS: frozenset[str] = frozenset({
    # AI / ML
    "transformer", "attention", "embedding", "gradient", "backprop",
    "encoder", "decoder", "tokenizer", "softmax", "relu", "sigmoid",
    "inference", "fine-tuning", "pretraining", "prompt", "token",
    "logits", "checkpoint", "hyperparameter", "convolution", "regression",
    "classification", "clustering", "perceptron", "quantization",

    # Software engineering
    "api", "endpoint", "schema", "payload", "microservice", "middleware",
    "kubernetes", "docker", "container", "pipeline", "webhook",
    "dependency", "framework", "runtime", "compiler", "bytecode",
    "database", "cache", "queue", "broker", "cluster", "deployment",

    # Business / finance
    "revenue", "margin", "ebitda", "capex", "opex", "valuation",
    "quarterly", "forecast", "dividend", "equity", "liquidity",
    "portfolio", "fiscal", "gaap", "shareholder", "earnings",
})
```

- [ ] **Step 2: Verify the module imports**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -c "from asr_correction.technical_keywords import TECHNICAL_KEYWORDS; print(len(TECHNICAL_KEYWORDS), 'keywords'); print('relu' in TECHNICAL_KEYWORDS)"
```

Expected: `55 keywords\nTrue` (count may differ if you adjust the list).

- [ ] **Step 3: Commit**

```bash
git add asr_correction/technical_keywords.py
git commit -m "feat(eval): add curated technical keyword list for target term extraction"
```

---

## Task 2: Target Term Extractor — Test First

**Files:**
- Create: `tests/test_target_term_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for asr_correction.target_term_extractor."""

import pytest


@pytest.fixture(scope="module")
def extract():
    from asr_correction.target_term_extractor import extract_target_terms
    return extract_target_terms


def test_extracts_person_names(extract):
    text = "Laura is the project manager and David is the industrial designer."
    terms = extract(text)
    assert "Laura" in terms
    assert "David" in terms


def test_extracts_org_and_product_names(extract):
    text = "We deployed the new service on Cloudflare and Kubernetes last week."
    terms = extract(text)
    assert "Cloudflare" in terms
    assert "Kubernetes" in terms


def test_extracts_acronyms(extract):
    text = "The CVPR paper introduces a new CNN architecture for GPU inference."
    terms = extract(text)
    assert "CVPR" in terms
    assert "CNN" in terms
    assert "GPU" in terms


def test_extracts_technical_keywords(extract):
    text = "We apply the relu activation function before the softmax output."
    terms = extract(text)
    # technical keywords are case-insensitive, extractor returns the as-written form
    lowered = {t.lower() for t in terms}
    assert "relu" in lowered
    assert "softmax" in lowered


def test_filters_common_function_words(extract):
    text = "THE cat sat on THE mat AND looked FOR food."
    terms = extract(text)
    # THE, AND, FOR must not count as acronyms
    assert "THE" not in terms
    assert "AND" not in terms
    assert "FOR" not in terms


def test_filters_common_english_proper_noun_false_positives(extract):
    text = "On Monday morning, the report was due."
    terms = extract(text)
    # Day-of-week words are NER DATE, not PERSON/ORG/PRODUCT — should be excluded
    assert "Monday" not in terms


def test_empty_input_returns_empty_set(extract):
    assert extract("") == set()


def test_returns_a_set_not_a_list(extract):
    result = extract("Laura and David met at Cloudflare.")
    assert isinstance(result, set)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_target_term_extractor.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'asr_correction.target_term_extractor'` — the module does not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_target_term_extractor.py
git commit -m "test(eval): add failing tests for target term extractor"
```

---

## Task 3: Target Term Extractor — Implementation

**Files:**
- Create: `asr_correction/target_term_extractor.py`

Implements §4.6 of the spec: four rules (NER, proper nouns, acronyms, technical keywords). Loads spaCy once at module level to avoid per-call overhead.

- [ ] **Step 1: Write `target_term_extractor.py`**

```python
"""Extract target terms from a ground-truth transcript for TTER evaluation.

Target terms are words whose misrecognition is semantically significant:
person names, organization / product names, technical jargon, and acronyms.
See docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.6.
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy

from .technical_keywords import TECHNICAL_KEYWORDS

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


def extract_target_terms(text: str) -> set[str]:
    """Return the set of target terms in `text`.

    Applied rules (union):
    1. spaCy NER entities with label in {PERSON, ORG, PRODUCT, WORK_OF_ART, GPE}
    2. Proper nouns (POS=PROPN) not in the common-word exclusion list
    3. Acronyms matching \\b[A-Z]{2,}(?:-?\\d+)?\\b, excluding function words
    4. Technical / business keywords from TECHNICAL_KEYWORDS (case-insensitive)
    """
    if not text.strip():
        return set()

    nlp = _get_nlp()
    doc = nlp(text)
    terms: set[str] = set()

    # Rule 1: named entities
    for ent in doc.ents:
        if ent.label_ in _ENTITY_LABELS:
            terms.add(ent.text)

    # Rule 2: proper nouns not in common exclusion list
    for token in doc:
        if token.pos_ == "PROPN" and token.text.lower() not in _COMMON_PROPN_WORDS:
            terms.add(token.text)

    # Rule 3: acronyms (text-level regex, not spaCy-dependent)
    for match in _ACRONYM_RE.findall(text):
        if match not in _FUNCTION_WORD_CAPS:
            terms.add(match)

    # Rule 4: technical keywords (case-insensitive match, preserve as-written form)
    lower_text_tokens = {t.text for t in doc if t.text.lower() in TECHNICAL_KEYWORDS}
    terms.update(lower_text_tokens)

    return terms
```

- [ ] **Step 2: Run the tests**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_target_term_extractor.py -v 2>&1 | tail -30
```

Expected: all 8 tests PASS.

- [ ] **Step 3: Smoke-test on a real SlideAVSR transcript**

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -c "
from asr_correction.target_term_extractor import extract_target_terms
from pathlib import Path
gt = Path('data/eval_dataset/slideavsr/transcripts/40Kw8QLym7E_ground_truth.txt').read_text()
terms = extract_target_terms(gt)
print(f'{len(terms)} target terms found:')
for t in sorted(terms)[:30]:
    print(' -', t)
"
```

Expected: at least 10 terms, including research-flavoured items (e.g. `MAGNET`, `CVPR`, `MULTI-SCALE`, proper nouns). If you see a flood of obvious non-terms, adjust the extractor — but do not over-tune to one file.

- [ ] **Step 4: Commit**

```bash
git add asr_correction/target_term_extractor.py
git commit -m "feat(eval): implement target term extractor with spaCy NER + rules"
```

---

## Task 4: TTER Metric — Test First

**Files:**
- Create: `tests/test_tter.py`

TTER is a word-level edit rate restricted to positions where the reference word is a target term. We test it on hand-crafted examples where the correct answer is obvious from the alignment.

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for asr_correction.tter."""

import pytest


@pytest.fixture(scope="module")
def compute_tter():
    from asr_correction.tter import compute_tter
    return compute_tter


def test_perfect_hypothesis_has_zero_tter(compute_tter):
    ref = "Laura deployed the service on Cloudflare yesterday"
    hyp = "Laura deployed the service on Cloudflare yesterday"
    target_terms = {"Laura", "Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["tter"] == 0.0
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["target_total"] == 2


def test_substitution_on_target_term_counts(compute_tter):
    ref = "Laura deployed to Cloudflare"
    hyp = "Laura deployed to Cloudware"  # Cloudflare → Cloudware
    target_terms = {"Laura", "Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["substitutions"] == 1
    assert result["target_total"] == 2
    assert result["tter"] == pytest.approx(0.5)


def test_substitution_on_non_target_word_is_ignored(compute_tter):
    ref = "Laura deployed the service"
    hyp = "Laura deployed a service"  # "the" → "a", not a target
    target_terms = {"Laura"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["substitutions"] == 0
    assert result["tter"] == 0.0


def test_deletion_of_target_term_counts(compute_tter):
    ref = "Laura and David met"
    hyp = "Laura met"  # David deleted
    target_terms = {"Laura", "David"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["deletions"] == 1
    assert result["target_total"] == 2
    assert result["tter"] == pytest.approx(0.5)


def test_insertion_of_target_shaped_word_counts(compute_tter):
    # "Kubernetes" inserted in hypothesis; it's a target term by name.
    ref = "we deployed the service"
    hyp = "we deployed Kubernetes the service"
    target_terms = {"Kubernetes"}
    result = compute_tter(ref, hyp, target_terms)
    # No target terms in ref → target_total = 0 → tter is defined as 0.0 by convention
    # but insertions are still tracked.
    assert result["insertions"] == 1
    assert result["target_total"] == 0
    assert result["tter"] == 0.0  # undefined → 0 by our convention


def test_insertion_of_non_target_word_is_ignored(compute_tter):
    ref = "Laura met David"
    hyp = "Laura quickly met David"  # "quickly" inserted — not a target
    target_terms = {"Laura", "David"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["insertions"] == 0


def test_empty_target_terms_returns_zero_tter(compute_tter):
    ref = "the cat sat on the mat"
    hyp = "the cat on mat"  # errors exist but nothing is a target
    target_terms = set()
    result = compute_tter(ref, hyp, target_terms)
    assert result["tter"] == 0.0
    assert result["target_total"] == 0


def test_case_insensitive_matching(compute_tter):
    ref = "Laura deployed to cloudflare"
    hyp = "Laura deployed to CloudFlare"  # case difference only
    target_terms = {"Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    # Case-insensitive compare — should NOT count as substitution
    assert result["substitutions"] == 0
    assert result["tter"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_tter.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'asr_correction.tter'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_tter.py
git commit -m "test(eval): add failing TTER metric tests"
```

---

## Task 5: TTER Metric — Implementation

**Files:**
- Create: `asr_correction/tter.py`

Uses `jiwer.process_words` to get a word-level alignment, then walks the alignment to count edits only at positions where the reference word is a target term. Insertions are classified as target insertions only if the inserted word is itself a target term.

- [ ] **Step 1: Write `tter.py`**

```python
"""Target Term Error Rate (TTER) metric.

Computes a word-level error rate restricted to positions where the reference
word is in a user-supplied set of "target terms" (domain-relevant vocabulary).
See docs/superpowers/specs/2026-04-15-dynamic-correction-track-design.md §4.7.
"""

from __future__ import annotations

from typing import Iterable, TypedDict

import jiwer


class TTERResult(TypedDict):
    tter: float
    substitutions: int
    deletions: int
    insertions: int
    target_total: int


def _normalize(tokens: Iterable[str]) -> list[str]:
    """Lowercase for case-insensitive comparison and target term matching."""
    return [t.lower() for t in tokens]


def compute_tter(
    reference: str,
    hypothesis: str,
    target_terms: set[str],
) -> TTERResult:
    """Compute Target Term Error Rate.

    Args:
        reference: ground-truth transcript text.
        hypothesis: predicted transcript text.
        target_terms: set of terms (case-insensitive) to restrict error counting to.

    Returns:
        Dict with `tter`, `substitutions`, `deletions`, `insertions`, `target_total`.
        `tter` is 0.0 when `target_total == 0` (undefined → 0 by convention).
    """
    target_lower = {t.lower() for t in target_terms}

    alignment = jiwer.process_words(reference, hypothesis)
    # process_words returns a WordOutput with `.references`, `.hypotheses`,
    # and `.alignments`. Each alignment is a list of Chunk objects per (ref, hyp) pair.

    ref_tokens = _normalize(alignment.references[0])
    hyp_tokens = _normalize(alignment.hypotheses[0])
    chunks = alignment.alignments[0]

    substitutions = 0
    deletions = 0
    insertions = 0
    target_total = sum(1 for t in ref_tokens if t in target_lower)

    for chunk in chunks:
        op = chunk.type  # one of: equal, substitute, delete, insert
        if op == "equal":
            continue
        if op == "substitute":
            # Reference positions [chunk.ref_start_idx : chunk.ref_end_idx]
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if ref_tokens[i] in target_lower:
                    substitutions += 1
        elif op == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if ref_tokens[i] in target_lower:
                    deletions += 1
        elif op == "insert":
            # No reference word — check if inserted hyp words are themselves targets
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if hyp_tokens[j] in target_lower:
                    insertions += 1

    if target_total == 0:
        tter = 0.0
    else:
        tter = (substitutions + deletions + insertions) / target_total

    return TTERResult(
        tter=tter,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        target_total=target_total,
    )
```

- [ ] **Step 2: Run the tests**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_tter.py -v 2>&1 | tail -30
```

Expected: all 8 tests PASS.

If `chunk.type` / `chunk.ref_start_idx` attribute names differ in your installed `jiwer` version, introspect first:

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -c "
import jiwer
r = jiwer.process_words('a b c', 'a x c')
print(type(r.alignments[0][0]))
print(r.alignments[0])
"
```

Then adjust the attribute names in `tter.py` accordingly. Keep the function signature and return shape identical.

- [ ] **Step 3: Commit**

```bash
git add asr_correction/tter.py
git commit -m "feat(eval): implement TTER metric with jiwer alignment walk"
```

---

## Task 6: Baseline Evaluation Runner — Dataset Discovery

**Files:**
- Create: `scripts/evaluation/run_tter_eval.py` (skeleton only in this task)
- Create: `data/eval_results_v3/.gitkeep`

Builds the skeleton that discovers datasets and reads ground truth + ScreenApp transcripts. Does NOT compute TTER yet — that comes in Task 7. Keeping the discovery logic isolated makes it testable by running the script with `--list-only`.

- [ ] **Step 1: Create output directory sentinel**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
mkdir -p data/eval_results_v3
touch data/eval_results_v3/.gitkeep
```

- [ ] **Step 2: Write the skeleton script**

```python
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
import json
import sys
from dataclasses import dataclass, asdict
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


def _load_slideavsr() -> list[EvalEntry]:
    root = PROJECT_ROOT / "data/eval_dataset/slideavsr/transcripts"
    if not root.exists():
        return []
    out = []
    for gt_path in sorted(root.glob("*_ground_truth.txt")):
        file_id = gt_path.name.removesuffix("_ground_truth.txt")
        sa_path = root / f"{file_id}_screenapp.json"
        if not sa_path.exists():
            continue
        sa_data = json.loads(sa_path.read_text())
        out.append(EvalEntry(
            dataset="slideavsr",
            file_id=file_id,
            gt_text=gt_path.read_text(),
            sa_text=sa_data.get("text", ""),
        ))
    return out


def _load_ami_v2() -> list[EvalEntry]:
    root = PROJECT_ROOT / "data/eval_dataset/ami_v2/transcripts"
    if not root.exists():
        return []
    out = []
    for gt_path in sorted(root.glob("*_ground_truth.txt")):
        file_id = gt_path.name.removesuffix("_ground_truth.txt")
        sa_path = root / f"{file_id}_screenapp.json"
        if not sa_path.exists():
            continue
        sa_data = json.loads(sa_path.read_text())
        out.append(EvalEntry(
            dataset="ami_v2",
            file_id=file_id,
            gt_text=gt_path.read_text(),
            sa_text=sa_data.get("text", ""),
        ))
    return out


def _load_earnings22() -> list[EvalEntry]:
    root = PROJECT_ROOT / "data/eval_dataset/earnings22/transcripts"
    if not root.exists():
        return []
    out = []
    for gt_path in sorted(root.glob("*_ground_truth.txt")):
        file_id = gt_path.name.removesuffix("_ground_truth.txt")
        sa_path = root / f"{file_id}_screenapp.json"
        if not sa_path.exists():
            continue
        sa_data = json.loads(sa_path.read_text())
        out.append(EvalEntry(
            dataset="earnings22",
            file_id=file_id,
            gt_text=gt_path.read_text(),
            sa_text=sa_data.get("text", ""),
        ))
    return out


def load_all_entries() -> list[EvalEntry]:
    return _load_slideavsr() + _load_ami_v2() + _load_earnings22()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true",
                        help="Just list discovered entries and exit")
    args = parser.parse_args()

    entries = load_all_entries()
    print(f"Discovered {len(entries)} evaluation entries:")
    by_dataset: dict[str, int] = {}
    for e in entries:
        by_dataset[e.dataset] = by_dataset.get(e.dataset, 0) + 1
    for ds, n in sorted(by_dataset.items()):
        print(f"  {ds}: {n} files")

    if args.list_only:
        return 0

    print("\n[ TTER computation wiring comes in Task 7 ]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the skeleton with `--list-only`**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_tter_eval.py --list-only
```

Expected output (numbers may differ slightly):

```
Discovered 24 evaluation entries:
  ami_v2: 10 files
  earnings22: 4 files
  slideavsr: 10 files
```

If a dataset shows 0 files, check that both `_ground_truth.txt` and `_screenapp.json` exist in that dataset's `transcripts/` directory. Do NOT hard-code any fallback — fix the discovery or note the dataset as missing in the Open Questions section of the spec.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/run_tter_eval.py data/eval_results_v3/.gitkeep
git commit -m "feat(eval): add baseline TTER runner skeleton with dataset discovery"
```

---

## Task 7: Baseline Evaluation Runner — TTER Computation

**Files:**
- Modify: `scripts/evaluation/run_tter_eval.py` (add the computation pass and output writers)

- [ ] **Step 1: Extend `run_tter_eval.py` with TTER computation**

Replace the `main()` function and add two new helpers. Show the complete replacement — keep everything above `def main()` exactly as-is.

```python
# Add these imports at the top of the file, right after `import json`
import csv
import time
from datetime import datetime, timezone
```

```python
# New helper — add between load_all_entries() and main()
def _normalize_for_metric(text: str) -> str:
    """Lowercase and collapse whitespace for WER/TTER input.

    Does NOT strip punctuation — `jiwer.process_words` handles tokenization.
    Mirrors the lightweight normalization used in existing upload scripts.
    """
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_one(entry: EvalEntry) -> dict:
    """Compute baseline WER + TTER for a single entry."""
    import jiwer
    from asr_correction.target_term_extractor import extract_target_terms
    from asr_correction.tter import compute_tter

    # WER on full transcript (for the "we did not make things worse" column)
    ref_norm = _normalize_for_metric(entry.gt_text)
    hyp_norm = _normalize_for_metric(entry.sa_text)
    wer = round(jiwer.wer(ref_norm, hyp_norm) * 100, 2)

    # Target terms come from the unnormalized ground truth so spaCy sees real casing
    target_terms = extract_target_terms(entry.gt_text)

    # TTER uses the normalized strings so case differences do not count as errors
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
```

Now replace the existing `main()` function with this full version:

```python
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
    summary = {
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
```

- [ ] **Step 2: Run on one dataset first (sanity check)**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_tter_eval.py --dataset slideavsr
```

Expected: 10 lines of per-file output, then a summary showing SlideAVSR with an average TTER value. If TTER comes out as `0.0%` across all files, the target term extractor is returning empty sets — debug the extractor before proceeding. If TTER comes out as identical to WER across all files, the target term filter is not doing its job — verify the target-term lowercased set is correct.

- [ ] **Step 3: Run on all datasets**

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_tter_eval.py
```

Expected: three rows in the summary table (ami_v2, earnings22, slideavsr), two output files written (`per_file.csv`, `summary.json`).

Expected ballparks (rough — actual numbers depend on the extractor's recall):

- `slideavsr`: avg WER 13–14%, avg TTER 5–15%. TTER should generally be higher than WER on target terms because ASR struggles with proper nouns / technical terms relative to common English.
- `ami_v2`: avg WER 29–30%, avg TTER 15–40%. Conversational speech, hard.
- `earnings22`: avg WER 15–16%, avg TTER 5–25%. Depends how many tickers/names made it into ground truth.

The exact numbers matter less than the relative picture — these are the baselines you will compare against after Phase 2+.

- [ ] **Step 4: Spot-check the CSV**

```bash
head -5 data/eval_results_v3/per_file.csv
cat data/eval_results_v3/summary.json
```

Expected: a CSV header line followed by rows, and a JSON summary with three dataset entries.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/run_tter_eval.py data/eval_results_v3/per_file.csv data/eval_results_v3/summary.json
git commit -m "feat(eval): compute baseline WER + TTER for all eval datasets"
```

---

## Task 8: Phase 1 Wrap-Up

**Files:**
- None (verification only)

- [ ] **Step 1: Full test suite green**

```bash
cd /Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest -v 2>&1 | tail -20
```

Expected: all tests pass (8 extractor tests + 8 TTER tests = 16 total).

- [ ] **Step 2: Confirm no pipeline regressions**

Phase 1 touches zero pipeline code, but verify by running a quick import smoke test on the correction entry point:

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -c "from asr_correction import correct_transcript; print('correct_transcript import OK')"
```

Expected: `correct_transcript import OK`. If this fails, something unrelated is broken and should be investigated before moving to Phase 2.

- [ ] **Step 3: Record baseline results in the report draft**

This is a manual step — copy the numbers from `data/eval_results_v3/summary.json` into the report's "Current State" / "Baseline Measurements" section. This is the first concrete, TTER-based number in the report and should be cited with the git SHA of the commit from Task 7 so the measurement is reproducible.

- [ ] **Step 4: Tag the commit**

```bash
git tag phase-1-baseline-complete
```

This gives you a named checkpoint to diff against when Phase 2 lands corrections on top.

---

## Phase 1 Done — What You Have

- A pytest-wired codebase with 16 passing unit tests.
- `asr_correction.target_term_extractor.extract_target_terms(text) -> set[str]` producing auditable target term lists.
- `asr_correction.tter.compute_tter(ref, hyp, target_terms) -> TTERResult` using `jiwer.process_words` alignments.
- `scripts/evaluation/run_tter_eval.py` producing `data/eval_results_v3/per_file.csv` + `summary.json` for all three datasets.
- Report-ready baseline WER and TTER numbers that your FYP thesis can cite today — independent of any pipeline changes.

What Phase 1 does NOT include (comes in later plans):

- Dynamic candidate detection (Phase 2)
- Evidence gathering via OCR / Wikipedia / web search (Phase 2–3)
- Modified reconciler with Group A/B/C support (Phase 2–4)
- Dashboard integration of TTER results (Phase 5)
