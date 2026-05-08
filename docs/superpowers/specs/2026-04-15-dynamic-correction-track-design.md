# Dynamic Correction Track Design

**Date:** 2026-04-15
**Author:** Sanuka Thamuditha (FYP)
**Status:** Design approved — pending implementation plan
**Report reference:** `Sanuka_Report_Updated_v13.docx`

---

## 1. Problem Statement

The current ScreenApp ASR correction pipeline achieves improvements only on terms explicitly listed in `asr_correction/domain_vocab.json`. Two limitations follow from this:

1. **Cold-start problem.** Any domain term not pre-registered in the vocab file is never corrected, regardless of how strong the visual or contextual evidence is. Examples encountered during development: "Cloudware" → "Cloudflare", "Arkut" → "Argo CD", "rock" → "ReLU".
2. **Scope mismatch with FYP proposal.** The report defines **Target Term Error Rate (TTER)** as the primary evaluation metric — errors specifically on domain-relevant words including person names, AI model names, software/product names, technical jargon, and business terms. A closed vocab cannot cover this scope.

The pipeline itself (six steps: vocab merge → candidate detection → topic + web vocab → OCR → Whisper Pass 2 → LLM reconciliation) is sound. The bottleneck is the **vocab gate** in Step 6, which drops any correction whose target is not in the vocabulary.

This design adds a parallel **Dynamic Discovery Track** that proposes corrections for terms outside the vocab, validates them against external evidence (OCR, Wikipedia, web search), and merges results at reconciliation time. The existing vocab track is not modified. The dynamic track is controlled by a feature flag defaulting to off.

## 2. Goals and Non-Goals

### Goals

- Reduce TTER on SlideAVSR, AMI v2, and Earnings-22 evaluation datasets relative to the baseline ScreenApp transcript.
- Preserve existing behavior when the feature flag is off (no regressions on vocab-backed corrections).
- Produce auditable corrections — every applied edit records its evidence source, tier, and confidence.
- Support technical, business, and proper-noun term categories without per-category code paths.
- Stay within the existing pipeline architecture — no retraining, no new models, no new services.

### Non-goals

- No fine-tuning of Whisper, the reconciler LLM, or any other model. LoRA experiments remain a separate report chapter.
- No new training data collection.
- No real-time / streaming optimization. Latency improvements come after TTER improvement is proven offline.
- No general English WER reduction. WER is reported alongside TTER only to demonstrate the absence of regressions.
- No replacement of the vocab track. Vocab entries remain the authoritative "safe path" for known tech terms.

## 3. Architecture

The dynamic track runs in parallel with the existing vocab track, sharing common infrastructure (topic classification, OCR extraction, Whisper Pass 2) and merging at the reconciliation step.

```text
                      ┌─────────────────────────────────────────┐
                      │            Step 1: Vocab Merge           │
                      └─────────────────────────────────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                                                  ▼
    ┌───────────────────────┐                        ┌──────────────────────────┐
    │  VOCAB TRACK (exists) │                        │  DYNAMIC TRACK (NEW)     │
    │                       │                        │                          │
    │  Step 2a: LLM detects │                        │  Step 2b: Dynamic detect │
    │  vocab-matching errors│                        │  • Whisper conf < 0.6    │
    │                       │                        │  • LLM semantic out-    │
    │                       │                        │    of-place check        │
    │                       │                        │  • OCR mismatch flag     │
    └───────────┬───────────┘                        └────────────┬─────────────┘
                │                                                  │
                │      Step 3: Topic + Web Vocab (shared)          │
                │      Step 4: OCR Extraction (shared)             │
                │      Step 5: Whisper Pass 2 (shared)             │
                │                                                  │
                │                                                  ▼
                │                                ┌────────────────────────────────┐
                │                                │  Step 4a: Per-candidate Evidence│
                │                                │  (only for dynamic candidates) │
                │                                │                                 │
                │                                │  For each candidate, try:      │
                │                                │  1. OCR text match             │
                │                                │  2. Wikipedia entity lookup    │
                │                                │  3. Topic-scoped web search    │
                │                                │                                 │
                │                                │  → produces evidence_tier +    │
                │                                │    candidate_replacements      │
                │                                └────────────┬───────────────────┘
                │                                             │
                ▼                                             ▼
          ┌─────────────────────────────────────────────────────────┐
          │         Step 6: LLM Reconciliation (MODIFIED)            │
          │                                                          │
          │  Input: vocab candidates + dynamic candidates + evidence│
          │  Output: per-correction {target, confidence, tier}      │
          └─────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
          ┌─────────────────────────────────────────────────────────┐
          │         Step 6b: Tiered Filter (MODIFIED)                │
          │                                                          │
          │  Vocab candidates:   pass if target ∈ vocab              │
          │  Dynamic + hard-ev:  pass if confidence ≥ 0.70           │
          │  Dynamic + LLM-only: pass if confidence ≥ 0.95 AND       │
          │                      Whisper Pass 2 alt-token agrees    │
          └─────────────────────────────────────────────────────────┘
```

Corrections flow through the reconciler in three explicit groups (A = vocab-backed, B = dynamic with hard evidence, C = dynamic with LLM reasoning only), each with its own acceptance criteria.

## 4. Component Design

### 4.1 Step 2b — Dynamic Candidate Detection

**Module:** `asr_correction/dynamic_detector.py`

Produces a deduped list of suspicious spans that the vocab track did not flag, fed by three independent signal channels.

**Channel A — Whisper confidence scan.** A mechanical pass over `transcript.segments[*].words[*]`. A word becomes a candidate if its `probability` is below `config.whisper_conf_threshold` (default `0.60`) AND it passes the `is_content_word` filter (excludes articles, prepositions, and single-character tokens). Each candidate records `{text, start, segment_id, signal="whisper_low_conf", whisper_prob}`.

**Channel B — LLM semantic out-of-place check.** A single LLM call per transcript. The prompt receives the topic classification output from Step 3 and asks the model to identify words that look semantically odd, grammatically broken, or contextually inappropriate. The response is a list of `{text, start, reason}`. This channel catches confidently-wrong errors that Channel A misses — e.g. Whisper confidently outputs "rock" where "ReLU" is meant. Channel B is the most expensive signal (one LLM call) but cannot be skipped without losing the most interesting error class.

**Channel C — OCR cross-reference.** For each OCR frame extracted in Step 4, candidate entity tokens are identified using three rules applied in order: (1) any capitalized token of length ≥ 3 not in the top-5000 English words list, (2) any acronym matching `\b[A-Z]{2,}(?:-?\d+)?\b`, (3) any multi-word sequence where all tokens are capitalized. For each such token T:

- If T does not appear in the transcript within ±10 seconds of the OCR frame's timestamp, the nearest low-confidence transcript word in that window becomes a candidate with `suggested_target=T` and `signal="ocr_missed"`.
- If a phonetically similar but distinct word appears in the transcript window, it becomes a candidate with `suggested_target=T` and `signal="ocr_mismatch"`.

Channels merge by span overlap. A candidate flagged by multiple channels carries a `signals` list of all channels that matched — used later as a tie-breaker in reconciliation.

### 4.2 Step 4a — Per-candidate Evidence Gathering

**Module:** `asr_correction/evidence_gatherer.py`

For each dynamic candidate, evidence sources are tried in priority order and the first hit wins. The function returns `{tier, source, target, confidence, evidence_snippet}`.

**Tier 1 — OCR direct match.** For every OCR token, compute Jaro-Winkler similarity on Metaphone-encoded strings (via `jellyfish`) against the candidate. If similarity ≥ `config.phonetic_threshold` (default `0.65`), return with `source="ocr"` and a base confidence of `0.90`.

**Tier 2 — Wikipedia entity lookup.** The Wikipedia REST API search endpoint is hit with query `"{candidate_text} {topic}"`. The top N results (default 5) are scored by phonetic similarity to the candidate. A result passes if similarity ≥ `config.phonetic_threshold` AND the result's short description contains at least one word from the topic's keyword list (topic relevance gate). Confidence: `0.75 + 0.25 * similarity`.

**Tier 3 — Topic-scoped web search.** Query constructed as `"{candidate_text}" OR "{phonetic_variants}" {topic}`. Top results' titles and snippets are parsed with a noun-phrase extractor; phrases are scored by phonetic similarity. First match above threshold returns with confidence `0.70 + 0.20 * similarity`.

If no tier produces a match, the candidate is returned with `tier="llm_only"`, `target=None`, `confidence=0.0` and is deferred to reconciliation (Group C).

**Wikipedia client.** `asr_correction/wikipedia_client.py` wraps the Wikipedia REST API with a 5-second timeout, in-memory LRU cache (128 entries), and a fallback to the opensearch endpoint if the primary search fails. No authentication required.

### 4.3 Step 6 — Modified LLM Reconciliation

**Module:** `asr_correction/reconciler.py` (modified)

The reconciliation prompt is restructured to present candidates in three explicit groups:

```text
=== CORRECTION CANDIDATES ===

GROUP A — Vocab-backed (known domain terms):
  - "Kime" → "Kimi K2" (vocab entry, alias match)
  - "Quen" → "Qwen" (vocab entry, alias match)

GROUP B — Dynamic with hard evidence:
  - "Cloudware" → "Cloudflare"
    evidence: Wikipedia entity, phonetic 0.76, topic=DevOps
    suggested_confidence: 0.86

GROUP C — Dynamic with no hard evidence (needs your reasoning):
  - "rock" at 02:15 — flagged as semantically out-of-place in ML context
    whisper_alt_tokens: ["rock", "ReLU", "rack"]
    context: "...apply the rock function to each layer..."

For Group A: validate each against the segment context, output accept/reject.
For Group B: validate the target fits the context, output final confidence.
For Group C: propose a target if you are confident, output {target, confidence}.
```

The LLM returns a structured JSON list with `group`, `original`, `target`, `accept` (for A), and `final_confidence` (for B and C). The reconciler validates the output schema, logs malformed responses, and falls back to "reject all" on parse failure.

### 4.4 Tiered Filter

The existing `_filter_corrections` function is replaced with a tier-aware version:

```python
def _filter_corrections(corrections, vocab):
    applied = []
    for c in corrections:
        if c.group == "A":
            if c.target in vocab and c.accept:
                applied.append(c)
        elif c.group == "B":
            if c.final_confidence >= config.hard_evidence_threshold:
                applied.append(c)
        elif c.group == "C":
            if (c.final_confidence >= config.llm_only_threshold and
                c.target in c.whisper_alt_tokens):
                applied.append(c)
    return applied
```

Every applied correction carries `tier`, `source`, `confidence`, and `evidence_snippet` into the final correction report for auditability.

### 4.5 Confidence Formulas

**Hard-evidence tier (Group B):**

```text
conf = 0.5 * source_prior + 0.3 * phonetic_similarity + 0.2 * topic_relevance
```

- `source_prior`: OCR = 1.00, Wikipedia = 0.85, web search = 0.70
- `phonetic_similarity`: Jaro-Winkler on Metaphone, 0.0–1.0
- `topic_relevance`: 1.0 if candidate context matches topic keywords, 0.5 otherwise

**LLM-only tier (Group C):**

```text
conf = llm_stated_confidence * whisper2_agreement_factor
```

- `whisper2_agreement_factor`: 1.0 if Whisper Pass 2 alt-token list contains the proposed target, 0.5 otherwise

**Vocab tier (Group A):** no score needed — pass iff `target ∈ vocab`.

### 4.6 Target Term Extractor

**Module:** `asr_correction/target_term_extractor.py`

Given a ground-truth transcript, returns a set of target terms using four rules:

1. **Named entities** via spaCy NER with labels `PERSON`, `ORG`, `PRODUCT`, `WORK_OF_ART`, `GPE`.
2. **Proper nouns** (POS tag `PROPN`) not in a common-English word list.
3. **Acronyms** matching `\b[A-Z]{2,}(?:-?\d+)?\b` excluding function words (`THE`, `AND`, `FOR`, `YOU`).
4. **Technical jargon** matching a curated keyword list (e.g. `transformer`, `gradient`, `API`, `endpoint`).

Output is written to `{dataset}/target_terms/{file_id}.json` and is auditable from the dashboard. A hybrid approach is recommended: auto-extract all datasets, then manually spot-check and correct the Earnings-22 files (4 files, smallest set) for the final report.

### 4.7 TTER Metric

**Module:** `asr_correction/tter.py`

TTER is computed via `jiwer.process_words(ref, hyp)` followed by a walk over the alignment operations, counting edits only at positions where the reference word is in `target_terms`:

```text
TTER = (S_t + D_t + I_t) / N_t
```

Insertions are classified as target-term insertions only if the inserted word would itself survive the target term extractor (preventing trivial "inserted a preposition" false positives).

## 5. Configuration

New flags added to `asr_correction/config.py`:

| Flag                      | Default | Purpose                                       |
| ------------------------- | ------- | --------------------------------------------- |
| `enable_dynamic_track`    | `False` | Master switch for the entire dynamic track    |
| `whisper_conf_threshold`  | `0.60`  | Channel A candidate threshold                 |
| `phonetic_threshold`      | `0.65`  | Minimum phonetic similarity for evidence match|
| `hard_evidence_threshold` | `0.70`  | Group B acceptance confidence                 |
| `llm_only_threshold`      | `0.95`  | Group C acceptance confidence                 |
| `wikipedia_cache_size`    | `128`   | Wikipedia client LRU cache capacity           |
| `wikipedia_timeout_sec`   | `5`     | Wikipedia API timeout                         |
| `semantic_check_enabled`  | `True`  | Channel B on/off (LLM semantic check)         |

## 6. Evaluation Plan

### 6.1 Datasets

- **SlideAVSR (10 videos)** — primary evaluation. Slide-based content where OCR drives the dynamic track. Target terms: research paper names, model architectures, acronyms.
- **AMI v2 (10 meetings)** — secondary. Conversational, multi-speaker. Tests Wikipedia + web search and the LLM-only tier. Target terms: person names, product names.
- **Earnings-22 (4 calls)** — tertiary. Business / financial vocabulary. Tests Wikipedia company lookups and web search for financial jargon.
- **ScreenApp team recordings** — qualitative case study demonstrating real-world fit on internal meetings.

### 6.2 Metrics

The headline results table per dataset:

| Metric                              | Baseline | With Dynamic Track | Delta |
| ----------------------------------- | -------- | ------------------ | ----- |
| WER (overall)                       |          |                    |       |
| **TTER (target terms only)**        |          |                    |       |
| Corrections applied — vocab (A)     | 0        |                    |       |
| Corrections applied — hard-ev (B)   | 0        |                    |       |
| Corrections applied — LLM-only (C)  | 0        |                    |       |
| Precision on applied corrections    | —        |                    |       |
| Recall on target-term errors        | —        |                    |       |
| Avg latency per file                |          |                    |       |

TTER delta is the primary figure for RQ1. WER is reported alongside to demonstrate absence of regressions.

### 6.3 Evaluation script

`scripts/evaluation/run_tter_eval.py` takes a dataset argument and runs end-to-end:

1. Load ground truth and ScreenApp baseline transcripts.
2. Extract target terms (cached after first run) → save `target_terms.json`.
3. Compute baseline TTER.
4. Run the correction pipeline with `enable_dynamic_track=False` → compute vocab-only TTER.
5. Run the correction pipeline with `enable_dynamic_track=True` → compute full TTER.
6. Compute per-tier correction counts, precision, and recall.
7. Append a row to `data/eval_results_v3/per_file.csv`.
8. Update `data/eval_results_v3/summary.json`.

### 6.4 Dashboard integration

A new section on `frontend/src/pages/Eval.jsx` reads `/api/eval/tter_results` and displays per-dataset TTER improvement with per-tier correction breakdown. No new pages.

## 7. Implementation Order

Each phase is independently testable. Phase boundaries are commit points.

**Phase 1 — Measurement infrastructure.**

1. `target_term_extractor.py` + unit test on SlideAVSR ground truth.
2. `tter.py` + unit test on a hand-crafted example.
3. `run_tter_eval.py` producing baseline TTER for all three datasets.

This phase alone enables the "current state" section of the report with real TTER baselines.

**Phase 2 — Dynamic detection (cheapest signals first).**

4. `dynamic_detector.py` — Channel A only.
5. `evidence_gatherer.py` — Tier 1 (OCR) only.
6. `confidence_scorer.py` — hard-evidence formula only.
7. `reconciler.py` — Group A + Group B support; hard-evidence filter.
8. Run evaluation; expected SlideAVSR TTER drop of 2–4 points.

**Phase 3 — External evidence sources.**

9. `wikipedia_client.py` + Tier 2 Wikipedia lookup in `evidence_gatherer.py`.
10. Run evaluation; expected AMI TTER drop of 2–3 points.
11. Tier 3 web search per candidate.
12. Run evaluation; expected Earnings-22 TTER drop of 1–3 points.

**Phase 4 — LLM-only tier.**

13. `dynamic_detector.py` — add Channel B (LLM semantic check).
14. `reconciler.py` — Group C handling + LLM-only filter (0.95 threshold + Whisper Pass 2 agreement).
15. Run evaluation. If precision drops, raise threshold to 0.97 or disable Channel B via config.

**Phase 5 — Dashboard and final evaluation.**

16. `/api/eval/tter_results` endpoint + Eval.jsx TTER section.
17. Final evaluation run across all datasets; generate report tables.
18. Write the results chapter of the report.

## 8. Rollout Safety

- The entire dynamic track is gated behind `config.enable_dynamic_track`, default `False`. When off, pipeline behavior is identical to today.
- `run_tter_eval.py` toggles the flag per run and produces baseline vs dynamic results side-by-side.
- Each phase is a separate git commit so regressions can be bisected.
- Every applied correction records its tier, source, confidence, and evidence snippet in the correction report — enabling post-hoc analysis and debugging.

## 9. Open Questions

- Target term extraction is auto-generated via NER rules. For the final report, manual spot-checking of the Earnings-22 files (smallest dataset) is recommended. Whether to extend manual annotation to other datasets depends on supervisor rigor expectations.
- The LLM-only tier threshold of 0.95 is a starting point. It may need tuning based on Phase 4 evaluation results. The design allows raising the threshold or disabling Channel B entirely via config without code changes.
- Latency impact of per-candidate Wikipedia and web search calls is unknown until Phase 3 evaluation. If it exceeds acceptable bounds for near-real-time operation (RQ2), evidence calls can be batched or backgrounded in a follow-up design.

## 10. Out of Scope

- Model fine-tuning (LoRA experiments remain a separate chapter).
- New training data collection.
- Streaming / real-time latency optimization.
- General English WER reduction.
- Replacement of the existing vocab track.
