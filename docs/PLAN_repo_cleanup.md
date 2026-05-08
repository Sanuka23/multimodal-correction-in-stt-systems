# Repo Cleanup — "Open the repo and see the code easily"

**Status:** Plan only · do not execute yet
**Branch:** `chore/repo-cleanup`
**Goal:** When someone opens this repo, what they see should be **only the code that runs in production today**. Anything else either gets deleted, hidden in `docs/archive/`, or moved out of the way.

---

## THE TEST

When you `cd` into the repo and run `ls`, you should see at most ~10 things. When you `ls` into any source directory, every file you see should be *imported by something that runs in production*. If a file's only consumer is a script that no one runs anymore, the file goes.

---

## BEFORE / AFTER — REPO ROOT

### Today (21 visible items at root)
```
DASHBOARD_PROMPT.md          ← stale brief
README.md
Roadmap for OCR and AVSR ... ← superseded
WEB_VOCAB_PROMPT.md          ← stale brief
app/
archive/                     ← legacy code
asr_correction/
data/                        ← gitignored
docs/
evaluation/
frontend/
logs/                        ← empty
models/                      ← gitignored, 1 GB local
pytest.ini
requirements.txt
run.sh
scripts/
static/                      ← legacy charts.js for Jinja2
templates/                   ← Jinja2 templates, replaced by React
tests/                       ← almost empty
training/
vision-key.json              ← credential, gitignored
```

### After cleanup (10 visible items)
```
README.md
app/                         ← FastAPI backend
asr_correction/              ← Pipeline library
docs/                        ← All planning + archive
evaluation/                  ← WER / TTER / compare
frontend/                    ← React dashboard
pytest.ini
requirements.txt
run.sh
training/                    ← Notebooks + scripts + figures
```

`scripts/` keeps the eval/training tooling but moves under `training/scripts/` and `evaluation/scripts/` (next to the code they exercise). `tests/` either grows up or goes; if it stays empty after this pass, delete.

---

## BEFORE / AFTER — `asr_correction/`

### Today (28 items, several dead)
```
__init__.py            ✓ live (1429 lines, the orchestrator)
adapters/              gitignored
adapters_qwen35/       gitignored
adapters_qwen35_v2/    gitignored
adapters_v2/           gitignored
avsr/                  ✓ live (mediapipe + auto_avsr)
collected_data/        gitignored
config.py              ✓ live
data_collector.py      ✓ live
domain_vocab.json      ✓ live
dynamic_detector.py    ✗ dead (only run_dynamic_eval.py uses it)
evidence_gatherer.py   ✗ dead (only run_dynamic_eval.py uses it)
llm_detector.py        ✓ live
llm_semantic_detector.py ✗ dead
model.py               ✓ live
model_weights/         gitignored config-only dir
ocr_extractor.py       ✓ live
ocr_parser.py          ✓ live
reconciler.py          ✓ live
requirements.txt       ✗ duplicate (the root one already covers everything)
segment_selector.py    ⚠ fallback only (inline + delete)
target_term_extractor.py ✓ live (used by tter)
technical_keywords.py  ✓ live (used by tter)
text_utils.py          ✓ live
tter.py                ⚠ duplicate of evaluation/tter.py
types.py               ✓ live
video_frames.py        ✓ live
vocabulary.py          ✓ live
whisper_pass2.py       ✓ live
wikipedia_client.py    ✓ live (used by web vocab v2 plan)
```

### After cleanup (16 visible files + 2 subdirs)
```
__init__.py
avsr/
config.py
data_collector.py
domain_vocab.json
llm_detector.py
model.py
ocr_extractor.py
ocr_parser.py
reconciler.py
target_term_extractor.py
technical_keywords.py
text_utils.py
types.py
video_frames.py
vocabulary.py
whisper_pass2.py
wikipedia_client.py
```

Everything you see imports something else you see, or is imported by `__init__.py`. No archaeology.

---

## BEFORE / AFTER — `app/routes/`

### Today
```
correction.py         ✓ live
dashboard.py          ⚠ half live (api/* live, /dashboard/* Jinja2 dead)
evaluation.py         ✓ live
health.py             ✓ live
pipeline_settings.py  ✓ live
training.py           ✓ live
```

### After cleanup
```
correction.py
dashboard.py          ← only the JSON /api/* endpoints remain
evaluation.py
health.py
pipeline_settings.py
training.py
```

Plus `templates/` and `static/` directories disappear from the repo root.

---

## THE ACTUAL CHANGES

Six commits, in this order. Each one is small and reverible.

### Commit 1 — Hide planning briefs
```
mkdir -p docs/archive
git mv DASHBOARD_PROMPT.md docs/archive/
git mv WEB_VOCAB_PROMPT.md docs/archive/
git mv "Roadmap for OCR and AVSR Extensions to the Multimodal STT Correction Framework.md" \
       docs/archive/Roadmap_OCR_AVSR_v1.md
```
**What you'll see:** repo root drops from 21 to 18 visible items.

### Commit 2 — Delete `archive/`
```
git rm -r archive/
```
**What you'll see:** 17 visible items. The directory was for legacy pre-v2 code; git history preserves it.

### Commit 3 — Tighten `.gitignore` + untrack noise
```
# Append to .gitignore:
node_modules/
frontend/dist/
.ipynb_checkpoints/
.pytest_cache/
.coverage
data/pipeline_settings.json
asr_correction/adapters*/

# Untrack:
git rm --cached .DS_Store
git rm --cached -r .pytest_cache 2>/dev/null || true
git rm --cached asr_correction/adapters/adapter_config.json 2>/dev/null || true
git rm --cached asr_correction/adapters/training_metadata.json 2>/dev/null || true
```
**What you'll see:** `git status` stops nagging about generated files.

### Commit 4 — Retire the parallel Jinja2 dashboard

The React app at `frontend/` already serves every page that the Jinja2 templates cover. Keep the JSON `/api/*` endpoints (the React app needs them); delete the HTML-rendering ones.

```python
# app/routes/dashboard.py — delete these route handlers:
#   @router.get("/dashboard")
#   @router.get("/dashboard/compare")
#   @router.get("/dashboard/jobs")
#   @router.get("/dashboard/pipeline/{job_id}")
#   @router.get("/dashboard/training")
# Remove templates init + Jinja2Templates import.
# Keep all @router.get("/api/*") handlers.
```

```bash
git rm -r templates/
git rm -r static/
# Drop jinja2 from requirements.txt
```
**What you'll see:** repo root drops by 2 directories. Backend has *one* surface — the React app.

### Commit 5 — Retire the v3 dynamic eval + its detector dependencies

`scripts/evaluation/run_dynamic_eval.py` is the only thing that imports `dynamic_detector.py`, `evidence_gatherer.py`, `llm_semantic_detector.py`. Its output (`data/eval_results_v3/`) is already frozen in the dissertation. Freeze it on purpose:

```bash
# Note the freeze
echo "v3 dynamic-evidence eval results are frozen at the dataset+code as of \
commit $(git rev-parse HEAD~1). Re-running requires reverting this commit." \
> docs/archive/eval_v3_freeze.md

# Delete script + its dependencies
git rm scripts/evaluation/run_dynamic_eval.py
git rm asr_correction/dynamic_detector.py
git rm asr_correction/evidence_gatherer.py
git rm asr_correction/llm_semantic_detector.py
```

**What you'll see:** `asr_correction/` loses three legacy modules totalling ~1,100 LOC. Nothing in production breaks because nothing imports them.

### Commit 6 — Inline `segment_selector` fallback into `llm_detector`

`llm_detector.py` imports two things from `segment_selector`:
- `SegmentAnalysis` dataclass
- `select_segments` rule-based fallback

Both move directly into `llm_detector.py`. The 506-line `segment_selector.py` had many code paths the fallback doesn't need; inline only the ~50 LOC actually used.

```bash
# Move SegmentAnalysis dataclass + minimal select_segments_rules() to llm_detector.py
# Delete segment_selector.py
git rm asr_correction/segment_selector.py
```

**What you'll see:** `asr_correction/` loses one more file. The fallback is still there, just colocated.

### Commit 7 — Drop the per-package `requirements.txt`

`asr_correction/requirements.txt` exists alongside the root `requirements.txt`. Two requirements files for the same package is a confusion source. Verify the root file covers it, then:
```bash
git rm asr_correction/requirements.txt
```

### Commit 8 — Move `tter.py` out (de-duplicate)

There are two TTER implementations:
- `evaluation/tter.py` — used by the API (`POST /api/evaluate`)
- `asr_correction/tter.py` — used by `target_term_extractor.py`

Pick `evaluation/tter.py` as canonical; have `target_term_extractor.py` import from there. Delete `asr_correction/tter.py`.

```bash
# Edit asr_correction/target_term_extractor.py:
#   from .tter import normalize, find_occurrences   →   from evaluation.tter import normalize, find_occurrences
git rm asr_correction/tter.py
```

### Commit 9 — Frontend lazy-load (drops the bundle warning)

`App.jsx` currently imports all 9 pages eagerly → ~890 kB main bundle. Lazy-load them:

```jsx
// frontend/src/App.jsx
import { lazy, Suspense } from 'react'

const Dashboard       = lazy(() => import('./pages/Dashboard'))
const Jobs            = lazy(() => import('./pages/Jobs'))
const Pipeline        = lazy(() => import('./pages/Pipeline'))
const Compare         = lazy(() => import('./pages/Compare'))
const Eval            = lazy(() => import('./pages/Eval'))
const Training        = lazy(() => import('./pages/Training'))
const TrainingData    = lazy(() => import('./pages/TrainingData'))
const PipelineControl = lazy(() => import('./pages/PipelineControl'))
const Annotate        = lazy(() => import('./pages/Annotate'))

// Wrap each <Route element={...} /> in <Suspense fallback={null}>.
```

**What you'll see:** `npm run build` no longer warns about chunk size. Main bundle ≈ 250 kB; pages load on demand.

### Commit 10 — Delete `frontend/src/components/pipeline/PipelineConnector.jsx`

Replaced by `PipelineFlow.jsx`. Self-referenced only.

```bash
git rm frontend/src/components/pipeline/PipelineConnector.jsx
```

### Commit 11 — Empty-dir sweep

```bash
# logs/ — empty directory. If pytest doesn't care, delete it.
[ -z "$(ls logs/)" ] && rmdir logs/

# tests/ — only fixture stubs. If you don't run pytest yet, move pytest.ini and the
# stub tests under app/tests/ or delete tests/ entirely. (Decide based on intent.)
```

---

## ONE-PASS SCRIPT

Once the above is reviewed, this is the whole sequence:

```bash
git checkout chore/repo-cleanup

# Commit 1
mkdir -p docs/archive
git mv DASHBOARD_PROMPT.md docs/archive/
git mv WEB_VOCAB_PROMPT.md docs/archive/
git mv "Roadmap for OCR and AVSR Extensions to the Multimodal STT Correction Framework.md" \
       docs/archive/Roadmap_OCR_AVSR_v1.md
git commit -m "chore: move planning briefs into docs/archive"

# Commit 2
git rm -r archive/
git commit -m "chore: delete pre-v2 archive/ — git history preserves it"

# Commit 3
# (edit .gitignore in editor)
git rm --cached .DS_Store
git commit -am "chore: tighten .gitignore + untrack generated files"

# Commit 4
# (edit app/routes/dashboard.py + app/main.py to drop Jinja2)
git rm -r templates/ static/
# (edit requirements.txt to drop jinja2)
git commit -am "chore: retire Jinja2 dashboard — React app is the only UI"

# Commit 5
echo "v3 dynamic-evidence eval frozen at $(git rev-parse HEAD~1)" > docs/archive/eval_v3_freeze.md
git rm scripts/evaluation/run_dynamic_eval.py
git rm asr_correction/{dynamic_detector,evidence_gatherer,llm_semantic_detector}.py
git add docs/archive/eval_v3_freeze.md
git commit -m "chore: retire v3 dynamic eval + its three legacy detector modules"

# Commit 6
# (move SegmentAnalysis + select_segments_rules into llm_detector.py)
git rm asr_correction/segment_selector.py
git commit -am "chore: inline segment_selector fallback into llm_detector"

# Commit 7
git rm asr_correction/requirements.txt
git commit -m "chore: drop duplicate requirements.txt under asr_correction/"

# Commit 8
# (edit target_term_extractor.py to import from evaluation/tter)
git rm asr_correction/tter.py
git commit -am "chore: de-duplicate tter — evaluation/tter is canonical"

# Commit 9
# (edit frontend/src/App.jsx to lazy-load pages)
git commit -am "chore: lazy-load page components — drops main bundle from 890 to ~250 kB"

# Commit 10
git rm frontend/src/components/pipeline/PipelineConnector.jsx
git commit -m "chore: remove deprecated PipelineConnector.jsx"

# Commit 11 — only if logs/ is empty
[ -z "$(ls logs/)" ] && git rm -r logs/ && git commit -m "chore: drop empty logs/"
```

---

## SAFETY GATES

After every commit, run:
```bash
python3 -c "from asr_correction import correct_transcript; print('ok')"
uvicorn app.main:app --port 8000 &; sleep 2
curl -s http://localhost:8000/api/health | grep -q ok && echo "backend OK"
kill %1
( cd frontend && npm run build > /tmp/build.log 2>&1 && echo "frontend OK" )
```

If any of those fail, `git revert HEAD` and investigate.

---

## WHAT THIS WON'T DO

- **No refactor of live code.** Pure subtraction + colocation. Every file that runs in production stays exactly where it is.
- **No data deletion.** `data/`, `models/`, adapter dirs are local-only and untouched.
- **No notebook changes.** `training/finetune_qwen35.ipynb` and the figures stay.

---

## EXPECTED FINAL STATE

When you `ls` the repo root, you see this:
```
README.md
app/
asr_correction/
docs/
evaluation/
frontend/
pytest.ini
requirements.txt
run.sh
training/
```

When you `ls asr_correction/`, you see only 17 files + 1 subdir (avsr) — every one of which is imported by `__init__.py` directly or transitively.

When you `npm run build`, no warnings.

That's the whole goal.

---

*End of plan. No code touched yet.*
