# Backend Workflow — End to End

**Project:** Multimodal Correction in Speech-to-Text Systems (FYP)
**Codebase:** `multimodal-correction-in-stt-systems/`
**Scope:** What happens from the moment a request arrives at the FastAPI server until the JSON response is sent back, with every file and function involved.

This document traces a single `POST /asr-correct` request — the main correction endpoint — start to finish. Other endpoints follow the same pattern with smaller pipelines.

---

## STAGE 0 — Server Boot (one time, before any request)

When `uvicorn app.main:app --port 8000` runs:

### 1. `app/main.py` is imported
- Reads `app.config.get_settings()` (via lifespan) — pulls env vars from `.env`.
- Creates the `FastAPI()` instance.
- Mounts CORS middleware.
- Calls `app.include_router(...)` six times:
  - `health.router` → `GET /health`
  - `correction.router` → `POST /asr-correct`, `/asr-analyze`
  - `evaluation.router` (prefix `/api`) → `/api/evaluate`, `/api/corrections`, dataset endpoints
  - `training.router` (prefix `/api`) → `/api/train`, `/api/training/*`
  - `dashboard.router` → `/api/jobs`, `/api/stats`, `/api/health`, `/api/jobs/{id}/steps`
  - `pipeline_settings.router` → `/api/pipeline/settings`

### 2. `lifespan` context fires
- Calls `app.database.connect_db()` — opens a Motor (async MongoDB) connection to the URL in `.env`. Creates a global `_client` singleton.

### 3. Server is now ready
No models are loaded yet. They lazy-load on the first correction request.

---

## STAGE 1 — Request Arrives

ScreenApp's Node backend (`enhanceTranscript.ts`) makes:

```http
POST /asr-correct HTTP/1.1
Authorization: Bearer <jwt>
Content-Type: application/json

{
  "file_id": "abc-123",
  "transcript": { "text": "...", "segments": [...] },
  "custom_vocabulary": "ChartMogul\nQwen\n",
  "video_url": "https://storage/.../video.mp4"
}
```

FastAPI's router matches the path to `app/routes/correction.py :: asr_correct(...)`.

---

## STAGE 2 — Auth + Validation (`app/auth.py`, `app/routes/correction.py`)

### `app/auth.py :: get_jwt_info(authorization: str) -> str`
- Extracts the `Bearer <jwt>` token.
- Verifies the signature using `SESSION_SECRET` (shared with ScreenApp's Node backend).
- Decodes the payload, returns the `user_id` claim.
- On any failure → raises `HTTPException(401)`. The pipeline never runs.

### `app/routes/correction.py`
- Pydantic auto-validates the request body against the `ASRCorrectionRequest` schema:
  ```python
  class ASRCorrectionRequest(BaseModel):
      transcript: Dict[str, Any]
      file_id: str
      custom_vocabulary: Union[str, List[str]] = ""
      ocr_xml: Optional[str] = None
      video_url: Optional[str] = None
  ```
- If the body is malformed → 422 automatically. No code runs further.

---

## STAGE 3 — Job Creation (`app/database.py`)

Inside `asr_correct(...)`:

### `app.database.create_job(job_type, file_id, input_summary) -> str`
- Inserts a fresh row into MongoDB collection `jobs` with:
  ```json
  {
    "_id": ObjectId,
    "job_type": "correction",
    "file_id": "abc-123",
    "status": "running",
    "created_at": ISODate,
    "pipeline_steps": [],
    "input_summary": { "file_id": "...", "user_id": "..." }
  }
  ```
- Returns the `job_id` (string form of the ObjectId).

### `app.database.update_job_step(job_id, "request_received", "completed", ...)`
- Adds the first row to `pipeline_steps[]`.
- This is the first event the dashboard's Pipeline Monitor will see when it polls.

---

## STAGE 4 — Build the Step Bridge (`app/routes/correction.py`)

The pipeline is a **synchronous** function (it has to run on the MLX backend in a worker thread). But MongoDB writes are **async** coroutines. The bridge connects them:

### `_make_step_bridge(loop, job_id) -> Callable`
Returns a sync callable that, when called from the worker thread:
1. Schedules `update_job_step(...)` on the main asyncio loop via `asyncio.run_coroutine_threadsafe(...)`.
2. **Blocks** the worker thread on `future.result(timeout=5)` until the Mongo write confirms.
3. This blocking is critical — it preserves event order, so a `running → completed` sequence for the same step name never races into a duplicate row.

---

## STAGE 5 — Apply Operator Toggles (`app/services/pipeline_settings.py`)

### `pipeline_settings.get_settings() -> dict`
- Reads `data/pipeline_settings.json` (file on disk).
- Falls back to defaults if the file is missing.

### `pipeline_settings.apply_to_config(config)`
- Mutates the freshly-built `CorrectionConfig` dataclass with operator overrides:
  - `enable_topic_classification`, `enable_web_vocab_enrichment`, `enable_candidate_validation`
  - `enable_ocr_extraction`, `enable_ocr_vocab_extraction`, `enable_whisper_pass2`, `enable_avsr`, `enable_data_collection`
  - `avsr_mode`, `avsr_run_on_all_flagged`, `avsr_min_speaking_confidence`

The Kinetic Console's **Pipeline Control** page writes this file. So toggling AVSR off in the UI affects the next request immediately — no restart needed.

---

## STAGE 6 — Run the Pipeline (worker thread)

### `app.routes.correction._run_in_thread(_run_correction)`
- Schedules `correct_transcript(...)` on the FastAPI thread pool so the asyncio loop stays responsive.

### `asr_correction/__init__.py :: correct_transcript(transcript, file_id, custom_vocabulary, video_url, config, step_callback)`

This is the orchestrator. It runs **14 internal stages**, emitting an event before and after each one. Below: every stage, its file, its function, and what it produces.

---

### Stage 6.1 — `vocab_merge`

**File:** `asr_correction/vocabulary.py`

```python
def load_domain_vocab(path) -> dict           # reads asr_correction/domain_vocab.json (199 terms)
def merge_vocabularies(custom, domain) -> list  # case-insensitive dedup, unions known_errors
```

**Output:** A list of `{term, category, known_errors}` dicts (~200–230 entries).

---

### Stage 6.2 — `model_load`

**File:** `asr_correction/model.py`

```python
def detect_backend() -> str                                    # "mlx" on Apple Silicon
def load_model(adapter_path, model_path, base_model, backend)  # singleton-cached
```

- First call: downloads `mlx-community/Qwen3.5-9B-MLX-4bit` from HuggingFace (~5.2 GB), loads into MLX memory.
- Subsequent calls in the same process: returns the cached `(model, tokenizer)` instantly.

---

### Stage 6.3 — `candidate_detection`

**File:** `asr_correction/llm_detector.py`

```python
def detect_errors(transcript, vocab_terms, model, tokenizer, config) -> List[SegmentAnalysis]
```

Internal flow:
1. Chunks the transcript at 5 000 chars (200 char overlap).
2. Builds the detection prompt — sends each chunk + vocab list to Qwen.
3. Parses `{"suspects":[{"word":"QME","likely_correct":"Qwen"}, ...]}`.
4. If the LLM fails → falls back to `asr_correction/segment_selector.py :: select_segments_rules(...)` (rule-based scanner over the same vocab).

**Output:** `SegmentAnalysis` objects per transcript segment, with `candidates: List[CorrectionCandidate]` for every flagged word.

---

### Stage 6.4 — `topic_classification`

**Function:** `__init__.py :: _classify_topic(transcript, vocab_terms, candidates, model, tokenizer)`

Sends the transcript + first 30 vocab terms + first 15 detected errors to Qwen with a system prompt:
> "You are a meeting transcript analyzer. Classify the meeting by field and topic, and suggest domain-specific vocabulary."

Returns `{field, topic, description, suggested_vocab}` — e.g.
```json
{
  "field": "tech",
  "topic": "ASR / speech recognition systems",
  "description": "Discussion of multimodal ASR correction architecture...",
  "suggested_vocab": ["diarization", "pyannote", "Whisper", "WER", "VAD", "ASR", "Qwen"]
}
```

---

### Stage 6.5 — `web_vocab_enrichment`

**Function:** `__init__.py :: _enrich_vocab_from_web(topic_info, model, tokenizer)`

1. Builds two DuckDuckGo queries from the topic.
2. Calls `ddgs.DDGS().text(query, max_results=5)` for each — pulls up to 10 result snippets.
3. Truncates the concatenated text on the last sentence boundary at 2 500 chars (`_truncate_at_sentence`).
4. Sends snippets to Qwen with the extraction prompt.
5. Parses the JSON array (`_parse_web_vocab_json` — handles Markdown fences + truncated arrays).
6. If first parse fails → one self-repair retry: sends the malformed output back asking Qwen to fix it.
7. Quality gate: drops empty/duplicate/short/wrong-category terms.

**Output:** `[{term, category}, ...]` — typically 10–20 terms, often catches things the operator never put in custom vocabulary.

---

### Stage 6.6 — `candidate_validation`

**Function:** `__init__.py :: _validate_candidates_via_web(candidates, vocab_terms, topic_info)`

Two passes:

**Pass 1 — Cross-chunk pooling.** When the LLM detector chunks the transcript, it may propose different corrections for the same error word in different chunks (e.g. `Post-Sog → Post-SOC` in chunk 5, `Post-Talk → Post-Hog` in chunk 6). This pass collects all proposed targets and uses `jellyfish.jaro_winkler_similarity()` (threshold 0.75) to pick the best phonetic match.

**Pass 2 — Web fallback.** For up to 8 unresolved candidates, runs DuckDuckGo searches with the error word + topic + "product software", regex-extracts PascalCase entities, overrides the candidate if similarity > 0.75 AND beats the original proposal.

**Output:** Mutates `candidates` list in place.

---

### Stage 6.7 — `ocr_extraction`

**Files:** `asr_correction/video_frames.py`, `asr_correction/ocr_extractor.py`

```python
# video_frames.py
def get_video_duration(video_url) -> float        # ffprobe
def extract_frames_periodic(video_url, interval_s, max_frames) -> List[ExtractedFrame]
def extract_frames_at_timestamps(video_url, timestamps) -> List[ExtractedFrame]

# ocr_extractor.py
def _ocr_single_frame(frame, min_confidence) -> Dict   # PaddleOCR on a numpy array
```

Flow:
1. `_compute_ocr_timestamps` (in `__init__.py`) picks 15 frame timestamps — 2/3 at flagged-segment midpoints, 1/3 evenly spaced across the video.
2. `extract_frames_at_timestamps` runs `ffmpeg -ss T -i URL -vframes 1 -f image2pipe -` per timestamp, decodes each frame with OpenCV.
3. `_ocr_single_frame` runs PaddleOCR on each frame.
4. Filters URLs, JSON-looking junk, lines > 80 chars or < 4 chars.

**Output:** A list of unique OCR text snippets — `["ChartMogul Dashboard", "Andre Dean Smith (Presenting)", "app.chartmogul.com", ...]`.

---

### Stage 6.8 — `ocr_vocab_extraction`

**Function:** `__init__.py :: _extract_vocab_from_ocr(ocr_hints, model, tokenizer)`

Sends up to 150 unique OCR lines to Qwen:
> "Extract person names, product/tool names, and company names from this screen text. Ignore UI text, URLs, emails. Respond with JSON array: `[{"term":"X","type":"person|product|company|tool"}]`."

**Output:** Structured terms like `[{"term":"Andre","type":"person"}, {"term":"ChartMogul","type":"product"}]`. These become **PROTECTED TERMS** — the reconciler is forbidden from changing their spelling later.

---

### Stage 6.9 — `whisper_pass2`

**File:** `asr_correction/whisper_pass2.py`

```python
def build_initial_prompt(ocr_names, custom_vocab, topic_vocab, web_vocab, max_chars=896) -> str
def retranscribe_flagged_segments(video_url, flagged_segments, initial_prompt, config) -> dict
```

Flow:
1. `build_initial_prompt` packs all known names + vocab into a Whisper-format prompt (≤ 896 chars):
   ```
   Speakers: Andre, Avindi. Key terms: ChartMogul, Qwen, pyannote. Topics: diarization, VAD.
   ```
2. For each flagged segment time range:
   - `extract_audio_segment` runs `ffmpeg -ss S -i URL -t D -ar 16000 -ac 1 -f wav OUT.wav` (16 kHz mono).
   - Loads `faster-whisper small` (cached singleton, `device=cpu`, `compute_type=int8`).
   - Runs transcription with `initial_prompt=...` so Whisper biases toward our vocabulary.
3. Returns `{(start, end): "transcribed text"}` — Version B for the reconciler.

---

### Stage 6.10 — `avsr_extraction`

**Files:** `asr_correction/avsr/__init__.py`, `asr_correction/avsr/mediapipe_hints.py`

```python
# avsr/__init__.py
def get_avsr_provider(mode="mediapipe") -> AVSRProvider

# avsr/mediapipe_hints.py
class MediaPipeHintProvider:
    def analyze_segment(video_url, start_s, end_s) -> AVSRHint
```

Flow (orchestrated by `__init__.py :: _gather_avsr_hints`):
1. Collects AVSR-eligible segments — ones with `person_name / content_word / custom` candidates (or all flagged segments if `avsr_run_on_all_flagged=True`).
2. For each segment (capped at 20):
   - Extracts ~12 frames from the segment time range.
   - Detects faces via MediaPipe Face Mesh.
   - Computes Mouth Aspect Ratio (MAR) variance per face.
   - Picks the active speaker (face with highest MAR variance).
3. Drops hints below `min_speaking_confidence` (default 0.55).

**Output:** `[{start, end, hint, confidence, lip_transcript}, ...]` — third-opinion hints for the reconciler. MediaPipe doesn't produce a real lip transcript; the optional Auto-AVSR mode does.

---

### Stage 6.11 — `llm_reconciliation`

**File:** `asr_correction/reconciler.py`

```python
def reconcile_segments(
    original_transcript, whisper_segments, vocab_terms, ocr_hints, model, tokenizer, config,
    protected_terms=None, error_candidates=None, topic_info=None,
    web_vocab=None, avsr_hints=None,
) -> Tuple[dict, list]
```

This is the **heart of the system**. For each transcript segment that has a Whisper Pass 2 counterpart, it builds a single comprehensive prompt:

```
Meeting context: tech — ASR / speech recognition systems

Version A (original):       Yeah we benchmarked QME against Kimi K2 on Diversation.
Version B (re-transcription): Yeah we benchmarked Qwen against Kimi K2 on diarization.

Known vocabulary (correct spellings): ["ChartMogul", "Qwen", "pyannote", ...]

Suspected ASR errors detected in this segment:
  - 'QME' is likely 'Qwen' (ai_model)
  - 'Diversation' is likely 'diarization' (tech_term)

PROTECTED TERMS (confirmed by on-screen text):
ChartMogul, Andre

Screen text visible during this segment (from OCR):
  - Qwen 2.5 Coder
  - Kimi K2 chat completions

Visual lip-reading hint (third opinion, ~19% WER — tiebreaker only):
  - Speaker actively speaking (confidence: 0.71)

Rules:
1. Start with Version A as the base text.
2. Pick the best word from EITHER version, OR from vocabulary.
3. Each swap must be exactly ONE word → ONE word.
4. NEVER add or remove words.
5. NEVER override a PROTECTED term.
... (9 rules total)

Respond with JSON: {"text": "...", "swaps": ["wrong → correct", ...]}
```

The LLM cannot invent words — it can only choose between Version A, Version B, or a vocabulary term. This eliminates the hallucination class entirely.

**Output:** `(enhanced_transcript_dict, list_of_swaps)`.

---

### Stage 6.12 — Build `CorrectionReport`

**File:** `asr_correction/types.py`

Constructs the report dataclass:
```python
CorrectionReport(
    file_id=...,
    corrections_attempted=...,
    corrections_applied=...,
    results=[CorrectionResult(...), ...],
    processing_time_ms=...,
    selector_info={...},
    topic_info={...},
    evidence_sources={"vocab":202, "ocr":47, "ocr_vocab":8, "web_vocab":11, "avsr":0},
)
```

---

### Stage 6.13 — `data_collection`

**File:** `asr_correction/data_collector.py`

```python
def collect_correction_data(results, system_prompt, output_dir) -> None
```

Appends each applied correction as a ChatML JSONL row to `data/collected_data/train.jsonl`. This builds the corpus for future LoRA fine-tunes.

---

### Stage 6.14 — `complete` event emitted

`__init__.py` calls `_emit("complete", "completed", total_ms, {evidence_sources, ...})` and returns:

```python
return enhanced_transcript_dict, correction_report
```

---

## STAGE 7 — Persist Results (`app/database.py`)

Back in `correction.py :: asr_correct(...)`:

### `save_correction(file_id, original_text, enhanced_text, corrections_detail, report_summary)`
- Inserts into MongoDB `corrections` collection.
- Used later by the Compare page (`useCorrections(50)` query).

### `update_job_step(job_id, "apply_corrections", "completed", ...)`
- Records the save step.

### `update_job_step(job_id, "complete", "completed", duration_ms, details)`
- Records the final step (umbrella).

### `complete_job(job_id, total_duration_ms, result_summary)`
- Updates the parent job document: `status="completed"`, `completed_at`, `duration_ms`, `result_summary={corrections_applied, corrections_attempted}`.

---

## STAGE 8 — Send Response

```python
return {
    "enhanced_transcript": enhanced_transcript_dict,
    "correction_report": {
        "file_id": "...",
        "corrections_attempted": 21,
        "corrections_applied": 17,
        "processing_time_ms": 694224,
        "results": [...],
        "evidence_sources": {"vocab":202, "ocr":47, "ocr_vocab":8, "web_vocab":11, "avsr":0}
    }
}
```

FastAPI serialises to JSON and sends `HTTP 200` back to ScreenApp.

ScreenApp's Node backend (`enhanceTranscript.ts`) saves the enhanced transcript to S3, updates the file's `textData` record in its own MongoDB.

---

## STAGE 9 — Live Dashboard Polling (parallel track)

While stages 6.1 → 6.14 are running, the operator can watch progress:

1. They open `http://localhost:5173/pipeline/<job_id>` in the React Kinetic Console.
2. The page mounts `Pipeline.jsx`, which calls `useJobSteps(jobId)`.
3. `useJobSteps` is a React Query hook that polls **`GET /api/jobs/{job_id}/steps`** every 2 seconds while a step is running.
4. The handler `app/routes/dashboard.py :: get_job_steps(job_id)` calls `app/database.py :: get_job_with_steps(job_id)`, which reads the job document from MongoDB and dedupes the `pipeline_steps[]` array.
5. The frontend's `PipelineFlow.jsx` renders the canvas — completed steps glow green, the running step has a pulsing halo + travelling particle on its incoming edge.

The **step bridge** (Stage 4) is what makes this work. Every `_emit(...)` call inside `correct_transcript()` blocks until MongoDB has the new step row, so by the time the dashboard polls, the data is there.

---

## File-By-File Summary — What Runs in This Workflow

### Files involved in every `/asr-correct` request

```
app/main.py                          (server boot only — already running)
app/auth.py                          (verify JWT)
app/routes/correction.py             (route handler + thread pool + step bridge)
app/database.py                      (5 calls: create_job, update_job_step×N,
                                      save_correction, complete_job, get_job_with_steps)
app/services/pipeline_settings.py    (load JSON-backed settings, apply to config)

asr_correction/__init__.py           (orchestrator, all _emit calls, helpers
                                      _enrich_vocab_from_web, _validate_candidates_via_web,
                                      _extract_vocab_from_ocr, _classify_topic,
                                      _gather_avsr_hints, _compute_ocr_timestamps)
asr_correction/config.py             (CorrectionConfig dataclass)
asr_correction/types.py              (CorrectionCandidate, CorrectionResult,
                                      CorrectionReport, OutputTranscript)
asr_correction/vocabulary.py         (load_domain_vocab, merge_vocabularies)
asr_correction/domain_vocab.json     (data file — 199 terms loaded by vocabulary.py)
asr_correction/model.py              (load_model, run_inference_raw)
asr_correction/llm_detector.py       (detect_errors)
asr_correction/segment_selector.py   (select_segments — rule-based fallback only)
asr_correction/text_utils.py         (normalize, find_occurrences, extract_context)
asr_correction/video_frames.py       (extract_frames_at_timestamps)
asr_correction/ocr_extractor.py      (_ocr_single_frame — PaddleOCR)
asr_correction/whisper_pass2.py      (retranscribe_flagged_segments,
                                      build_initial_prompt, extract_audio_segment)
asr_correction/avsr/__init__.py      (get_avsr_provider, AVSRHint)
asr_correction/avsr/mediapipe_hints.py  (MediaPipeHintProvider.analyze_segment)
asr_correction/reconciler.py         (reconcile_segments, _build_reconciliation_prompt,
                                      _parse_response)
asr_correction/data_collector.py     (collect_correction_data)
```

### Files NOT involved per-request, but loaded at server start

```
app/__init__.py                      (empty package marker)
app/config.py                        (read .env once at startup)
app/routes/__init__.py               (empty marker)
app/routes/health.py                 (different endpoint)
app/routes/evaluation.py             (different endpoint)
app/routes/training.py               (different endpoint)
app/routes/dashboard.py              (called by the dashboard polling, not /asr-correct)
app/routes/pipeline_settings.py      (settings API)
app/services/__init__.py             (empty marker)
```

### Files NOT involved in serving requests

```
asr_correction/target_term_extractor.py     (used by evaluation endpoint only)
asr_correction/technical_keywords.py        (used by tter)
asr_correction/tter.py                      (used by evaluation endpoint only)
asr_correction/wikipedia_client.py          (used by web vocab v2 future work)
asr_correction/ocr_parser.py                (used when ocr_xml is supplied — alt path)
asr_correction/avsr/auto_avsr.py            (only when avsr_mode=auto_avsr)
asr_correction/avsr/mouth_extractor.py      (used by mediapipe_hints internally)
evaluation/wer.py                           (POST /api/evaluate)
evaluation/tter.py                          (POST /api/evaluate)
evaluation/compare.py                       (POST /api/evaluate)
```

### Files that NEVER run when the backend is serving requests

`frontend/`, `training/`, `scripts/`, `tests/`, `docs/`, `data/`, `models/`, `asr_correction/adapters*/`. All of those produce things — they don't serve them.

---

## Visual Recap

```
┌────────────────────────────────────────────────────────────────────────┐
│  ScreenApp Backend (Node)                                              │
│         │ POST /asr-correct (JWT)                                      │
│         ▼                                                              │
│  app/main.py         ◄── boot once, lifespan opens Mongo               │
│  app/auth.py         ◄── verify JWT                                    │
│  app/routes/correction.py :: asr_correct()                             │
│         │                                                              │
│         ├─ app/database.py :: create_job()           [Mongo INSERT]    │
│         ├─ app/database.py :: update_job_step("request_received")      │
│         ├─ app/services/pipeline_settings.py :: apply_to_config()      │
│         │                                                              │
│         ├─ THREAD ► asr_correction.correct_transcript(...)             │
│         │       step_callback=bridge → update_job_step on every event  │
│         │                                                              │
│         │   ┌───────────────────────────────────────────────────────┐  │
│         │   │ 1.  vocabulary.py → vocab_merge                       │  │
│         │   │ 2.  model.py → model_load (Qwen3.5-9B singleton)      │  │
│         │   │ 3.  llm_detector.py → candidate_detection             │  │
│         │   │ 4.  __init__.py :: _classify_topic                    │  │
│         │   │ 5.  __init__.py :: _enrich_vocab_from_web (DDGS+LLM)  │  │
│         │   │ 6.  __init__.py :: _validate_candidates_via_web       │  │
│         │   │ 7.  video_frames.py + ocr_extractor.py → OCR          │  │
│         │   │ 8.  __init__.py :: _extract_vocab_from_ocr            │  │
│         │   │ 9.  whisper_pass2.py → faster-whisper Pass 2          │  │
│         │   │ 10. avsr/mediapipe_hints.py → AVSR hints              │  │
│         │   │ 11. reconciler.py → final LLM reconciliation          │  │
│         │   │ 12. types.py → build CorrectionReport                 │  │
│         │   │ 13. data_collector.py → append training pairs JSONL   │  │
│         │   │ 14. _emit("complete")                                 │  │
│         │   └───────────────────────────────────────────────────────┘  │
│         │                                                              │
│         ├─ app/database.py :: save_correction()      [Mongo INSERT]    │
│         ├─ app/database.py :: complete_job()         [Mongo UPDATE]    │
│         │                                                              │
│         ▼ HTTP 200 + JSON (enhanced_transcript + correction_report)    │
└────────────────────────────────────────────────────────────────────────┘
       ▲
       │  meanwhile, in parallel:
       │
   React Kinetic Console at /pipeline/<job_id>
   useJobSteps(jobId) polls /api/jobs/<id>/steps every 2 s
   PipelineFlow.jsx renders steps on the canvas as MongoDB receives them
```

---

## In One Sentence

> A request hits FastAPI → JWT verified → a job document is created in MongoDB → the pipeline runs in a worker thread, emitting 14 step events (each blocking on a MongoDB write) → the React dashboard polls those events live → the pipeline returns an enhanced transcript and a correction report → the route persists them and sends JSON back to ScreenApp.

That's the whole backend, end to end.
