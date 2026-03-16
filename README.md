# Multimodal Correction in STT Systems

A post-processing pipeline that corrects context-critical errors in automatic speech recognition (ASR) transcripts using a fine-tuned LLM, custom vocabulary, and on-screen text (OCR) hints.

Built as a companion service to [ScreenApp](https://screenapp.io) — integrates directly with the ScreenApp backend via JWT-authenticated API.

## Architecture

```
┌──────────────────┐       POST /asr-correct        ┌──────────────────────┐
│  ScreenApp       │ ──────────────────────────────> │  This Python Service │
│  Backend (Node)  │  { transcript, file_id,         │  (FastAPI)           │
│                  │    custom_vocabulary,            │                      │
│                  │    ocr_xml?, video_url? }        │  ┌────────────────┐  │
│                  │ <────────────────────────────── │  │ ASR Corrector  │  │
│                  │  { enhanced_transcript,          │  │ (Qwen2.5-7B   │  │
│                  │    correction_report }           │  │  + LoRA)       │  │
└──────────────────┘                                 │  └────────────────┘  │
                                                     │  ┌────────────────┐  │
                                                     │  │ OCR Extractor  │  │
                                                     │  │ (stub → future │  │
                                                     │  │  Gemini/Vertex)│  │
                                                     │  └────────────────┘  │
                                                     │  ┌────────────────┐  │
                                                     │  │ Dashboard DB   │  │
                                                     │  │ (MongoDB)      │  │
                                                     │  └────────────────┘  │
                                                     └──────────────────────┘
```

## How It Works

### Correction Pipeline

1. **Vocabulary Loading** — Merges team-specific custom vocabulary with a built-in domain vocabulary (`domain_vocab.json`) containing known ASR error patterns
2. **Candidate Identification** — Scans transcript text for known error patterns (e.g., "screen app" → "ScreenApp")
3. **OCR Context** — For each candidate, fetches on-screen text from the corresponding video timestamp to provide visual context
4. **LLM Inference** — A fine-tuned Qwen2.5-7B model (with LoRA adapters on Apple Silicon via MLX) evaluates each candidate with vocabulary + OCR context and decides whether to apply the correction
5. **Application** — Corrections above the confidence threshold are applied; the enhanced transcript and a detailed report are returned

### OCR Resolution (video_url flow)

When the ScreenApp backend doesn't have cached OCR data:
1. Backend sends `video_url` instead of `ocr_xml`
2. Python service checks its own `ocr_cache` collection
3. If not cached, calls `extract_ocr_from_video()` (currently a stub — returns None)
4. Future: Will use Gemini/Vertex AI to extract visible text from video frames

### Evaluation

- **WER** (Word Error Rate) and **CER** (Character Error Rate) via `jiwer`
- **TTER** (Target Term Error Rate) — custom metric measuring accuracy specifically on domain-critical terms
- **Word-level diff** — Aligned diff using jiwer's `process_words` alignments

### Dashboard

A web-based comparison dashboard at `/compare` shows:
- Side-by-side original vs enhanced transcript with synced scrolling
- Word-level diff highlighting (deletions in red, insertions in green, substitutions in yellow)
- Auto-loads past corrections from the dashboard database

## Project Structure

```
├── app/                        # FastAPI application
│   ├── main.py                 # App factory, lifespan, routes
│   ├── config.py               # Settings from .env
│   ├── auth.py                 # JWT verification (shared secret with ScreenApp)
│   ├── database.py             # MongoDB (motor) — dashboard DB operations
│   ├── routes/
│   │   ├── correction.py       # POST /asr-correct — main correction endpoint
│   │   ├── evaluation.py       # POST /api/evaluate, GET /api/corrections
│   │   ├── dashboard.py        # GET /compare — web dashboard
│   │   ├── training.py         # Training management endpoints
│   │   └── health.py           # GET /health
│   └── services/
├── asr_correction/             # Core correction module
│   ├── __init__.py             # correct_transcript() entry point
│   ├── corrector.py            # Candidate identification + correction orchestrator
│   ├── model.py                # LLM loading (MLX/transformers) + inference
│   ├── config.py               # CorrectionConfig dataclass
│   ├── types.py                # CorrectionCandidate, CorrectionResult, CorrectionReport
│   ├── vocabulary.py           # Vocabulary loading and merging
│   ├── ocr_parser.py           # OCR XML parsing + hint extraction
│   ├── ocr_extractor.py        # Video OCR extraction (stub)
│   ├── text_utils.py           # Text normalization, context extraction
│   ├── data_collector.py       # Collects correction pairs for future training
│   ├── domain_vocab.json       # Built-in domain vocabulary with known errors
│   └── dashboard.py            # Legacy Streamlit dashboard (reference)
├── evaluation/                 # Evaluation metrics
│   ├── compare.py              # WER/CER/TTER + word-level diff
│   ├── wer.py                  # WER calculator
│   └── tter.py                 # Target Term Error Rate
├── training/                   # LoRA fine-tuning
│   ├── prepare_data.py         # Training data preparation
│   └── train_lora.py           # MLX LoRA training script
├── templates/                  # Jinja2 HTML templates
│   └── compare.html            # Side-by-side diff comparison page
├── static/
│   └── js/charts.js            # Dashboard charts
├── requirements.txt
├── run.sh                      # Development server launcher
├── .env.example
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10+
- MongoDB (same instance as ScreenApp backend, port 27018)
- Apple Silicon Mac for MLX inference (or CUDA GPU with transformers/peft)

### Installation

```bash
# Clone
git clone https://github.com/Sanuka23/multimodal-correction-in-stt-systems.git
cd multimodal-correction-in-stt-systems

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your SESSION_SECRET (must match screenapp-backend)
```

### Model Weights

The fine-tuned Qwen2.5-7B model weights are not included in the repo (4GB+). Place them in:
- `asr_correction/model_weights/` — Base model weights
- `asr_correction/adapters/` — LoRA adapter weights

Without model weights, the service falls back to **dry-run mode** (rule-based vocabulary replacement).

### Running

```bash
# Development
./run.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# The API will be available at http://localhost:8000
# Dashboard at http://localhost:8000/compare
# API docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/asr-correct` | Correct a transcript (JWT required) |
| `POST` | `/api/evaluate` | Compare two transcripts (WER/CER/TTER) |
| `GET` | `/api/corrections` | List past corrections |
| `GET` | `/api/evaluations` | List past evaluation jobs |
| `GET` | `/compare` | Web dashboard — side-by-side diff viewer |
| `GET` | `/health` | Health check |

### POST /asr-correct

```json
{
  "transcript": { "text": "...", "segments": [...], "words": [...] },
  "file_id": "abc-123",
  "custom_vocabulary": "ScreenApp\nKubernetes\nMLX",
  "ocr_xml": "<ocr-extraction>...</ocr-extraction>",
  "video_url": "https://storage.example.com/video.mp4"
}
```

**Response:**
```json
{
  "enhanced_transcript": { "text": "...", "segments": [...] },
  "correction_report": {
    "file_id": "abc-123",
    "corrections_attempted": 5,
    "corrections_applied": 3,
    "processing_time_ms": 1234,
    "vocab_used": [...],
    "corrections": [...]
  }
}
```

## Databases

This service uses two MongoDB databases on the same instance:

| Database | Purpose | Access |
|----------|---------|--------|
| `screenapp` | Shared with ScreenApp backend | Read-only (PostProcessedText for cached OCR) |
| `asr_correction_dashboard` | Dashboard data: jobs, corrections, OCR cache | Read/Write |

## Integration with ScreenApp Backend

The ScreenApp backend (`enhanceTranscript.ts`) calls this service when a user clicks "Enhance with AI":

1. Checks `PostProcessedText` collection for cached OCR data
2. If cached OCR exists → sends `ocr_xml`
3. If no cached OCR → sends `video_url` for the Python service to handle extraction
4. Receives enhanced transcript and saves it to S3
5. Updates the file's `textData` with the new transcript URL

## Tech Stack

- **FastAPI** — Async API framework
- **Motor** — Async MongoDB driver
- **MLX** — Apple Silicon ML framework for model inference
- **jiwer** — Word/character error rate computation
- **Jinja2** — Server-side HTML templates for dashboard
- **Qwen2.5-7B** — Base LLM, fine-tuned with LoRA adapters for ASR correction

## License

MIT
