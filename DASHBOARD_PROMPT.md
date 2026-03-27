# Claude Code Prompt — Build Kinetic Console Dashboard (React + Vite)

## OBJECTIVE

Build a fully wired React + Vite frontend dashboard in a new `/frontend` directory inside the project root. The FastAPI backend stays completely unchanged — it becomes a pure API server. React handles all UI. Every page must match the designs in `/Users/sanukathamuditha/Downloads/stitch 2/` and pull real data from the FastAPI endpoints.

---

## DESIGN REFERENCE

All 7 screen designs are in `/Users/sanukathamuditha/Downloads/stitch 2/`:

| Folder | Page |
|--------|------|
| `dashboard_refined/` | Home dashboard |
| `compare_transcripts_refined/` | Transcript diff |
| `pipeline_control_refined/` | Pipeline config + toggles |
| `training_stats_refined/` | Training + dataset stats |
| `jobs_history_refined/` | Job history table |
| `live_pipeline_refined/` | Live job pipeline view |
| `evaluation_refined/` | Eval metrics + ablation |

Each folder has:
- `code.html` — Stitch-generated HTML with exact Tailwind classes, colors, fonts
- `screen.png` — visual reference screenshot

**Before writing any component, read the `code.html` for that page.** Extract exact class names, color tokens, layout structure, and component hierarchy. Do not guess the design — read it from the file.

---

## TECH STACK

### Frontend (NEW)
- **React 18** with functional components and hooks
- **Vite** — zero-config bundler, fast HMR
- **Tailwind CSS v3** — via PostCSS (NOT CDN — real Tailwind install)
- **React Router v6** — client-side routing
- **Recharts** — for charts (loss curve, bar charts, distribution charts)
- **Lucide React** — icons (replaces Material Symbols)
- **Axios** — for API calls

### Backend (DO NOT TOUCH)
- FastAPI + Motor (MongoDB async)
- All existing routes in `app/routes/` stay unchanged
- Only ADD new API routes described in this prompt

---

## PROJECT STRUCTURE TO CREATE

```
project_root/
├── frontend/                          ← CREATE THIS ENTIRE DIRECTORY
│   ├── index.html
│   ├── vite.config.js                 ← proxy /api → http://localhost:8000
│   ├── tailwind.config.js             ← with stitch 2 color tokens
│   ├── postcss.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                    ← router setup
│       ├── index.css                  ← Tailwind directives + global styles
│       ├── api/
│       │   └── client.js              ← axios instance pointing to localhost:8000
│       ├── hooks/
│       │   ├── usePolling.js          ← generic polling hook (interval, auto-stop)
│       │   └── useApi.js              ← data fetching with loading/error states
│       ├── components/
│       │   ├── layout/
│       │   │   ├── Sidebar.jsx
│       │   │   ├── TopNav.jsx
│       │   │   └── Layout.jsx         ← wraps Sidebar + TopNav + <Outlet />
│       │   ├── ui/
│       │   │   ├── StatCard.jsx
│       │   │   ├── Badge.jsx
│       │   │   ├── Toggle.jsx
│       │   │   ├── StepNode.jsx       ← pipeline step with icon + status
│       │   │   ├── SliderInput.jsx
│       │   │   ├── TabPanel.jsx
│       │   │   └── LogConsole.jsx     ← scrollable dark log output
│       │   └── charts/
│       │       ├── LossChart.jsx      ← recharts line, going DOWN
│       │       ├── BarChart.jsx       ← horizontal bars (source breakdown)
│       │       └── MiniBar.jsx        ← inline mini progress bar
│       └── pages/
│           ├── Dashboard.jsx
│           ├── Compare.jsx
│           ├── Jobs.jsx
│           ├── Training.jsx
│           ├── PipelineControl.jsx
│           ├── Pipeline.jsx           ← live view, takes :jobId param
│           └── Eval.jsx
├── app/                               ← EXISTING, do not touch structure
│   └── routes/
│       ├── dashboard.py               ← add new API routes here
│       ├── training.py                ← add training status routes
│       └── evaluation.py             ← add eval breakdown routes
└── ...
```

---

## STEP 1 — Project Setup

Run these commands from the project root:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install react-router-dom recharts lucide-react axios
```

### vite.config.js
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/asr-correct': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

### tailwind.config.js

Read the exact color tokens from `/Users/sanukathamuditha/Downloads/stitch 2/dashboard_refined/code.html` (the `tailwind.config` script block) and copy ALL of them into this file:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // COPY ALL COLORS FROM STITCH code.html tailwind.config HERE
        // Key ones: background, surface, surface-container, surface-container-high,
        // surface-container-highest, surface-container-low, surface-container-lowest,
        // primary, secondary, tertiary, on-surface, on-surface-variant,
        // outline, outline-variant, error, etc.
      },
      fontFamily: {
        headline: ['Manrope', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        label: ['Space Grotesk', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '0.75rem',
        '3xl': '1rem',
        '4xl': '1.5rem',
      },
    },
  },
  plugins: [],
}
```

### index.css
```css
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@400;500;700&display=swap');

@tailwind base;
@tailwind components;
@tailwind utilities;

html { background-color: #0b1326; }
body { font-family: 'Inter', sans-serif; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0b1326; }
::-webkit-scrollbar-thumb { background: #2d3449; border-radius: 9999px; }

/* Live pulse animation for running pipeline steps */
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(123, 208, 255, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(123, 208, 255, 0); }
}
.step-running { animation: pulse-glow 1.5s ease-in-out infinite; }

/* Blinking live dot */
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.live-dot { animation: blink 1.5s ease-in-out infinite; }
```

---

## STEP 2 — API Client + Hooks

### src/api/client.js
```js
import axios from 'axios'

const api = axios.create({
  baseURL: '/',  // vite proxy handles /api → FastAPI
  timeout: 30000,
})

export default api
```

### src/hooks/usePolling.js
```js
import { useEffect, useRef } from 'react'

export function usePolling(fn, intervalMs, deps = []) {
  const savedFn = useRef(fn)
  useEffect(() => { savedFn.current = fn }, [fn])

  useEffect(() => {
    savedFn.current()
    const id = setInterval(() => savedFn.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, ...deps])
}
```

### src/hooks/useApi.js
```js
import { useState, useEffect } from 'react'
import api from '../api/client'

export function useApi(url, defaultValue = null) {
  const [data, setData] = useState(defaultValue)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = async () => {
    try {
      const res = await api.get(url)
      setData(res.data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetch() }, [url])
  return { data, loading, error, refetch: fetch }
}
```

---

## STEP 3 — Layout Components

### src/components/layout/Sidebar.jsx
Read `/Users/sanukathamuditha/Downloads/stitch 2/dashboard_refined/code.html` for sidebar structure.

- Brand: "Kinetic Console" + "AI Correction Engine" subtitle, both in `font-headline`
- Nav items using `NavLink` from react-router-dom (active class applies cyan left border + bg tint)
- Icons from `lucide-react`: LayoutDashboard, Briefcase, GitCompare, Brain, Settings2, BarChart3, FileText
- Nav routes: `/`, `/jobs`, `/compare`, `/training`, `/pipeline-control`, `/eval`, external `/docs`
- Bottom: "Deploy Model" button → navigates to `/pipeline-control`
- Theme toggle: toggles `dark` class on `<html>`, persists to localStorage

### src/components/layout/TopNav.jsx
- Left: search input (styled to match stitch design)
- Center: 3 live status pills — MODEL, OCR, AVSR
  - Poll `/api/health` every 30s via `usePolling`
  - Each pill shows colored dot + label text
  - MODEL: green if loaded, yellow if loading
  - OCR: cyan + engine name (PaddleOCR / Google Cloud Vision / Disabled)
  - AVSR: yellow + mode name (MediaPipe / Auto-AVSR / Disabled)
- Right: "RUN EVAL" button (→ `/eval`), "PROMOTE V2" button (hidden until `/api/training/adapters` shows v2 available and not current), notification bell icon

### src/components/layout/Layout.jsx
```jsx
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopNav from './TopNav'

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopNav />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
```

### src/App.jsx
```jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Jobs from './pages/Jobs'
import Training from './pages/Training'
import PipelineControl from './pages/PipelineControl'
import Pipeline from './pages/Pipeline'
import Eval from './pages/Eval'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="compare" element={<Compare />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="training" element={<Training />} />
          <Route path="pipeline-control" element={<PipelineControl />} />
          <Route path="pipeline/:jobId" element={<Pipeline />} />
          <Route path="eval" element={<Eval />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
```

---

## STEP 4 — Dashboard Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/dashboard_refined/screen.png` and `code.html`

### Data fetching:
- `GET /api/stats` → correction_count, total_applied, total_attempted, avg_confidence, success_rate
- `GET /api/corrections?limit=8` → live correction feed
- `GET /api/jobs?limit=5` → job flow sidebar
- `GET /api/training/status` → training banner (poll every 15s)
- `GET /api/health` → pipeline health cards

### Sections:
**1. Stats row — 4 `<StatCard>` components:**
- Correction Count (icon: Zap)
- Total Applied (icon: CheckCircle, green value)
- Avg Confidence (icon: Target, amber value)
- Success Rate (icon: TrendingUp, green value)

**2. Training/Session Banner:**
- If `status.running === true`: show "ACTIVE TRAINING RUN" with live dot, iteration progress, current loss, ETA
- If not running: show "P50: 161s / P95: 342s" latency + adapter version + "View Last Job" button
- Banner is full-width with dark gradient card, uses `font-headline` for big numbers

**3. Pipeline Health (2/3 width) + Execution Summary (1/3 width):**
- Health: 3 equal cards — Model Status (adapter name), OCR Engine (provider), AVSR Module (mode)
- Each card has colored status dot + name + subtitle + colored bottom line
- Execution Summary: TTER Improvement %, Corrections Today, Avg Confidence — numbers in large `font-headline`

**4. Live Correction Feed (2/3 width) + Right Sidebar (1/3 width):**

Correction Feed items — for each correction with applied > 0:
```jsx
<div className="bg-surface-container rounded-3xl p-4">
  <div className="flex justify-between text-xs text-on-surface-variant mb-2">
    <span>ENTRY ID: {correction._id.substring(0,8)}</span>
    <span>{timeAgo(correction.created_at)}</span>
  </div>
  <h3 className="font-label font-semibold mb-3">{correction.corrections_applied} correction(s) applied</h3>
  <div className="grid grid-cols-2 gap-2 mb-3">
    <div className="bg-surface rounded-2xl p-3 border border-outline-variant">
      <div className="text-xs text-error font-bold mb-1">DETECTED</div>
      <div className="line-through text-red-400 font-mono text-sm">{first.error_found}</div>
    </div>
    <div className="bg-surface rounded-2xl p-3 border border-secondary/30">
      <div className="text-xs text-secondary font-bold mb-1">CORRECTED</div>
      <div className="text-secondary font-mono text-sm font-semibold">{first.term}</div>
    </div>
  </div>
  <div className="flex items-center gap-2">
    <Badge color="primary">{first.category}</Badge>
    <Badge color="cyan">Conf: {(first.confidence * 100).toFixed(0)}%</Badge>
    <span className="ml-auto text-xs text-on-surface-variant">{duration(correction.processing_time_ms)}</span>
  </div>
</div>
```

Right Sidebar:
- Job Flow list: each job shows colored status bar + file_id + type + duration
- Accent Distribution: static bars with real values from `data/collected_data/metadata.json`:
  - American: 6,096
  - South Asian: 5,958
  - European: 345
  - Unknown: 3,683

**Auto-refresh**: use `usePolling` at 15s for corrections and jobs.

---

## STEP 5 — Compare Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/compare_transcripts_refined/screen.png` and `code.html`

### State:
- `selectedJobId` — from history dropdown
- `reference` / `hypothesis` — loaded text or manually pasted
- `metrics` — WER, CER, TTER from POST `/api/evaluate`
- `corrections` — list of applied corrections with hints

### Metrics Row (top):
- WER % with mini progress bar
- CER % (show as "accuracy" = 100 - CER) with green bar
- Per-category TTER bars (horizontal): person_name, tech_term, product_name, tech_acronym, compliance, feature
  - Values from the loaded correction's per-category breakdown
  - Each bar has label + % value + colored fill

### Workspace header:
- "LOAD FROM HISTORY" dropdown — fetch `/api/corrections`, show each as `{file_id} · {date} · {N} corrections`
- "PASTE NEW" button — clears panels for manual paste

### Side-by-side panels:
```
┌─────────────────────────┬─────────────────────────┐
│  GROUND TRUTH (REF)     │  MODEL INFERENCE (HYP)  │
│  Clean text, NO marks   │  Inline diff shown here  │
│  [00:12] timestamp cyan │  errors + corrections    │
└─────────────────────────┴─────────────────────────┘
```

Ground truth panel: clean text, timestamps like `[00:12]` in `text-primary` color, clickable to scroll hypothesis to that position.

Hypothesis panel — build diff inline:
- Loop through corrections array
- For each correction: find `error_found` in text → wrap in `<span className="line-through text-red-400">`
- Follow immediately with `<span className="text-secondary font-semibold">{term}</span>`
- If `ocr_hints_used.length > 0`: append `<span className="text-primary text-xs bg-primary/10 px-1 rounded ml-1">[{ocr_hints_used[0]} | {confidence.toFixed(2)}]</span>`
- If `need_lip`: append `<span className="text-tertiary text-xs bg-tertiary/10 px-1 rounded ml-1">AVSR: Mouth Tracking</span>`

### Right sidebar:
- Error Distribution: Substitutions, Deletions, Insertions with recharts horizontal bars
- Comparison Metadata: Algorithm, Normalization, Segments, Execution time
- Precision Alert box (amber/warning): shown if WER > 10%, says which category has most errors

### Bottom:
- Recent Evaluations row: last 4 eval jobs as clickable cards with WER badge

---

## STEP 6 — Pipeline Control Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/pipeline_control_refined/screen.png` and `code.html`

### State management:
- Load current config from `GET /api/config/current` on mount
- Local state mirrors config — changes are staged, not applied until "Apply Changes" click
- POST to `/api/config/update` on apply

### Header:
- "Pipeline Configuration" title
- "Load Preset..." dropdown (fetch `/api/config/presets`)
- "Save Preset" button (modal with name input)
- "Reset" button (reloads from `/api/config/current`)
- "Apply Changes" button (cyan, primary)

### Left column — Global State + Node Registry:

**Global Pipeline State card:**
- Master Switch: custom `<Toggle>` component, large, green when on
- Pipeline Connectivity: green dot + "OPTIMAL"
- Current Latency: "p50: 161s"

**Active Node Registry — 5 rows with `<Toggle>` each:**
```jsx
const nodes = [
  { id: 'llm_detector',   label: 'LLM Detector',  icon: Eye },
  { id: 'ocr_engine',     label: 'OCR Engine',     icon: ScanLine },
  { id: 'avsr_sync',      label: 'AVSR Sync',      icon: Waves },
  { id: 'batch_corrector',label: 'Batch Corrector',icon: Layers },
  { id: 'data_collection',label: 'Data Collection',icon: Database },
]
```

### Right column — Core Model + Tabbed Settings:

**Core Model Orchestration:**
- Base Architecture: `<select>` with options matching models in `asr_correction/config.py`
- Adapter Version: button group showing adapters found on disk (v1 / v2 / Custom)
  - Fetch from `GET /api/training/adapters` to populate
  - Active adapter highlighted in cyan
- Compute Backend: button group `MLX` / `Torch`
- Active Adapter Path: text input + folder icon

**4 Tabs using `<TabPanel>` component:**

Tab 1 — OCR Settings:
- Engine Provider: 3 selectable icon cards — `PaddleOCR (Local)` / `Google Cloud` / `Disabled`
- Confidence Threshold: `<SliderInput>` 0.0→1.0, labels: Aggressive / Recommended(0.85) / Strict
- Frame Interval: `<SliderInput>` 5→60 seconds
- Max Frames: number input

Tab 2 — LLM Filter:
- Confidence Threshold: `<SliderInput>` 0.0→1.0, default 0.7
- Max Tokens: number input, default 512
- Fallback to Rules: `<Toggle>`

Tab 3 — AVSR Config:
- Mode: 3 button options — `MediaPipe` / `Auto-AVSR` / `Disabled`
- Confidence Threshold: `<SliderInput>`
- Max Segments: number input, default 20
- Auto-AVSR Model Dir: text input showing `models/auto_avsr`

Tab 4 — Presets:
- List saved presets from `/api/config/presets`
- Each row: preset name + "Load" button + "Delete" button
- Built-in presets: "Fast Mode", "Full Multimodal", "Vocab Only", "Dry Run"

**Bottom — Live System Output:**
- `<LogConsole>` component: dark bg, monospace, scrollable, 120px height
- Poll `/api/logs/recent` every 3s (add this simple endpoint to return last 10 log lines)
- Lines color-coded: INFO=gray, RUN=cyan, WARN=amber, ERR=red

---

## STEP 7 — Training Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/training_stats_refined/screen.png` and `code.html`

### Data:
- `GET /api/training/status` — poll every 5s if running
- `GET /api/training/adapters` — adapter list
- Read `data/collected_data/metadata.json` via `GET /api/training/dataset-stats`

### Header:
- "Model Training Orchestration" — `font-headline text-3xl font-bold`
- "Manage hyper-parameters, monitor convergence in real-time..." subtitle
- Top-right: "SYSTEM STATUS: NOMINAL" label + "Apple M3 Max / MLX backend"

### Left column:

**Hyperparameters card:**
- Iterations: number input (default 2000)
- Batch Size: number input (default 4)
- LR: text input (default "1e-5")
- LoRA Rank: number input (default 16)
- "INITIALIZE TRAINING RUN" button → POST `/api/train` with form values
- Disable button + show spinner if training already running

**Dataset Synthesis card:**
- Total Examples: big number from metadata (18,920)
- OCR Coverage bar: 11.3% (from metadata.ocr_coverage)
- AVSR Coverage bar: 0.0% (from metadata.avsr_coverage)
- ScreenApp Source Mix — horizontal bars per source:
  - Read from `metadata.sources` dict
  - Show top 5 by count with source name + count + %
- Accent Distribution — 4 colored dots grid:
  - South Asian 34.2% (green)
  - American 32.2% (cyan)
  - European 1.8% (amber)
  - Unknown 19.5% (gray)

### Right column:

**Convergence Metrics card:**
- Current Loss + dummy "Accuracy" display at top right
- `<LossChart>` — recharts LineChart, x=iteration, y=loss, line goes DOWN
  - Seed with the real loss data points: iter 30→1.827, 100→1.139, 200→1.108, 300→0.847, 400→0.810, 460→0.828
  - If training running, append new points from polling
  - Y-axis: 0→2, x-axis: 0→2000
  - Line color: `#7bd0ff` (primary), area fill below line with 10% opacity

**Training Progress (below chart):**
- "Step {current_iter} / {total_iter}" + percentage complete
- Progress bar (cyan fill)
- "Started: {time}" left + "Est. completion: {eta}" right

**Quality Filter Log (small card):**
```
REMOVED: Stop Words Pattern          -12,402
REMOVED: Short Context (<4s)         -48,912
PASSED: High Confidence (OCR)      +1,179,118
```

**Adapter Management table:**
| Identifier | Val Loss | Timestamp | Action |
|---|---|---|---|
| ADAPTERS_V2/ · V2, TRAINING | 0.0421 | Active Run | CURRENT badge |
| ADAPTERS/ · V1, VAL LOSS 0.773 | 0.7730 | 2025-10-20 | ROLLBACK button |

- "COMPARE ALL" button at top right of table
- "PROMOTE TO PROD" button (cyan) on V2 row when training is complete (val_loss exists)
- Promote → POST `/api/training/promote-v2`

---

## STEP 8 — Jobs Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/jobs_history_refined/screen.png` and `code.html`

### Data:
- `GET /api/jobs?type={type}&status={status}&limit={limit}&page={page}` → paginated jobs list
- `GET /api/jobs/stats` → daily corrections count for last 7 days

### Layout:

**Execution Velocity chart (2/3 width) + Stats sidebar (1/3 width):**
- Recharts BarChart: x=day name (MON-SUN), y=corrections count, bars in `#7bd0ff` (recent 2 days in `#4edea3`)
- Stats: Active Runs count + Failure Rate (24h) with critical/stable label

**Filter row:**
- Type dropdown: All Jobs / correction / evaluation / training
- Status dropdown: Any / completed / running / failed
- Date picker: Last 24h / Last 7 days / Last 30 days / Custom
- "Showing N of M records" counter

**Jobs table:**
```
Job Type | Job UUID | Status | Duration | Result Summary | Pipeline
```
- Job Type: small badge (correction=blue, eval=green, train=purple)
- Job UUID: first 12 chars of MongoDB ObjectId, clickable → copy full UUID
- Status: colored dot + text
- Duration: format as "14m 22s" or "2m 11s" (NOT "00:14:22")
- Result Summary: "98.2% Confidence · 14 corrections applied" for correction jobs
- Pipeline: "VIEW_FLOW" link → `/pipeline/{job_id}` | "MONITOR" for running | "LOGS" for failed

**Pagination:** numbered page buttons + Previous/Next

**Bulk Delete:**
- Checkbox column (select all / individual)
- "Bulk Delete" button in header (appears when rows selected) → confirm dialog → DELETE `/api/jobs/bulk`

**Footer:**
- Pipeline Health donut + text
- System Alerts (amber warning text if any)

---

## STEP 9 — Live Pipeline Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/live_pipeline_refined/screen.png` and `code.html`

**Route**: `/pipeline/:jobId`

### Data:
- `GET /api/jobs/{jobId}/steps` — poll every 2s while job is running
- Stop polling when status becomes "completed" or "failed"

### Header:
- Breadcrumb: `Jobs > PIPELINE-{jobId.substring(0,8)}`
- "Live Pipeline: {jobId.substring(0,8)}" title
- "RE-RUN CONFIG" button (ghost)
- "STOP JOB" button (red/danger)

### Step Nodes Row:

5 nodes connected by lines:
```
[Vocab Merge] ——— [LLM Detector] ——— [Targeted OCR] ——— [Batch Corrector] ——— [AVSR]
```

Each node is a `<StepNode>` component:
- Icon in rounded-3xl square (52px × 52px)
- Status determines style:
  - `done` → green bg + green border + checkmark
  - `running` → cyan bg + cyan border + step-running animation class
  - `pending` → surface-container bg + outline-variant border + gray icon
  - `error` → red bg + red border + X icon
  - `skipped` → surface-container bg + dashed border + minus icon
- Step name below in `font-label text-xs`
- Duration + status text below name

Connector lines between nodes: gray by default, turn cyan/green as preceding step completes.

Icons per step (from lucide-react):
- Vocab Merge: BookOpen
- LLM Detector: Eye
- Targeted OCR: ScanLine
- Batch Corrector: Layers
- AVSR: Waves

### Active Stream Analysis panel:
Pull stats from step details in the polling response:
- OCR Confidence %: from `ocr_extraction.details.confidence`
- Corrections Applied: from `ml_inference.details.corrections_applied`
- Stream Latency ms: from job `duration_ms` so far

### Real Extracted Video Frames section:
- Show frame thumbnails if `ocr_extraction.details.frames` exists in step data
- Each thumbnail: dark card with timestamp label + extracted OCR text overlay
- If no frames available: show placeholder "Extracting frames..." cards

### AVSR Fusion Context panel:
- Header status: "SPEAKER DETECTED, ACTIVELY SPEAKING (0.84)" or "WAITING FOR OCR TRIGGER" — from step details
- Log lines from `avsr_extraction.details.log` if available

### Right sidebar:
- Pipeline Health: total completion % progress bar
- System Events log: formatted step events with color-coded tags [INFO] [RUN] [WARN]
- "Download Audit Log" button → GET `/api/jobs/{jobId}/audit-log`
- Model Parameters: Engine: Qwen2.5-7B, Runtime: MLX, Correction: ENABLED (from job config)

---

## STEP 10 — Eval Page

Read: `/Users/sanukathamuditha/Downloads/stitch 2/evaluation_refined/screen.png` and `code.html`

**Route**: `/eval`

### Data:
- `GET /api/eval/source-breakdown` → per-video WER/TTER
- `GET /api/eval/ablation` → ablation study results
- `GET /api/eval/accent-breakdown` → TTER per accent group
- `GET /api/eval/failures?limit=10` → failed samples audit

### Header:
- "Evaluation Metrics" title + breadcrumb `PROJECTS / SCREENAPP_AUDIO_V9`
- "Configure Filters" button + "RUN EVALUATION" button (cyan)

### Top metrics row (3 cards):

WER card:
- "AVG. WORD ERROR RATE" label
- Big WER number with ↓ change indicator
- Mini bar chart (Recharts) showing last 7 eval runs

TTER card:
- "OVERALL TTER %" label
- "48.37%" in large font
- Previous version comparison: "42.10% +6.27%" (shows improvement direction)
- Stability + samples count

Active Model Specs card:
- ENGINE: QWEN2.5-7B (cyan)
- RUNTIME: MLX OPTIMIZED
- ADAPTER: LoRA (RANK 16)
- PRECISION: 4-BIT / QUANT

### Source Breakdown (ScreenApp) — main chart:
Recharts horizontal BarChart per video file:
- aws_migration.mp4
- followup_julien.mov
- screenapp_migration_kimi.mp4
- troubleshooting_dimiter.mp4
- compliance_discussion.mp4
- onboarding_andre.mp4
- project_update.mp4
- business_discussion.mp4
- zachary_onboarding.mp4

Each row has 2 bars: WER (cyan) + TTER (secondary/green)
WER value shown at right end of bar.

### Right sidebar — 3 cards:

**Ablation Study card** (MOST IMPORTANT — highlight this):
```
Vocab Only           8.42% ████████████
Vocab + OCR          5.12% ███████
Vocab + OCR + AVSR   3.42% █████  ← BEST (highlighted in cyan)
```
Label: "WER Δ" in top right

**Accent Group TTER card:**
- American: 52.1% (+2.1%)
- South Asian: 41.4% (-1.2%) — highlighted green (best improvement)
- European: 48.9% (0.0%)
- Unknown: 45.7% (+0.8%)

**Recent Eval Jobs card:**
- Last 3 eval jobs: timestamp + name + WER badge + arrow button

### Critical Failure Audit table (bottom):
```
Sample ID | Ground Truth | Model Output | WER | Action
```
- Model Output errors highlighted in red text
- "HIGH SEVERITY" filter badge + "EXPORT ALL" button
- "LOAD MORE FAILURES" button at bottom

---

## NEW API ROUTES TO ADD

Add these routes to the existing FastAPI backend. Do NOT break existing routes.

### In `app/routes/dashboard.py`:

```python
GET  /api/stats                  # {correction_count, total_applied, total_attempted, avg_confidence, success_rate, avg_duration_ms}
GET  /api/jobs                   # expose list_jobs as JSON API (query params: type, status, limit, page)
GET  /api/jobs/stats             # corrections per day for last 7 days (for bar chart)
POST /api/jobs/bulk-delete       # bulk delete job IDs
GET  /api/jobs/{job_id}/audit-log # return job pipeline_steps as formatted text
GET  /api/health                 # {model_loaded, ocr_status, ocr_engine, avsr_mode, latency_p50_ms}
GET  /api/config/current         # return CorrectionConfig defaults as dict
POST /api/config/update          # update runtime config in memory
GET  /api/config/presets         # list presets (hardcoded + any saved in MongoDB)
POST /api/config/presets         # save new preset to MongoDB
GET  /api/logs/recent            # last 10 lines from Python log handler (in-memory buffer)
```

### In `app/routes/training.py`:

```python
GET  /api/training/status        # {running, current_iter, total_iter, loss, eta, start_time, last_run}
GET  /api/training/adapters      # list adapters/ and adapters_v2/ with config + val_loss
GET  /api/training/dataset-stats # read data/collected_data/metadata.json and return it
POST /api/training/promote-v2    # copy adapters_v2/ to adapters/ with timestamp backup
```

### In `app/routes/evaluation.py`:

```python
GET  /api/eval/source-breakdown  # per-video WER+TTER from data/eval_results/ (return placeholder if missing)
GET  /api/eval/ablation          # ablation results (return static data if not computed yet)
GET  /api/eval/accent-breakdown  # TTER per accent group
GET  /api/eval/failures          # last eval run failed samples
```

For routes that read files that may not exist yet, return sensible placeholder/mock data rather than erroring.

### Enable CORS for React dev server in `app/main.py`:
The existing CORS middleware already has `allow_origins=["*"]` so this should work. Verify it's still there.

---

## IMPORTANT CONTENT CORRECTIONS

Every single one of these must be correct in the final UI:

| Wrong (from stitch template) | Correct (your real project) |
|---|---|
| "Llama-3-70B-Instruct-v1.4" | "mlx-community/Qwen2.5-7B-Instruct-4bit" |
| "A100 x 8 Cluster" | "Apple M3 Max" |
| "TensorRT 8.5" / "FP16" | "Qwen2.5-7B" / "MLX" |
| "1,240,492 examples" | "18,920 examples" |
| "OCR Coverage 94.2%" | "OCR Coverage 11.3%" |
| "AVSR Alignment 88.7%" | "AVSR Coverage 0.0%" |
| "V2.4.0-ADAPTIVE" | "adapters_v2/ (v2, training)" |
| "V1.9.2-LEGACY" | "adapters/ (v1, val_loss 0.773)" |
| "General American / British (RP)" | "American / South Asian / European / Unknown" |
| "Medical / Legal / Aviation" TTER | "person_name / tech_term / product_name / tech_acronym / compliance" |
| Job IDs "FL_9921_X_01" format | MongoDB ObjectId first 12 chars |
| Duration "00:14:22" | "14m 22s" |
| "Corrections per day" vague chart | Actual daily count from MongoDB |
| Loss curve going UP | Loss curve going DOWN (1.8 → 0.8) |
| "Synthetic Logs vs Human Verified" pie | Per-video source breakdown (aws_migration, etc.) |

---

## HOW TO RUN (add to README)

```bash
# Terminal 1 — FastAPI backend
cd project_root
python run.sh   # or: uvicorn app.main:app --reload --port 8000

# Terminal 2 — React frontend
cd frontend
npm run dev     # starts on http://localhost:5173
```

Open http://localhost:5173 for the dashboard.

For production build:
```bash
cd frontend
npm run build   # outputs to frontend/dist/
```
To serve from FastAPI in production, mount `frontend/dist` as StaticFiles and add catch-all route.

---

## EXECUTION ORDER

1. Read ALL 7 `code.html` files in `/Users/sanukathamuditha/Downloads/stitch 2/` to extract the full design system
2. Run `npm create vite` and install dependencies
3. Configure `tailwind.config.js` with all color tokens from stitch
4. Build shared components: Layout, Sidebar, TopNav, StatCard, Badge, Toggle, SliderInput, StepNode, LogConsole, LossChart
5. Build pages in this order: Dashboard → Compare → Training → Jobs → PipelineControl → Pipeline → Eval
6. Add all missing API routes to FastAPI
7. Wire every component to its real API endpoint
8. Test all pages: run both servers, navigate each route, verify real data appears

---

## QUALITY CHECKLIST

Before finishing, verify every item:

- [ ] `npm run dev` starts without errors
- [ ] All 7 routes render without console errors
- [ ] Manrope font used for all headlines and big numbers
- [ ] All cards use `rounded-3xl` minimum
- [ ] Dark mode is default, theme toggle works
- [ ] TopNav status pills update from `/api/health`
- [ ] Dashboard correction feed shows real `error → correction` pairs
- [ ] Loss chart shows downward curve with real data points
- [ ] Compare page: ground truth is clean (no strikethroughs), hypothesis shows inline diff with OCR/AVSR hints
- [ ] Pipeline page shows exactly 5 steps: Vocab Merge → LLM Detector → Targeted OCR → Batch Corrector → AVSR
- [ ] Pipeline steps poll every 2s and update in real time
- [ ] Eval page ablation study is visually prominent with "Vocab + OCR + AVSR" highlighted as best
- [ ] Jobs page durations show "14m 22s" format
- [ ] Training page loss curve goes DOWN
- [ ] Pipeline Control tabbed settings have 4 correct tabs: OCR Settings / LLM Filter / AVSR Config / Presets
- [ ] AVSR mode options are: MediaPipe / Auto-AVSR / Disabled (NOT "Sync Only" or "Continuous")
- [ ] OCR engine options are: PaddleOCR (Local) / Google Cloud / Disabled
- [ ] Adapter version shows real adapters from disk
- [ ] No "A100", "TensorRT", "Llama-3", "Synthetic Logs", "British RP", "FL_9921" anywhere
- [ ] Accent distribution shows: American 6096 / South Asian 5958 / European 345 / Unknown 3683
