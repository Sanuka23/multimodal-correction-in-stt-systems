# Roadmap for OCR and AVSR Extensions to the Multimodal STT Correction Framework

## Executive Summary

This report proposes a concrete next-steps roadmap for extending the existing Multimodal ASR Error Correction System ("transcript enhancer") with stronger OCR, audio‑visual speech recognition (AVSR), higher‑quality data, and additional fine‑tuning.
The plan is designed to fit the current architecture implemented for ScreenApp: a post‑ASR correction layer built around candidate detection, OCR‑enriched context, and a LoRA‑tuned Qwen2.5‑7B model, already integrated and evaluated with TTER metrics.[^1]
The roadmap is structured into practical phases you can implement and document directly in the GitHub repository.

## 1. Current System Status

### 1.1 Implemented Architecture

The current system is a transcript‑level post‑ASR correction framework that operates on top of ScreenApp’s existing ASR providers (Groq, Google, Fireworks, etc.), not as a replacement acoustic model.[^1]
It takes a completed ASR transcript, detects high‑risk candidate segments (technical terms, proper nouns, known misrecognitions), enriches them with OCR‑extracted screen text, and sends structured prompts to a LoRA‑tuned Qwen2.5‑7B model running locally on Apple Silicon (M3 Max) with 4‑bit quantization.[^1]
Corrections are only applied if model confidence exceeds a threshold (≥0.7), and the system computes Target Term Error Rate (TTER) along with summary metrics for evaluation.[^1]

### 1.2 Implemented Results and Gaps

Measured results show an 8% relative improvement in TTER for domain‑specific terms compared to baseline ASR, with a reduction of 39 target term errors on the evaluated dataset.[^1]
The pipeline is modular (ASR‑agnostic input, OCR parsing, candidate detector, LLM inference, evaluation), and ScreenApp integration (v6) plus a dashboard for evaluation are already in place.[^1]
However, visual speech information (lip movements) is still only planned as an optional future module, OCR is limited to ScreenApp’s current pipeline, and fine‑tuning relies on a relatively small, single‑task dataset.

## 2. Design Goals for the Next Phase

### 2.1 Technical and Product Goals

The next stage should focus on:

- Improving the reliability of corrections on accent‑heavy, noisy, and domain‑dense segments (names, product terms, financial metrics, etc.).[^1]
- Incorporating stronger OCR and, later, AVSR cues, but keeping the correction layer lightweight and decoupled from the underlying ASR engines.[^1]
- Maintaining near‑real‑time or "batch after recording" performance suitable for ScreenApp’s production environment.

### 2.2 Research and Evaluation Goals

From the original research questions, relevant goals for the extension phase are:

- Quantify additional gains from better OCR and optional AVSR on WER/CER and especially TTER across accents and noise conditions.[^1]
- Understand the latency and compute trade‑offs of each added modality in realistic meeting scenarios.[^1]
- Ensure improvements generalize beyond the fine‑tuning set via robust test splits and, ideally, cross‑dataset evaluation.

## 3. OCR: Stronger Context and Better Integration

### 3.1 Current OCR Usage

The existing system consumes OCR XML exported by ScreenApp’s pipeline, cleans and filters the text, and aligns it to candidate timestamps using a ±15 second window.[^1]
This is primarily used to enrich prompts with domain‑specific terms from slides, dashboards, and documents.
While this already improves context critical terms, the OCR source is still classical and may suffer on low‑resolution slides, complex layouts, and UI‑heavy screens.

### 3.2 Modern OCR and Vision‑Language Options

Recent vision‑language models (VLMs) such as Qwen2.5‑VL‑7B and related variants offer state‑of‑the‑art OCR and document understanding, including strong performance on DocVQA and TextVQA and robust GUI reasoning.[^2]
Qwen2.5‑Omni‑7B extends this to fully multimodal processing across text, images, audio, and video with time‑aligned rotary embeddings for temporal inputs, making it suitable for screen and video understanding workloads.[^3]
These models can often replace separate OCR + rule‑based parsing with end‑to‑end extraction of structured context (e.g., "slide title", "metric labels", "axis names").

### 3.3 Recommended OCR Roadmap

1. **Benchmark current ScreenApp OCR**
   - Build a small gold dataset of slide screenshots and UI screens with human‑labeled text regions.
   - Compare ScreenApp OCR vs. Tesseract/EasyOCR vs. a hosted or local VLM (e.g., Qwen2.5‑VL‑7B) on character‑level accuracy and coverage.[^2][^1]

2. **Introduce an OCR abstraction layer**
   - In the repo, create a `ocr_providers/` module with a common interface (e.g., `extract_text(image, timestamp) -> List[OcrSpan]`).
   - Implement backends:
     - `ScreenAppXmlProvider` (current behaviour).
     - `LocalTesseractProvider` (baseline classical OCR).
     - `VlmOcrProvider` (when/if you adopt Qwen2.5‑VL, Qwen2.5‑Omni, or similar).[^3][^2]

3. **Improve OCR filtering and alignment**
   - Add scoring for OCR spans based on confidence and spatial position (titles, headings, chart labels vs. body text).
   - Support per‑candidate time windows narrower than ±15 seconds where possible (use scene detection or slide‑change timestamps already available in ScreenApp).[^1]
   - Add heuristic de‑duplication (e.g., prefer slide titles and high‑contrast text for prompts).

4. **Update prompts to use structured OCR context**
   - Instead of passing raw OCR text blobs, feed structured items such as: `slide_title`, `key_terms`, `metric_labels`, `UI_labels` into the LLM prompt.
   - Reserve special sections in the prompt template (e.g., "Relevant screen terms:") so the model reliably uses them.

5. **Add OCR‑only ablation evaluation**
   - Re‑run TTER evaluation with three configurations: `no OCR`, `ScreenApp OCR`, and `enhanced OCR (best provider)`.
   - Quantify how much of the improvement in the next iteration comes purely from better OCR and attribute gains accordingly.[^1]

## 4. AVSR: Lip and Visual Speech Cues

### 4.1 Why AVSR Helps

Audio‑visual speech recognition research consistently shows that adding lip movement features improves recognition when audio is noisy or accented, with Deep AVSR‑style models reporting large WER reductions in controlled settings.[^4][^1]
In real meetings, this can help with difficult consonant clusters, dropped syllables, and heavily accented vowels that confuse purely acoustic models.[^1]
However, full AVSR models are heavy to train and deploy; a lightweight, assistance‑only approach is more realistic for this project.

### 4.2 Practical AVSR Integration Strategy

Instead of training a full audiovisual ASR model, the next step should be to:

- Use off‑the‑shelf AVSR models or multimodal LLMs to generate *confidence hints* or *phonetic support* around already identified transcript candidates.[^4][^3]
- Keep the current ASR transcript as the primary text, using AVSR output only when there is disagreement or low ASR confidence.

Candidate strategies include:

1. **External AVSR model for short segments**
   - Use a public AVSR implementation such as the KU Leuven multimodal SR repository (audio + lip CNN‑LSTM with attention) to re‑transcribe short windows around candidate terms.[^4]
   - Compare AVSR output against baseline ASR only on high‑risk words; if AVSR strongly supports an alternative spelling or term, surface that as an extra feature into the LLM prompt.

2. **Multimodal LLMs for lip reading hints**
   - With models like Qwen2.5‑Omni, send a short video clip (e.g., 1–2 seconds) plus the ASR snippet and ask the model: "Is the spoken phrase closer to A or B?" for two candidate terms.[^3]
   - Use this as a discrete feature (`lip_support = {term: score}`) in the correction prompt, rather than replacing the whole transcript.

3. **Fallback and failure modes**
   - If no reliable face region is detected (user camera off, poor angle), simply skip AVSR features for that segment.
   - AVSR should never be mandatory for the pipeline to run, only an additional signal.

### 4.3 AVSR Development Phases

- **Phase A (Prototype)**
  - Implement a small Python module that, given a transcript snippet and timestamps, extracts corresponding video frames, detects the mouth region with MediaPipe/OpenCV, and runs an AVSR or multimodal model for a few test cases.[^4][^1]
  - Evaluate impact on a tiny, curated set of clips where ASR clearly fails on accent or noise.

- **Phase B (Feature Integration)**
  - Define a compact JSON structure for AVSR hints (e.g., `"lip_confirms_candidate": true/false`, `"alternative_phoneme_pattern"`).
  - Extend the correction prompt builder to optionally insert a short "Visual hint" section when this data is present.

- **Phase C (Optional Fine‑Tuning)**
  - If AVSR proves consistently helpful on a subset of patterns (e.g., particular accents), consider creating an auxiliary training dataset where prompts include AVSR hints and the LoRA adapters are fine‑tuned to learn when and how to trust them.

## 5. Data and Fine‑Tuning Strategy

### 5.1 Current Fine‑Tuning Setup

The present system fine‑tunes Qwen2.5‑7B Instruct with LoRA on a dataset of 1,454 examples (1,193 train, 261 validation) using 4‑bit quantization and a small adapter size (≈11 MB), optimized for offline correction tasks.[^1]
This dataset is tailored to ScreenApp’s domain but is relatively small and primarily text‑only (transcript + OCR hints → corrected transcript), limiting generalization across topics and accents.[^1]

### 5.2 Data Collection Improvements

1. **Expand ScreenApp‑based correction pairs**
   - Systematically log manual user edits from a sample of real meetings (with consent and anonymization) to build additional (ASR, corrected) pairs, focusing especially on high‑value terms.
   - Use the existing evaluation dashboard to flag episodes with high TTER and prioritize them for manual labeling.[^1]

2. **Leverage public corpora for robustness**
   - Use meeting corpora such as AMI and ICSI and AVSR datasets like LRS3 to expose the model to more accent and conversation styles, even if these are only used for evaluation or synthetic correction tasks.[^1]
   - For segments where human reference transcripts exist but ASR still fails systematically, generate pseudo‑correction pairs (ASR output, ground truth) to diversify training patterns.

3. **Targeted vocabulary augmentation**
   - Maintain curated lists of domain vocabularies (ScreenApp product terms, common SaaS metrics, finance and engineering jargon) and intentionally inject them into training prompts and expected outputs.
   - Monitor Target Term Error Rate on these curated vocabularies as a primary success metric.[^1]

### 5.3 Multi‑Task and Curriculum Fine‑Tuning

To avoid overfitting and to better exploit multimodal context, consider a staged fine‑tuning approach:

- **Stage 1: Pure text correction**
  - Train on large, diverse STT correction pairs (including external corpora where available) using only transcript context.

- **Stage 2: Text + OCR**
  - Add tasks where prompts include structured OCR fields (slide titles, key terms) and the target is the corrected transcript with properly spelled domain terms.

- **Stage 3: (Optional) Text + OCR + AVSR hints**
  - Once AVSR hints exist, augment a subset of training data to include them and let the model learn under what conditions to trust them.

Throughout, use LoRA on Qwen2.5‑7B or Qwen2.5‑VL‑7B to keep training efficient while benefiting from strong base model capabilities.[^5][^2]

### 5.4 Evaluation Enhancements

In addition to current TTER, WER, and CER, consider:

- **Accent‑stratified metrics**: compute TTER/WER separately for speakers from South Asia, Europe, North America, etc., to quantify fairness and robustness.[^1]
- **Semantic WER or Semantic similarity**: where possible, compute a semantic similarity between corrected transcripts and references to ensure improvements are not just lexical.
- **Latency tracking**: integrate time‑to‑final‑segment–style metrics (TTFS) similar to STT benchmarking frameworks, to ensure added multimodality does not break real‑time expectations.[^6]

## 6. Repository Structure and Documentation Plan

### 6.1 Recommended Repo Layout

To make future development predictable and collaborator‑friendly, extend the repo along these lines:

- `src/`
  - `asr_input/` – adapters for different ASR providers.
  - `ocr_providers/` – ScreenApp XML, Tesseract, VLM‑based OCR.
  - `avsr/` – optional AVSR feature extraction and lip hints.
  - `correction/` – candidate detection, prompt building, LLM inference, confidence filtering.
  - `evaluation/` – TTER, WER, CER, accent‑stratified metrics, dashboards.

- `scripts/`
  - `run_correction.py` – CLI entry for correcting a single transcript.
  - `run_batch_eval.py` – batch evaluation and report generation.
  - `export_training_data.py` – generate fine‑tuning datasets from logs.

- `docs/`
  - `ARCHITECTURE.md` – high‑level diagrams and explanations taken from the IPD design chapters.[^1]
  - `OCR_AND_AVSR_PLAN.md` – this roadmap compressed into a dev‑oriented plan.
  - `DATASETS_AND_TRAINING.md` – what datasets are used, where they live, and how to reproduce fine‑tuning.
  - `EVALUATION.md` – description of metrics, expected baselines, and how to run eval.

### 6.2 Roadmap Document Template

A concrete file you can add (e.g., `docs/ROADMAP.md`) could follow this structure:

- **1. Overview** – one paragraph on the goal of the transcript enhancer.
- **2. Current Status** – bullet list of what is implemented and what is missing.
- **3. Phase 1 – OCR Enhancements** – tasks, milestones, and success criteria.
- **4. Phase 2 – AVSR Integration (Optional)** – prototype, integration, and evaluation.
- **5. Phase 3 – Data & Fine‑Tuning** – dataset expansion and multi‑task training.
- **6. Phase 4 – Productization** – monitoring, logging, and ScreenApp rollout.
- **7. Risks and Mitigations** – sync issues, privacy, compute costs.

This makes the GitHub project self‑documenting for future contributors.

## 7. Common Pitfalls and Mistakes to Avoid

### 7.1 Overcorrection and Hallucination

If the LLM is allowed to rewrite long spans of text without constraints, it may hallucinate content or change correct words, degrading trust.
The current candidate‑based approach and confidence threshold should be preserved and strengthened; future iterations should log before/after examples and manually review a sample of corrections every time the model or prompts change.[^1]

### 7.2 Blind Trust in OCR or AVSR

OCR and AVSR outputs can be wrong, especially on low‑quality video or complex layouts.[^2][^1]
Treat these as hints, not ground truth: always keep the original transcript in view, and design prompts that ask the LLM to "prefer" but not "force" terms from OCR/AVSR when they fit the sentence semantics.

### 7.3 Poor Synchronization Between Modalities

Misaligned timestamps between audio, video, and screen content can cause the system to associate the wrong slide or lip movements with a spoken phrase, leading to incorrect corrections.[^1]
Always validate alignment logic on a small test set with visual inspection before relying on it at scale.

### 7.4 Ignoring Privacy and Consent

Using participant video and screen content for AVSR and OCR greatly increases the sensitivity of the data.
Continue following the IPD’s ethical guidelines: explicit consent, anonymization where needed, secure storage, and limiting usage to research and evaluation.[^1]

### 7.5 Evaluation Drift

As models and prompts evolve, it is easy to lose track of which changes actually improved TTER and WER.
Automate evaluation (same test set, same metrics) and require that any change to prompts, OCR provider, or fine‑tuning configuration be accompanied by an updated metrics report in the repo.

## 8. Phase‑Wise Implementation Plan

### 8.1 Phase 0 – Cleanup and Baseline Freeze

- Export current code structure and ensure `ARCHITECTURE.md` and `EVALUATION.md` reflect the implementation described in Chapters 6–7 of the IPD.[^1]
- Freeze a baseline model + prompt configuration and record its metrics.

### 8.2 Phase 1 – OCR Upgrade

- Implement the OCR abstraction layer and at least one alternative OCR backend.
- Run ablation experiments and choose a default provider per scenario (slides vs. dashboards vs. text‑heavy documents).
- Update prompts to use structured OCR context and re‑baseline metrics.

### 8.3 Phase 2 – AVSR Prototype (Optional but Recommended)

- Build a small AVSR prototype for a handful of clips, integrating either a classic AVSR repo or a multimodal LLM.
- Design and log a few examples where AVSR clearly changes a correction decision for the better.
- Decide whether the benefit justifies production complexity and, if so, proceed to integration.

### 8.4 Phase 3 – Data Expansion and Fine‑Tuning

- Implement logging and export scripts to collect more (ASR, corrected) pairs.
- Expand the fine‑tuning dataset with additional ScreenApp examples and (optionally) public corpora.
- Run staged fine‑tuning and compare against the frozen baseline, focusing on TTER and accent‑stratified metrics.

### 8.5 Phase 4 – Productization and Monitoring

- Integrate the improved correction engine into ScreenApp staging, with feature flags to control rollout.
- Add monitoring dashboards for latency, error rates, and success metrics.
- Gradually roll out to production users, collect feedback, and schedule periodic re‑training based on drift.

## 9. Conclusion

The existing Multimodal ASR Error Correction System already delivers measurable improvements in context‑critical term accuracy for ScreenApp using a carefully designed transcript‑level correction pipeline and a fine‑tuned Qwen2.5‑7B model.[^1]
By systematically upgrading OCR, introducing optional AVSR cues, expanding and diversifying training data, and strengthening evaluation and documentation, the next phase can turn this prototype into a robust, extensible transcript enhancer suitable for long‑term maintenance and future research.
This roadmap is designed so that each phase can be reflected directly in the GitHub repository through clear modules, documentation files, and reproducible evaluation scripts, making it easier for future contributors (including your future self) to continue building on this work.

---

## References

1. Sanuka Alles - IPD Report: Multimodal ASR Error Correction System - Informatics Institute of Technology / University of Westminster

2. [Qwen2.5-VL-7B](https://en.immers.cloud/ai/Qwen/qwen2.5-vl-7b/) - Rent GPU servers for hosting Qwen2.5-VL-7B. Cloud and dedicated GPU servers for rent.

3. [README.md at main · geic2025/Qwen2.5-Omni-7B](https://cnb.cool/geic2025/Qwen2.5-Omni-7B/-/blob/main/README.md) - Our developer platform is dedicated to helping developers accelerate software development.

4. [matthijsvk/multimodalSR: Multimodal speech recognition ...](https://github.com/matthijsvk/multimodalSR) - Multimodal speech recognition using lipreading (with CNNs) and audio (using LSTMs). Sensor fusion is...

5. [Qwen2.5-LLM: Extending the boundary of LLMs](https://qwenlm.github.io/blog/qwen2.5-llm/) - GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In this blog, we delve into the details of ...

6. [pipecat-ai/stt-benchmark](https://github.com/pipecat-ai/stt-benchmark) - A framework for benchmarking Speech-to-Text services with TTFS (Time To Final Segment) latency and S...

