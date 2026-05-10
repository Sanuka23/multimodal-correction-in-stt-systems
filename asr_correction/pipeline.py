"""ASR correction pipeline orchestrator.

This module contains the two top-level entry points the rest of the
codebase calls:

- :func:`correct_transcript` — full 14-stage correction pipeline
- :func:`analyze_transcript` — lightweight detect+classify path

Each numbered stage delegates to a focused module under
:mod:`asr_correction.stages` so that "where is X?" has a clean answer.

Stage map
---------
- Stage 2.5  → ``stages.topic_classification``
- Stage 2.6  → ``stages.web_vocab``
- Stage 2.7  → ``stages.candidate_validation``
- Stage 3    → ``stages.ocr_orchestration``
- Stage 3.5  → ``stages.ocr_vocab``
- Stage 4.5  → ``stages.avsr_hints``
- Stage 4    → :mod:`asr_correction.whisper_pass2`
- Stage 5    → :mod:`asr_correction.reconciler`
"""

from __future__ import annotations

import logging
import time

from .config import CorrectionConfig
from .data_collector import collect_correction_data
from .stages.avsr_hints import gather_avsr_hints
from .stages.candidate_validation import validate_candidates_via_web
from .stages.ocr_orchestration import compute_ocr_timestamps
from .stages.ocr_vocab import extract_vocab_from_ocr
from .stages.topic_classification import classify_topic
from .stages.web_vocab import enrich_vocab_from_web
from .types import CorrectionReport
from .vocabulary import load_domain_vocab, merge_vocabularies

logger = logging.getLogger(__name__)


def correct_transcript(
    transcript: dict,
    file_id: str,
    custom_vocabulary: list = None,
    ocr_provider=None,
    avsr_provider=None,
    video_url: str = None,
    config: CorrectionConfig = None,
    step_callback=None,
    **config_overrides,
) -> tuple:
    """Correct ASR errors in a ScreenApp transcript.

    New two-stage pipeline:
    1. Segment selection (rules + model) → identifies ~5-20% of segments
    2. Targeted OCR/AVSR → only processes flagged time ranges
    3. Correction → runs model on flagged candidates only

    Args:
        transcript: ScreenApp OutputTranscript dict with text, words,
            segments, and meta fields.
        file_id: ScreenApp file ID (for OCR requests and data collection).
        custom_vocabulary: Team/file custom vocabulary list.
        ocr_provider: Pre-existing OCR provider (cached OCR XML).
            If provided, used instead of targeted extraction.
        avsr_provider: AVSR provider for lip-reading analysis.
        video_url: Video URL for targeted OCR/AVSR extraction.
        config: Full CorrectionConfig. If None, uses defaults.
        **config_overrides: Override individual config fields.

    Returns:
        (enhanced_transcript, correction_report) tuple.
    """
    if config is None:
        config = CorrectionConfig(**config_overrides)
    else:
        for key, val in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)

    pipeline_start = time.time()

    # ── Step-emitter — fans out pipeline-step events to dashboard / logs ──
    def _emit(name: str, status: str, duration_ms=None, details=None):
        """Emit a pipeline-step event. Safe to call when no callback is set."""
        if step_callback is None:
            return
        try:
            step_callback(name, status, duration_ms, details or {})
        except Exception as cb_err:  # never let dashboard plumbing break the pipeline
            logger.debug("step_callback raised: %s", cb_err)

    # Step 1: Merge vocabularies
    _emit("vocab_merge", "running")
    t_step = time.time()
    domain_vocab = load_domain_vocab(config.domain_vocab_path)
    vocab_terms = merge_vocabularies(custom_vocabulary, domain_vocab)
    logger.info("Step 1: Merged %d domain + %d custom → %d vocab terms",
                len(domain_vocab), len(custom_vocabulary or []), len(vocab_terms))
    _emit("vocab_merge", "completed", (time.time() - t_step) * 1000, {
        "domain_terms": len(domain_vocab),
        "custom_terms": len(custom_vocabulary or []),
        "merged_terms": len(vocab_terms),
    })

    # Step 2: Error detection — LLM Pass 1 (primary) → rule-based (fallback)
    from .model import load_model
    from .llm_detector import detect_errors

    _emit("model_load", "running")
    t_step = time.time()
    model, tokenizer = None, None
    if not config.dry_run:
        try:
            model, tokenizer = load_model(
                adapter_path=config.adapter_path,
                model_path=config.model_path,
                base_model=config.base_model,
                backend=config.backend,
            )
            _emit("model_load", "completed", (time.time() - t_step) * 1000, {
                "base_model": config.base_model,
                "backend": config.backend or "auto",
                "adapter_path": config.adapter_path or "(prompt-only)",
                "loaded": model is not None,
            })
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            _emit("model_load", "failed", (time.time() - t_step) * 1000, {"error": str(e)})
    else:
        _emit("model_load", "skipped", 0, {"reason": "dry_run"})

    _emit("candidate_detection", "running")
    t_step = time.time()
    analyses = detect_errors(
        transcript, vocab_terms, model, tokenizer, config=config,
    )

    flagged = [a for a in analyses if a.needs_correction]
    candidates = [c for a in flagged for c in a.candidates]

    if not candidates:
        logger.info("No candidates found — returning original transcript unchanged")
        _emit("candidate_detection", "completed", (time.time() - t_step) * 1000, {
            "total_segments": len(analyses),
            "flagged_segments": 0,
            "candidates_found": 0,
        })
        report = CorrectionReport(
            file_id=file_id,
            corrections_attempted=0,
            corrections_applied=0,
            results=[],
            processing_time_ms=0,
        )
        _emit("complete", "completed", (time.time() - pipeline_start) * 1000, {
            "reason": "no_candidates",
        })
        return transcript, report

    logger.info("Step 2: Found %d candidates in %d/%d segments",
                len(candidates), len(flagged), len(analyses))
    for i, c in enumerate(candidates[:10]):
        logger.info("  Candidate %d: '%s' (error='%s', category='%s')",
                     i + 1, c.term, c.error_found, c.category)
    if len(candidates) > 10:
        logger.info("  ... and %d more candidates (showing first 10)", len(candidates) - 10)

    _emit("candidate_detection", "completed", (time.time() - t_step) * 1000, {
        "total_segments": len(analyses),
        "flagged_segments": len(flagged),
        "candidates_found": len(candidates),
        "candidates": [
            {
                "error_found": c.error_found,
                "likely_correct": c.term,
                "category": c.category,
                "timestamp": f"{c.timestamp_start:.0f}-{c.timestamp_end:.0f}s",
            }
            for c in candidates[:50]
        ],
    })

    # Step 2.5: Topic/field classification
    if config.enable_topic_classification:
        _emit("topic_classification", "running")
        t_step = time.time()
        topic_info = classify_topic(transcript, vocab_terms, candidates, model, tokenizer)
        _emit("topic_classification", "completed", (time.time() - t_step) * 1000, {
            "field": topic_info.get("field"),
            "topic": topic_info.get("topic"),
            "description": topic_info.get("description"),
            "suggested_vocab": (topic_info.get("suggested_vocab") or [])[:20],
        })
    else:
        topic_info = {"field": "unknown", "topic": "unknown", "description": "", "suggested_vocab": []}
        _emit("topic_classification", "skipped", 0, {"reason": "Disabled in pipeline settings"})

    # Step 2.6: Web vocab enrichment (search web using description, extract vocab with Qwen)
    if config.enable_web_vocab_enrichment:
        _emit("web_vocab_enrichment", "running")
        t_step = time.time()
        web_vocab = enrich_vocab_from_web(topic_info, model, tokenizer)
        _emit(
            "web_vocab_enrichment",
            "completed" if web_vocab else "skipped",
            (time.time() - t_step) * 1000,
            {
                "terms_added": len(web_vocab or []),
                "terms": (web_vocab or [])[:30],
            },
        )
    else:
        web_vocab = []
        _emit("web_vocab_enrichment", "skipped", 0, {"reason": "Disabled in pipeline settings"})

    # Step 2.7: Per-candidate web search validation
    if config.enable_candidate_validation:
        _emit("candidate_validation", "running")
        t_step = time.time()
        validate_candidates_via_web(candidates, vocab_terms, topic_info)
        _emit("candidate_validation", "completed", (time.time() - t_step) * 1000, {
            "candidates_after": len(candidates),
        })
    else:
        _emit("candidate_validation", "skipped", 0, {"reason": "Disabled in pipeline settings"})

    # Step 3: Smart OCR — targeted frames at error timestamps + evenly-spaced fallback
    _emit("ocr_extraction", "running")
    t_step = time.time()
    targeted_ocr_provider = ocr_provider
    quick_ocr_hints = []

    if not config.enable_ocr_extraction:
        # Toggled off — skip the whole block.
        pass
    elif not ocr_provider and video_url and not config.dry_run:
        try:
            from .video_frames import extract_frames_periodic, extract_frames_at_timestamps, get_video_duration
            from .ocr_extractor import _ocr_single_frame

            t3 = time.time()
            num_frames = config.quick_ocr_num_frames
            duration = get_video_duration(video_url)
            if duration and duration > 0:
                # Compute smart timestamps from flagged segments
                ocr_timestamps = compute_ocr_timestamps(analyses, duration, config)

                if ocr_timestamps:
                    n_targeted = sum(1 for a in analyses if getattr(a, 'needs_ocr', False))
                    logger.info("Step 3: Smart OCR — %d frames (%d targeted from flagged segments + evenly-spaced)",
                                len(ocr_timestamps), min(n_targeted, len(ocr_timestamps)))
                    frames = extract_frames_at_timestamps(video_url, ocr_timestamps)
                else:
                    # No flagged segments — fall back to periodic extraction
                    interval = max(duration / num_frames, 5.0)
                    logger.info("Step 3: No flagged OCR segments — falling back to %d evenly-spaced frames",
                                num_frames)
                    frames = extract_frames_periodic(
                        video_url, interval_s=interval, max_frames=num_frames,
                    )

                if frames:
                    seen = set()
                    for frame in frames:
                        result = _ocr_single_frame(frame)
                        if result:
                            for txt_entry in result.get("texts", []):
                                line = txt_entry.get("text", "").strip()
                                if not line or line in seen or len(line) <= 3:
                                    continue
                                # Filter noise
                                if any(skip in line for skip in [
                                    "http", "://", "&quot;", "&amp;", "{", "}", "[", "]",
                                    "index", "logprobs", "finish_reason", "content\"",
                                    "In-call messages", "access this chat",
                                ]):
                                    continue
                                if len(line) > 80:
                                    continue
                                seen.add(line)
                                quick_ocr_hints.append(line)

                    t3_ms = (time.time() - t3) * 1000
                    logger.info("Step 3: Quick OCR found %d text snippets in %.0fms", len(quick_ocr_hints), t3_ms)
                    for i, txt in enumerate(quick_ocr_hints[:5]):
                        logger.info("  OCR[%d]: %s", i, txt[:100])
                    if len(quick_ocr_hints) > 5:
                        logger.info("  ... and %d more", len(quick_ocr_hints) - 5)

                    # Create provider that returns all hints for any timestamp
                    if quick_ocr_hints:
                        all_xml = "\n".join(f"<frame><text>{t}</text></frame>" for t in quick_ocr_hints)
                        targeted_ocr_provider = lambda fid, start, end: all_xml
                else:
                    logger.info("Step 3: No frames extracted from video")
            else:
                logger.info("Step 3: Could not determine video duration")
        except (ImportError, Exception) as e:
            logger.warning("Step 3: Quick OCR failed — %s", e)
    elif ocr_provider:
        logger.info("Step 3: Using pre-cached OCR provider")
    else:
        logger.info("Step 3: No video URL — skipping OCR")

    if not config.enable_ocr_extraction:
        _emit("ocr_extraction", "skipped", (time.time() - t_step) * 1000, {
            "reason": "Disabled in pipeline settings",
        })
    else:
        _emit(
            "ocr_extraction",
            "skipped" if (not video_url and not ocr_provider) else "completed",
            (time.time() - t_step) * 1000,
            {
                "hints": len(quick_ocr_hints),
                "samples": quick_ocr_hints[:8],
                "source": "cached" if ocr_provider else ("video" if video_url else "none"),
                "reason": None if (video_url or ocr_provider) else "No video URL or cached OCR",
            },
        )

    # Step 3.5: Extract structured vocab from OCR text (names, products, companies)
    if not config.enable_ocr_vocab_extraction:
        ocr_vocab = []
        _emit("ocr_vocab_extraction", "skipped", 0, {"reason": "Disabled in pipeline settings"})
    else:
        _emit("ocr_vocab_extraction", "running")
        t_step = time.time()
        ocr_vocab = []
        if quick_ocr_hints and model and not config.dry_run:
            ocr_vocab = extract_vocab_from_ocr(quick_ocr_hints, model, tokenizer)
        _emit(
            "ocr_vocab_extraction",
            "completed" if ocr_vocab else "skipped",
            (time.time() - t_step) * 1000,
            {
                "terms_extracted": len(ocr_vocab),
                "terms": ocr_vocab[:30],
                "reason": None if ocr_vocab else "No OCR text to mine",
            },
        )

    # Step 4: Whisper Pass 2 — re-transcribe flagged segments with vocab hints
    _emit("whisper_pass2", "running")
    t_step = time.time()
    whisper_segments = {}
    if not config.enable_whisper_pass2:
        pass  # toggled off — skip
    elif video_url and flagged and not config.dry_run:
        try:
            from .whisper_pass2 import retranscribe_flagged_segments, build_initial_prompt

            # Build initial prompt — OCR-extracted terms get highest priority
            topic_suggested = topic_info.get("suggested_vocab", []) if topic_info else []
            custom_terms = [t["term"] for t in vocab_terms]
            ocr_names = [v["term"] for v in ocr_vocab if v.get("type") == "person"]
            ocr_products = [v["term"] for v in ocr_vocab if v.get("type") in ("product", "company")]
            web_terms = [v.get("term", "") for v in web_vocab] if web_vocab else []

            initial_prompt = build_initial_prompt(
                ocr_names=ocr_names,
                custom_vocab=custom_terms + ocr_products,
                topic_vocab=topic_suggested,
                web_vocab=web_terms,
            )

            # Collect flagged segment timestamps
            flagged_ts = [(a.start, a.end) for a in flagged if a.start < a.end]

            whisper_segments = retranscribe_flagged_segments(
                video_url=video_url,
                flagged_segments=flagged_ts,
                initial_prompt=initial_prompt,
                config=config,
            )
        except ImportError:
            logger.warning("Step 4: faster-whisper not installed — skipping")
        except Exception as e:
            logger.warning("Step 4: Whisper Pass 2 failed — %s", e)
    elif not video_url:
        logger.info("Step 4: No video URL — skipping Whisper Pass 2")
    else:
        logger.info("Step 4: No flagged segments — skipping Whisper Pass 2")

    if not config.enable_whisper_pass2:
        _emit("whisper_pass2", "skipped", (time.time() - t_step) * 1000, {
            "reason": "Disabled in pipeline settings",
        })
    else:
        skip_reason = None
        if not video_url:
            skip_reason = "No video URL"
        elif not flagged:
            skip_reason = "No flagged segments to re-transcribe"
        _emit(
            "whisper_pass2",
            "completed" if whisper_segments else ("skipped" if skip_reason else "completed"),
            (time.time() - t_step) * 1000,
            {
                "segments_re_transcribed": len(whisper_segments),
                "model": getattr(config, "whisper_model_size", "small"),
                "device": getattr(config, "whisper_device", "cpu"),
                "compute_type": getattr(config, "whisper_compute_type", "int8"),
                "reason": skip_reason,
            },
        )

    # Step 4.5: AVSR (lip-reading) hints for AVSR-eligible candidates.
    # Lazy-create a default provider if the caller didn't pass one.
    _emit("avsr_extraction", "running")
    t_step = time.time()
    avsr_hints = []
    avsr_skip_reason = None

    if not config.enable_avsr:
        avsr_skip_reason = "Disabled in pipeline settings"
    elif not video_url:
        avsr_skip_reason = "No video URL"
    elif not flagged:
        avsr_skip_reason = "No flagged segments"
    else:
        avsr_hints = gather_avsr_hints(
            avsr_provider=avsr_provider,
            flagged=flagged,
            candidates=candidates,
            video_url=video_url,
            config=config,
        )
        if not avsr_hints:
            # gather_avsr_hints already logged the precise reason — surface it.
            avsr_skip_reason = (
                "No AVSR-eligible candidates (no person_name / content_word "
                "candidates flagged). Toggle 'Run AVSR on all flagged' in "
                "settings to bypass this gate."
            )

    _emit(
        "avsr_extraction",
        "completed" if avsr_hints else "skipped",
        (time.time() - t_step) * 1000,
        {
            "mode": getattr(config, "avsr_mode", "mediapipe"),
            "hints": len(avsr_hints),
            "lip_transcripts": sum(1 for h in (avsr_hints or []) if h.get("lip_transcript")),
            "samples": [
                {"start": h["start"], "end": h["end"], "hint": h["hint"][:80]}
                for h in (avsr_hints or [])[:5]
            ],
            "reason": avsr_skip_reason,
        },
    )

    # Step 5: LLM Reconciliation — compare original vs Whisper, pick best words
    _emit("llm_reconciliation", "running")
    t_step = time.time()
    _reconcile_failed = False
    enhanced = dict(transcript)
    all_changes = []
    if whisper_segments and not config.dry_run:
        try:
            from .reconciler import reconcile_segments

            term_names = [t["term"] for t in vocab_terms]
            # OCR-confirmed terms as protected ground truth AND first-class vocab.
            # This ensures that terms visible on screen (e.g. PostHog, ChartMogul)
            # pass the vocab gate even if they are not in domain_vocab.json.
            protected_terms = [v["term"] for v in ocr_vocab]
            for v in ocr_vocab:
                ocr_term = v.get("term", "")
                if ocr_term and ocr_term not in term_names:
                    term_names.append(ocr_term)
                    logger.info("  Added OCR term to vocab: %s", ocr_term)

            # Build error candidate dicts for reconciler (from Step 2)
            error_candidate_dicts = []
            for c in candidates:
                error_candidate_dicts.append({
                    "error_found": c.error_found,
                    "likely_correct": c.term,
                    "category": c.category,
                    "context": c.context,
                    "timestamp_start": c.timestamp_start,
                    "timestamp_end": c.timestamp_end,
                })

            enhanced, all_changes = reconcile_segments(
                original_transcript=transcript,
                whisper_segments=whisper_segments,
                vocab_terms=term_names,
                ocr_hints=quick_ocr_hints,
                protected_terms=protected_terms,
                model=model,
                tokenizer=tokenizer,
                config=config,
                error_candidates=error_candidate_dicts,
                topic_info=topic_info,
                web_vocab=web_vocab,
                avsr_hints=avsr_hints,
            )
        except Exception as e:
            logger.warning("Step 5: Reconciliation failed — %s", e)
            enhanced = dict(transcript)
            _reconcile_failed = True
            _emit("llm_reconciliation", "failed", (time.time() - t_step) * 1000, {"error": str(e)})
    else:
        logger.info("Step 5: No Whisper segments to reconcile — returning original")

    if not _reconcile_failed:
        _emit(
            "llm_reconciliation",
            "completed" if all_changes else ("skipped" if not whisper_segments else "completed"),
            (time.time() - t_step) * 1000,
            {
                "swaps": len(all_changes),
                "sample_swaps": [
                    (item.get("swap") if isinstance(item, dict) else str(item))
                    for item in (all_changes or [])[:10]
                ],
            },
        )

    # Build report
    from .types import CorrectionResult, CorrectionCandidate
    results = []
    for change_item in all_changes:
        # all_changes is now a list of dicts: {"swap": "old → new", "confidence": 0.95}
        # (backwards compat: handle plain strings from older code paths too)
        if isinstance(change_item, dict):
            change_str = change_item.get("swap", "")
            swap_confidence = change_item.get("confidence", 0.9)
        else:
            change_str = str(change_item)
            swap_confidence = 0.9

        if "→" in change_str:
            parts = change_str.split("→")
            if len(parts) == 2:
                old_w = parts[0].strip()
                new_w = parts[1].strip()
                results.append(CorrectionResult(
                    candidate=CorrectionCandidate(
                        term=new_w, category="reconciled",
                        known_errors=[old_w], error_found=old_w,
                        char_position=0, timestamp_start=0, timestamp_end=0,
                        context=change_str,
                    ),
                    corrected_text=new_w,
                    changes=[change_str],
                    confidence=swap_confidence,
                    need_lip=False,
                    ocr_hints_used=[],
                    applied=True,
                ))

    report = CorrectionReport(
        file_id=file_id,
        corrections_attempted=len(whisper_segments),
        corrections_applied=len(all_changes),
        results=results,
        processing_time_ms=0,
    )

    # Step 6: Collect data for future training
    _emit("data_collection", "running")
    t_step = time.time()
    collected_count = 0
    skip_reason = None
    if not config.enable_data_collection:
        skip_reason = "Disabled in pipeline settings"
    elif not config.collect_data:
        skip_reason = "collect_data=False on config"
    elif not report.results:
        skip_reason = "No corrections to collect"
    else:
        logger.info("Step 6: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )
        collected_count = len(report.results)
    _emit(
        "data_collection",
        "completed" if collected_count else "skipped",
        (time.time() - t_step) * 1000,
        {
            "collected": collected_count,
            "output_dir": config.data_output_dir,
            "reason": skip_reason,
        },
    )

    # Add metadata to report
    report.selector_info = {
        "total_segments": len(analyses),
        "flagged_segments": len(flagged),
        "ocr_hints": len(quick_ocr_hints),
        "whisper_segments": len(whisper_segments),
        "candidates_found": len(candidates),
        "web_vocab_count": len(web_vocab or []),
        "avsr_hints_count": len(avsr_hints or []),
        "avsr_lip_transcripts": sum(
            1 for h in (avsr_hints or []) if h.get("lip_transcript")
        ),
    }
    report.topic_info = topic_info
    # Compact evidence breakdown — handy for the Compare page's
    # "Evidence Sources" panel and the Pipeline Monitor hover details.
    report.evidence_sources = {
        "vocab":     len(vocab_terms or []),
        "ocr":       len(quick_ocr_hints or []),
        "ocr_vocab": len(ocr_vocab or []),
        "web_vocab": len(web_vocab or []),
        "avsr":      len(avsr_hints or []),
    }

    total_ms = (time.time() - pipeline_start) * 1000
    report.processing_time_ms = total_ms
    logger.info("Pipeline complete in %.0fms", total_ms)

    _emit("complete", "completed", total_ms, {
        "corrections_attempted": report.corrections_attempted,
        "corrections_applied": report.corrections_applied,
        "evidence_sources": getattr(report, "evidence_sources", None),
    })

    return enhanced, report


def analyze_transcript(
    transcript: dict,
    file_id: str,
    custom_vocabulary: list = None,
    config: CorrectionConfig = None,
) -> dict:
    """Analyze a transcript: detect errors + classify meeting topic/field.

    Lightweight pipeline — no correction, no OCR, no AVSR.
    Returns dict with errors found and topic classification.
    """
    import json

    if config is None:
        config = CorrectionConfig()

    pipeline_start = time.time()

    # Step 1: Merge vocabularies
    domain_vocab = load_domain_vocab(config.domain_vocab_path)
    vocab_terms = merge_vocabularies(custom_vocabulary, domain_vocab)
    logger.info("[analyze] Step 1: Merged %d domain + %d custom → %d vocab terms",
                len(domain_vocab), len(custom_vocabulary or []), len(vocab_terms))

    # Step 2: Error detection (LLM primary, rule-based fallback)
    from .model import load_model, run_inference_raw
    from .llm_detector import detect_errors

    model, tokenizer = None, None
    if not config.dry_run:
        try:
            model, tokenizer = load_model(
                adapter_path=config.adapter_path,
                model_path=config.model_path,
                base_model=config.base_model,
                backend=config.backend,
            )
        except Exception as e:
            logger.error("[analyze] Model loading failed: %s", e)

    analyses = detect_errors(
        transcript, vocab_terms, model, tokenizer, config=config,
    )

    flagged = [a for a in analyses if a.needs_correction]
    candidates = [c for a in flagged for c in a.candidates]

    errors = []
    for c in candidates:
        errors.append({
            "error_found": c.error_found,
            "likely_correct": c.term,
            "category": c.category,
            "context": c.context,
        })

    logger.info("[analyze] Step 2: Found %d errors in %d/%d segments",
                len(errors), len(flagged), len(analyses))
    for i, e in enumerate(errors):
        logger.info("  Error %d: '%s' → '%s' (category=%s)",
                     i + 1, e["error_found"], e["likely_correct"], e["category"])

    # Step 3: Topic/field classification
    transcript_text = transcript.get("text", "")
    if not transcript_text:
        segments = transcript.get("segments", [])
        transcript_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", ""))

    # Build classification prompt
    vocab_list = [t["term"] for t in vocab_terms[:30]]
    error_summary = ", ".join(f"'{e['error_found']}'→'{e['likely_correct']}'" for e in errors[:15])

    classify_prompt = (
        "Analyze this meeting transcript and classify it.\n\n"
        f"Transcript: {transcript_text[:3000]}\n\n"
    )
    if vocab_list:
        classify_prompt += f"Vocabulary terms: {json.dumps(vocab_list)}\n\n"
    if error_summary:
        classify_prompt += f"ASR errors detected: {error_summary}\n\n"

    classify_prompt += (
        "Respond with JSON:\n"
        "{\n"
        '  "field": "broad field (tech, business, healthcare, education, finance, entertainment, etc.)",\n'
        '  "topic": "specific topic (e.g. AWS cloud deployment, marketing campaign planning)",\n'
        '  "description": "1-2 sentence summary of what this meeting discusses",\n'
        '  "suggested_vocab": ["term1", "term2", ...]\n'
        "}\n"
        "The suggested_vocab should be 10-20 domain-specific words likely discussed in this type of meeting."
    )

    system_prompt = "You are a meeting transcript analyzer. Classify the meeting by field and topic, and suggest domain-specific vocabulary."

    topic_info = {"field": "unknown", "topic": "unknown", "description": "", "suggested_vocab": []}

    if model is not None and tokenizer is not None:
        try:
            raw_response = run_inference_raw(
                classify_prompt, system_prompt, model, tokenizer, max_tokens=512,
            )
            # Parse JSON from response
            import re
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                topic_info = {
                    "field": parsed.get("field", "unknown"),
                    "topic": parsed.get("topic", "unknown"),
                    "description": parsed.get("description", ""),
                    "suggested_vocab": parsed.get("suggested_vocab", []),
                }
        except Exception as e:
            logger.warning("[analyze] Topic classification failed: %s", e)
    else:
        logger.warning("[analyze] No model available — skipping topic classification")

    logger.info("[analyze] Step 3: Topic classification:")
    logger.info("  Field:       %s", topic_info["field"])
    logger.info("  Topic:       %s", topic_info["topic"])
    logger.info("  Description: %s", topic_info["description"])
    logger.info("  Suggested vocab: %s", topic_info["suggested_vocab"])

    total_ms = (time.time() - pipeline_start) * 1000
    logger.info("[analyze] Complete in %.0fms — %d errors, field=%s, topic=%s",
                total_ms, len(errors), topic_info["field"], topic_info["topic"])

    return {
        "file_id": file_id,
        "errors": errors,
        "topic_info": topic_info,
        "processing_time_ms": total_ms,
    }
