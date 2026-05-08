"""ScreenApp ASR Correction Module (v2 — Whisper Reconciliation).

Pipeline:
1. LLM Detection — identify ASR errors using Qwen3.5-9B
2. Topic Classification + Web Vocab Enrichment
3. Quick OCR — extract screen text from video frames
4. Whisper Pass 2 — re-transcribe flagged segments with vocab hints
5. LLM Reconciliation — compare original vs Whisper, pick best words

Usage:
    from asr_correction import correct_transcript

    enhanced, report = correct_transcript(
        transcript=screenapp_output_transcript,
        file_id="abc-123",
        custom_vocabulary=[{"term": "ScreenApp", "category": "product_name"}],
        video_url="http://...",
    )
"""

import logging
import time

from .config import CorrectionConfig
from .data_collector import collect_correction_data
from .types import CorrectionReport
from .vocabulary import load_domain_vocab, merge_vocabularies

__version__ = "4.1.0"  # Whisper reconciliation pipeline + cleanup

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
        topic_info = _classify_topic(transcript, vocab_terms, candidates, model, tokenizer)
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
        web_vocab = _enrich_vocab_from_web(topic_info, model, tokenizer)
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
        _validate_candidates_via_web(candidates, vocab_terms, topic_info)
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
                ocr_timestamps = _compute_ocr_timestamps(analyses, duration, config)

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
            ocr_vocab = _extract_vocab_from_ocr(quick_ocr_hints, model, tokenizer)
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
        avsr_hints = _gather_avsr_hints(
            avsr_provider=avsr_provider,
            flagged=flagged,
            candidates=candidates,
            video_url=video_url,
            config=config,
        )
        if not avsr_hints:
            # _gather_avsr_hints already logged the precise reason — surface it.
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


_ALLOWED_WEB_VOCAB_CATEGORIES = {
    "product_name", "tech_term", "person_name",
    "company_name", "domain_term", "tech_acronym",
}


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text on the last sentence boundary at or before max_chars.

    Avoids the JSON-confusion problem where mid-sentence truncation feeds the
    LLM a half-thought that it tries to "complete" inside the JSON array.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer the last sentence-end punctuation; fall back to the last whitespace.
    for delim in (". ", "! ", "? ", "; ", "\n"):
        idx = cut.rfind(delim)
        if idx >= max_chars * 0.6:  # don't lose more than ~40 % of context
            return cut[: idx + 1]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0 else cut


def _parse_web_vocab_json(raw: str):
    """Best-effort JSON-array extraction. Returns list[dict] or None."""
    import json as _json
    import re as _re

    if not raw:
        return None
    # Strip optional Markdown fences
    raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=_re.MULTILINE)

    # Prefer a fully-bracketed array if present.
    match = _re.search(r"\[.*\]", raw, _re.DOTALL)
    snippet = match.group() if match else None

    if snippet is None:
        # Truncated / unclosed array — find the first '[' and try to seal it
        # at the last complete '}' we can see.
        lb = raw.find("[")
        rb = raw.rfind("}")
        if lb >= 0 and rb > lb:
            snippet = raw[lb : rb + 1] + "]"

    if snippet is None:
        return None

    try:
        parsed = _json.loads(snippet)
        return parsed if isinstance(parsed, list) else None
    except _json.JSONDecodeError:
        # Tolerant repair: trim to the last complete object inside the snippet.
        last_obj = snippet.rfind("}")
        if last_obj > 0:
            try:
                return _json.loads(snippet[: last_obj + 1] + "]")
            except _json.JSONDecodeError:
                return None
    return None


def _enrich_vocab_from_web(topic_info, model, tokenizer):
    """Step 2.6: Search web using topic description, extract vocab with Qwen.

    Quick-win improvements:
      - Honours `ASR_WEB_VOCAB_ENABLED=false` env kill switch (demo safety).
      - Sentence-boundary truncation of snippets (cleaner LLM input).
      - One self-repair retry when the LLM returns malformed JSON.
      - Quality gate: drops obviously bad terms before returning.
      - Structured single-line summary log for observability.
    """
    import os
    import time as _time
    from .model import run_inference_raw

    if os.environ.get("ASR_WEB_VOCAB_ENABLED", "true").lower() == "false":
        logger.info("Step 2.6: Disabled via ASR_WEB_VOCAB_ENABLED=false")
        return []

    description = topic_info.get("description", "") if topic_info else ""
    field = topic_info.get("field", "") if topic_info else ""
    topic = topic_info.get("topic", "") if topic_info else ""

    if not description or model is None or tokenizer is None:
        logger.info("Step 2.6: Skipped — no description or no model")
        logger.info("WEB_VOCAB summary status=skipped reason=no_description_or_model")
        return []

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info("Step 2.6: WEB VOCAB ENRICHMENT")

    # ── Search ──
    snippets = []
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("  ddgs not installed — run `pip install ddgs>=4.0`")
        logger.info("WEB_VOCAB summary status=skipped reason=ddgs_missing")
        logger.info("=" * 60)
        return []

    queries = [
        f"{field} {topic} terminology glossary",
        f"{description[:100]} technical terms",
    ]
    for q in queries:
        logger.info("  Searching: %s", q[:80])
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=5))
            for r in results:
                body = r.get("body", "")
                if body:
                    snippets.append(body)
        except Exception as e:
            logger.warning("  Search failed: %s", e)

    if not snippets:
        logger.info("  No search results found")
        logger.info("WEB_VOCAB summary status=empty queries=%d", len(queries))
        logger.info("=" * 60)
        return []

    web_context = _truncate_at_sentence(" ".join(snippets), 2500)
    logger.info("  Got %d chars of web context from %d snippets",
                len(web_context), len(snippets))

    # ── Extract via LLM (with one repair retry) ──
    extract_prompt = (
        f"This meeting is about: {description}\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n\n"
        f"Web search context about this domain:\n{web_context}\n\n"
        "List domain-specific terms that ASR (speech-to-text) commonly gets wrong.\n\n"
        "Return ONLY a compact JSON array on a single line:\n"
        '[{"term":"Groq","category":"company_name"},{"term":"Kubernetes","category":"tech_term"},...]\n\n'
        "Categories: product_name, tech_term, person_name, company_name, domain_term, tech_acronym\n"
        "Give 10-20 terms. Compact single-line JSON, no newlines."
    )
    system_prompt = (
        "You extract domain-specific vocabulary. Return only a compact JSON array."
    )

    raw_response = ""
    parsed = None
    try:
        raw_response = run_inference_raw(
            extract_prompt, system_prompt, model, tokenizer, max_tokens=1024,
        )
        parsed = _parse_web_vocab_json(raw_response)
    except Exception as e:
        logger.warning("  First extraction call failed: %s", e)

    # One repair retry — feed the malformed output back and ask for valid JSON.
    if parsed is None and raw_response:
        logger.info("  Retrying with self-repair…")
        repair_prompt = (
            "The previous response was not valid JSON. Return the SAME content "
            "as a single-line, well-formed JSON array of "
            '{"term":"…","category":"…"} objects, nothing else.\n\n'
            "Previous response:\n" + raw_response[:1500]
        )
        try:
            repaired = run_inference_raw(
                repair_prompt, system_prompt, model, tokenizer, max_tokens=1024,
            )
            parsed = _parse_web_vocab_json(repaired)
        except Exception as e:
            logger.warning("  Repair call failed: %s", e)

    if parsed is None:
        logger.warning("  Could not parse web-vocab JSON after retry")
        logger.info("  Raw (truncated): %s", (raw_response or "")[:240])
        logger.info("WEB_VOCAB summary status=parse_failed snippets=%d", len(snippets))
        logger.info("=" * 60)
        return []

    # ── Quality gate ──
    accepted, rejected = [], []
    seen = set()
    for v in parsed:
        if not isinstance(v, dict):
            continue
        term = (v.get("term") or "").strip()
        category = (v.get("category") or "").strip().lower()
        if not term:
            rejected.append(("empty_term", v))
            continue
        if len(term) < 3 or len(term) > 40:
            rejected.append(("bad_length", term))
            continue
        if term.lower() in seen:
            rejected.append(("duplicate", term))
            continue
        if category and category not in _ALLOWED_WEB_VOCAB_CATEGORIES:
            rejected.append(("bad_category", term))
            continue
        seen.add(term.lower())
        accepted.append({"term": term, "category": category or "domain_term"})

    elapsed_ms = int((_time.time() - t0) * 1000)
    logger.info("  Accepted %d / %d terms in %dms", len(accepted), len(parsed), elapsed_ms)
    for v in accepted[:15]:
        logger.info("    [%s] %s", v["category"], v["term"])
    if len(accepted) > 15:
        logger.info("    ... and %d more", len(accepted) - 15)
    if rejected:
        logger.info("  Rejected %d (sample): %s", len(rejected), rejected[:5])
    logger.info(
        "WEB_VOCAB summary status=ok accepted=%d rejected=%d snippets=%d latency_ms=%d",
        len(accepted), len(rejected), len(snippets), elapsed_ms,
    )
    logger.info("=" * 60)
    return accepted


def _validate_candidates_via_web(candidates, vocab_terms, topic_info):
    """Step 2.7: Cross-chunk target pooling + per-candidate web search.

    Two-pass validation:

    Pass 1 — Cross-chunk target pool (free, no web calls):
      The LLM processes the transcript in chunks and may propose DIFFERENT
      targets for the SAME entity across chunks (e.g. "Post-Sog → Post-SOC"
      in chunk 5 but "Post-Talk → Post-Hog" in chunk 6). Pass 1 collects
      ALL proposed targets into a pool, then for each candidate checks if
      another target in the pool is a better phonetic match for the error.

    Pass 2 — Web search fallback (for remaining unresolved candidates):
      For candidates still not in vocab after Pass 1, web-search to find
      the real entity name.
    """
    import jellyfish

    if not candidates:
        return

    vocab_set = {t["term"].lower() for t in vocab_terms}
    topic = topic_info.get("topic", "") if topic_info else ""
    field = topic_info.get("field", "") if topic_info else ""

    logger.info("=" * 60)
    logger.info("Step 2.7: CANDIDATE VALIDATION")

    # ── Pass 1: Cross-chunk target pooling ──
    # Collect all unique proposed targets across all candidates.
    target_pool = {}  # lowercased → original form
    for c in candidates:
        t = c.term.strip()
        if t and t.lower() not in vocab_set:
            target_pool[t.lower()] = t

    overrides = 0
    for c in candidates:
        error_word = c.error_found.strip()
        proposed = c.term.strip()
        if not error_word or error_word.lower() == proposed.lower():
            continue
        if proposed.lower() in vocab_set:
            continue  # Already in vocab — no need to validate

        error_clean = error_word.lower().replace("-", "").replace("'", "")
        proposed_clean = proposed.lower().replace("-", "").replace("'", "")
        proposed_sim = jellyfish.jaro_winkler_similarity(error_clean, proposed_clean)

        # Check if any OTHER target in the pool is a better match
        best_alt = None
        best_alt_sim = 0.0
        for pool_low, pool_orig in target_pool.items():
            if pool_low == proposed.lower():
                continue  # Same as current proposal
            if pool_low == error_clean:
                continue  # Same as error word
            pool_clean = pool_low.replace("-", "").replace("'", "")
            sim = jellyfish.jaro_winkler_similarity(error_clean, pool_clean)
            if sim > best_alt_sim and sim > 0.75:
                best_alt_sim = sim
                best_alt = pool_orig

        if best_alt and best_alt_sim > proposed_sim:
            logger.info("  CROSS-CHUNK OVERRIDE: '%s' → '%s' (was '%s', sim=%.2f vs %.2f)",
                        error_word, best_alt, proposed, best_alt_sim, proposed_sim)
            c.term = best_alt
            overrides += 1

    if overrides:
        logger.info("  Pass 1: %d candidates overridden from cross-chunk target pool", overrides)

    # ── Pass 2: Web search for remaining unresolved ──
    needs_web = []
    for c in candidates:
        if (c.term.lower() not in vocab_set
                and c.error_found.lower() != c.term.lower()
                and c.term.lower() not in target_pool):
            needs_web.append(c)

    if needs_web:
        needs_web = needs_web[:8]  # Cap at 8 web searches
        logger.info("  Pass 2: Web search for %d unresolved candidates", len(needs_web))
        web_overrides = 0
        try:
            from ddgs import DDGS
            import re
            for c in needs_web:
                error_word = c.error_found.strip()
                proposed = c.term.strip()
                error_nohyph = error_word.replace("-", " ")
                query = f'{error_nohyph} {field} {topic} product software'
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=3))
                except Exception:
                    continue
                # Extract PascalCase/camelCase entity words from results
                cw_set = set()
                for r in results:
                    for w in re.findall(r'\b[A-Z][a-zA-Z0-9]+\b',
                                        f"{r.get('title','')} {r.get('body','')}"):
                        if len(w) >= 4:
                            cw_set.add(w)
                error_clean = error_word.lower().replace("-", "")
                proposed_sim = jellyfish.jaro_winkler_similarity(
                    error_clean, proposed.lower().replace("-", ""))
                best_w, best_s = None, 0.0
                for cw in cw_set:
                    s = jellyfish.jaro_winkler_similarity(error_clean, cw.lower().replace("-", ""))
                    if s > best_s and s > 0.75 and cw.lower() != error_clean:
                        best_s, best_w = s, cw
                if best_w and best_s > proposed_sim:
                    logger.info("  WEB OVERRIDE: '%s' → '%s' (was '%s', sim=%.2f)",
                                error_word, best_w, proposed, best_s)
                    c.term = best_w
                    web_overrides += 1
            logger.info("  Pass 2: %d web overrides", web_overrides)
        except ImportError:
            logger.warning("  ddgs not installed — skipping web search")

    logger.info("=" * 60)


def _extract_vocab_from_ocr(ocr_hints, model, tokenizer):
    """Extract person names, product names, and company names from OCR text.

    Uses the LLM to identify high-value terms from raw screen text.
    These become protected ground truth for the reconciler.
    """
    from .model import run_inference_raw

    if not ocr_hints or not model:
        return []

    t0 = time.time()

    # Deduplicate and evenly sample across ALL OCR lines (not just first 80)
    unique = list(dict.fromkeys(ocr_hints))
    if len(unique) > 150:
        step = max(1, len(unique) // 150)
        unique = unique[::step][:150]
    sample = "\n".join(unique)

    prompt = (
        "Extract person names, product/tool names, and company names from this screen text.\n"
        "This text was read from a video meeting screen using OCR.\n\n"
        f"Screen text:\n{sample}\n\n"
        "Rules:\n"
        "- Only extract proper nouns (names, products, companies)\n"
        "- Ignore UI text like 'Stop presenting', 'Add people', 'Search'\n"
        "- Ignore URLs, email addresses, and random characters\n"
        "- Include full names for people (e.g. 'Avindi De Silva' → extract 'Avindi')\n\n"
        'Respond with JSON array: [{"term": "Avindi", "type": "person"}, '
        '{"term": "ChartMogul", "type": "product"}, {"term": "Stripe", "type": "company"}]\n'
        "Only include terms you are confident about."
    )

    system = "You extract structured data from OCR text. Output only valid JSON."

    try:
        raw = run_inference_raw(prompt, system, model, tokenizer, max_tokens=512)

        import json as _json
        import re as _re

        # Parse JSON array from response
        match = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if match:
            parsed = _json.loads(match.group())
            if isinstance(parsed, list):
                # Validate and deduplicate
                seen = set()
                valid = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    term = item.get("term", "").strip()
                    term_type = item.get("type", "").strip().lower()
                    if not term or len(term) < 2 or term.lower() in seen:
                        continue
                    if term_type not in ("person", "product", "company", "tool"):
                        continue
                    seen.add(term.lower())
                    valid.append({"term": term, "type": term_type})

                duration_ms = (time.time() - t0) * 1000
                logger.info("============================================================")
                logger.info("Step 3.5: OCR VOCAB EXTRACTION")
                logger.info("  Extracted %d terms from OCR in %.0fms:", len(valid), duration_ms)
                for v in valid[:15]:
                    logger.info("    [%s] %s", v["type"], v["term"])
                if len(valid) > 15:
                    logger.info("    ... and %d more", len(valid) - 15)
                logger.info("============================================================")
                return valid

    except Exception as e:
        logger.warning("Step 3.5: OCR vocab extraction failed — %s", e)

    return []


def _classify_topic(transcript, vocab_terms, candidates, model, tokenizer):
    """Classify the meeting topic/field using the LLM."""
    import json as _json
    import re as _re
    from .model import run_inference_raw

    topic_info = {"field": "unknown", "topic": "unknown", "description": "", "suggested_vocab": []}

    if model is None or tokenizer is None:
        logger.warning("[topic] No model available — skipping topic classification")
        return topic_info

    # Build transcript text
    transcript_text = transcript.get("text", "")
    if not transcript_text:
        segments = transcript.get("segments", [])
        transcript_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", ""))

    vocab_list = [t["term"] for t in vocab_terms[:30]]
    error_summary = ", ".join(
        f"'{c.error_found}'→'{c.term}'" for c in candidates[:15]
    )

    classify_prompt = (
        "Analyze this meeting transcript and classify it.\n\n"
        f"Transcript: {transcript_text[:3000]}\n\n"
    )
    if vocab_list:
        classify_prompt += f"Vocabulary terms: {_json.dumps(vocab_list)}\n\n"
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

    try:
        raw_response = run_inference_raw(
            classify_prompt, system_prompt, model, tokenizer, max_tokens=512,
        )
        json_match = _re.search(r"\{.*\}", raw_response, _re.DOTALL)
        if json_match:
            parsed = _json.loads(json_match.group())
            topic_info = {
                "field": parsed.get("field", "unknown"),
                "topic": parsed.get("topic", "unknown"),
                "description": parsed.get("description", ""),
                "suggested_vocab": parsed.get("suggested_vocab", []),
            }
    except Exception as e:
        logger.warning("[topic] Topic classification failed: %s", e)

    logger.info("=" * 60)
    logger.info("Step 2.5: TOPIC CLASSIFICATION")
    logger.info("  Field:       %s", topic_info["field"])
    logger.info("  Topic:       %s", topic_info["topic"])
    logger.info("  Description: %s", topic_info["description"])
    logger.info("  Suggested vocab: %s", topic_info["suggested_vocab"])
    logger.info("=" * 60)

    return topic_info


def _compute_ocr_timestamps(analyses, duration, config):
    """Compute targeted OCR timestamps from flagged segments.

    Strategy:
    - Extract frame at midpoint of each segment with needs_ocr=True
    - Add evenly-spaced frames to fill remaining budget
    - Cap at config.quick_ocr_num_frames total
    """
    max_frames = config.quick_ocr_num_frames

    # Targeted frames: midpoint of each needs_ocr segment
    targeted_ts = []
    for a in analyses:
        if getattr(a, 'needs_ocr', False) and a.start < a.end:
            midpoint = (a.start + a.end) / 2.0
            targeted_ts.append(midpoint)

    # Deduplicate timestamps that are very close (within 5s)
    targeted_ts.sort()
    deduped = []
    for ts in targeted_ts:
        if not deduped or ts - deduped[-1] > 5.0:
            deduped.append(ts)
    targeted_ts = deduped

    # Reserve up to 2/3 of budget for targeted, rest for evenly-spaced
    max_targeted = min(len(targeted_ts), int(max_frames * 0.67))
    targeted_ts = targeted_ts[:max_targeted]

    # Fill remaining slots with evenly-spaced frames
    remaining = max_frames - len(targeted_ts)
    if remaining > 0 and duration and duration > 0:
        interval = duration / (remaining + 1)
        for i in range(remaining):
            ts = interval * (i + 1)
            # Skip if too close to a targeted frame
            if not any(abs(ts - t) < 5.0 for t in targeted_ts):
                targeted_ts.append(ts)

    targeted_ts.sort()
    return targeted_ts[:max_frames]


_AVSR_ELIGIBLE_CATEGORIES = {"person_name", "content_word", "custom"}


def _gather_avsr_hints(avsr_provider, flagged, candidates, video_url, config):
    """Step 4.5: Run AVSR on AVSR-eligible candidates and collect hints.

    Quick-win wiring: lazily creates a default provider from `config.avsr_mode`
    when the caller didn't pass one. Caps work at `config.avsr_max_candidates`.
    Drops low-confidence hints below `config.avsr_min_speaking_confidence`.

    Returns:
        list[dict] — one entry per analyzed segment with shape
        {"start": float, "end": float, "hint": str, "confidence": float,
         "lip_transcript": Optional[str]}
    """
    import os
    import time as _time

    if config.dry_run or not video_url:
        return []
    if os.environ.get("ASR_AVSR_ENABLED", "true").lower() == "false":
        logger.info("Step 4.5: AVSR disabled via ASR_AVSR_ENABLED=false")
        return []

    mode = (getattr(config, "avsr_mode", "mediapipe") or "none").lower()
    if mode == "none":
        logger.info("Step 4.5: AVSR mode=none — skipping")
        return []

    # Lazy-create the provider if the caller didn't pass one.
    if avsr_provider is None:
        try:
            from .avsr import get_avsr_provider
            avsr_provider = get_avsr_provider(mode)
        except Exception as e:
            logger.warning("Step 4.5: Could not init AVSR provider (%s) — %s", mode, e)
            return []
    if avsr_provider is None:
        logger.info("Step 4.5: No AVSR provider available — skipping")
        return []

    # Pick AVSR-eligible candidates: person_name, content_word, custom.
    # When `avsr_run_on_all_flagged` is true, ignore the category gate and
    # process every flagged segment instead.
    run_on_all = bool(getattr(config, "avsr_run_on_all_flagged", False))
    eligible_ts = []
    seen = set()
    for a in flagged:
        if not (a.start < a.end):
            continue
        cats = {getattr(c, "category", "") for c in (a.candidates or [])}
        if run_on_all or (cats & _AVSR_ELIGIBLE_CATEGORIES):
            key = (round(a.start, 1), round(a.end, 1))
            if key in seen:
                continue
            seen.add(key)
            eligible_ts.append((a.start, a.end))

    if not eligible_ts:
        logger.info(
            "Step 4.5: No AVSR-eligible segments — skipping (run_on_all=%s)",
            run_on_all,
        )
        return []

    max_n = max(1, int(getattr(config, "avsr_max_candidates", 20)))
    eligible_ts = eligible_ts[:max_n]
    pad = float(getattr(config, "avsr_segment_padding_s", 0.5))
    min_conf = float(getattr(config, "avsr_min_speaking_confidence", 0.55))

    logger.info("=" * 60)
    logger.info("Step 4.5: AVSR (mode=%s) on %d segments (cap=%d)", mode, len(eligible_ts), max_n)

    hints = []
    t0 = _time.time()
    analyzed = 0
    for (s, e) in eligible_ts:
        try:
            hint = avsr_provider.analyze_segment(video_url, max(0.0, s - pad), e + pad)
        except Exception as exc:
            logger.warning("  AVSR failed on [%.2f-%.2f]: %s", s, e, exc)
            continue
        analyzed += 1
        if hint is None or not getattr(hint, "face_detected", False):
            continue

        conf = float(getattr(hint, "speaking_confidence", 0.0) or 0.0)
        if conf < min_conf and not getattr(hint, "lip_transcript", None):
            continue  # gated out — useless to the reconciler

        hints.append({
            "start": s,
            "end": e,
            "hint": hint.to_prompt_hint(),
            "confidence": conf,
            "lip_transcript": getattr(hint, "lip_transcript", None),
        })

    elapsed_ms = int((_time.time() - t0) * 1000)
    logger.info("  Produced %d usable hints (analyzed=%d) in %dms", len(hints), analyzed, elapsed_ms)
    for h in hints[:5]:
        logger.info("    [%.1f-%.1fs] %s", h["start"], h["end"], h["hint"])
    if len(hints) > 5:
        logger.info("    ... and %d more", len(hints) - 5)
    logger.info(
        "AVSR summary status=ok mode=%s analyzed=%d kept=%d dropped=%d latency_ms=%d",
        mode, analyzed, len(hints), max(0, analyzed - len(hints)), elapsed_ms,
    )
    logger.info("=" * 60)
    return hints


def _create_targeted_ocr_provider(ocr_hints_by_ts: dict):
    """Create a simple OCR provider from targeted extraction results.

    Returns a callable that matches the OCR provider interface:
    provider(file_id, start_s, end_s) -> OCR XML string
    """
    def provider(file_id, start_s, end_s):
        # Find all OCR results within the time window
        lines = ["<ocr-extraction>"]
        for ts, texts in sorted(ocr_hints_by_ts.items()):
            if start_s <= ts <= end_s:
                text_content = "\n".join(
                    t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    for t in texts
                )
                minutes = int(ts) // 60
                secs = int(ts) % 60
                lines.append(f'  <frame timestamp="{minutes:02d}:{secs:02d}" type="slide">')
                lines.append(f"    <text>{text_content}</text>")
                lines.append("  </frame>")
        lines.append("</ocr-extraction>")
        return "\n".join(lines)

    return provider
