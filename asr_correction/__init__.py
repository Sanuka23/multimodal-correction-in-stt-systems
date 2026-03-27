"""ScreenApp ASR Correction Module.

Two-stage selective correction pipeline:
1. Segment Selector — analyzes transcript + vocab to identify which segments
   need fixing and which modality (OCR/AVSR) each needs.
2. Targeted Correction — only fetches OCR/runs AVSR on flagged segments,
   then corrects with the fine-tuned model.

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

from .batch_corrector import correct_transcript_batch
from .config import CorrectionConfig
from .corrector import correct_candidates
from .data_collector import collect_correction_data
from .segment_selector import select_segments
from .types import CorrectionReport
from .vocabulary import load_domain_vocab, merge_vocabularies

__version__ = "3.0.0"

logger = logging.getLogger(__name__)


def correct_transcript(
    transcript: dict,
    file_id: str,
    custom_vocabulary: list = None,
    ocr_provider=None,
    avsr_provider=None,
    video_url: str = None,
    config: CorrectionConfig = None,
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

    # Step 1: Merge vocabularies
    domain_vocab = load_domain_vocab(config.domain_vocab_path)
    vocab_terms = merge_vocabularies(custom_vocabulary, domain_vocab)
    logger.info("Step 1: Merged %d domain + %d custom → %d vocab terms",
                len(domain_vocab), len(custom_vocabulary or []), len(vocab_terms))

    # Step 2: Error detection — LLM Pass 1 (primary) → rule-based (fallback)
    from .model import load_model
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
            logger.error("Model loading failed: %s", e)

    analyses = detect_errors(
        transcript, vocab_terms, model, tokenizer, config=config,
    )

    flagged = [a for a in analyses if a.needs_correction]
    candidates = [c for a in flagged for c in a.candidates]

    if not candidates:
        logger.info("No candidates found — returning original transcript unchanged")
        report = CorrectionReport(
            file_id=file_id,
            corrections_attempted=0,
            corrections_applied=0,
            results=[],
            processing_time_ms=0,
        )
        return transcript, report

    logger.info("Step 2: Found %d candidates in %d/%d segments",
                len(candidates), len(flagged), len(analyses))
    for i, c in enumerate(candidates[:10]):
        logger.info("  Candidate %d: '%s' (error='%s', category='%s')",
                     i + 1, c.term, c.error_found, c.category)
    if len(candidates) > 10:
        logger.info("  ... and %d more candidates (showing first 10)", len(candidates) - 10)

    # Step 2.5: Topic/field classification
    topic_info = _classify_topic(transcript, vocab_terms, candidates, model, tokenizer)

    # Step 2.6: Web vocab enrichment (search web using description, extract vocab with Qwen)
    web_vocab = _enrich_vocab_from_web(topic_info, model, tokenizer)

    # Step 3: Quick OCR — grab ~10 evenly spaced frames, OCR them for screen text
    targeted_ocr_provider = ocr_provider
    quick_ocr_hints = []

    if not ocr_provider and video_url and not config.dry_run:
        try:
            from .video_frames import extract_frames_periodic, get_video_duration
            from .ocr_extractor import _ocr_single_frame

            t3 = time.time()
            num_frames = config.quick_ocr_num_frames
            duration = get_video_duration(video_url)
            if duration and duration > 0:
                interval = max(duration / num_frames, 5.0)
                logger.info("Step 3: Quick OCR — extracting %d frames (every %.0fs from %.0fs video)",
                            num_frames, interval, duration)

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

    # Step 3.5: Pre-collect AVSR hints for flagged segments (before batch correction)
    lip_hints = {}  # (ts_start, ts_end) → hint_string
    avsr_needs = [a for a in flagged if a.needs_avsr]
    if avsr_provider and video_url and avsr_needs and not config.dry_run:
        avsr_pre = avsr_needs[:config.avsr_max_segments]
        logger.info("Step 3.5: Pre-collecting AVSR hints for %d segments", len(avsr_pre))
        for analysis in avsr_pre:
            try:
                hint = avsr_provider.analyze_segment(
                    video_url, analysis.start, analysis.end
                )
                if hint and hint.face_detected:
                    lip_hints[(analysis.start, analysis.end)] = hint.to_prompt_hint()
            except Exception as e:
                logger.warning("AVSR pre-collection failed for [%.1f-%.1f]: %s",
                             analysis.start, analysis.end, e)
        if lip_hints:
            logger.info("Step 3.5: Collected %d AVSR hints", len(lip_hints))
    else:
        logger.info("Step 3.5: AVSR pre-collection skipped")

    # Step 4: Batch correction — send whole transcript chunks to model
    logger.info("Step 4: Batch correction (whole transcript, not word-by-word)...")
    enhanced, report = correct_transcript_batch(
        transcript, file_id, vocab_terms,
        ocr_provider=targeted_ocr_provider,
        config=config,
        lip_hints=lip_hints,
    )
    logger.info("Step 4 complete: %d corrections applied in %.0fms",
                report.corrections_applied, report.processing_time_ms)

    # Step 4.5: Whisper Pass 2 — re-transcribe flagged segments with vocab hints
    if config.whisper_pass2_enabled and video_url and report.results and not config.dry_run:
        try:
            from .whisper_pass2 import run_whisper_pass2, build_initial_prompt

            # Build initial prompt from all vocab sources
            topic_suggested = topic_info.get("suggested_vocab", []) if topic_info else []
            custom_terms = [t["term"] for t in vocab_terms]
            # Extract speaker names from OCR hints
            ocr_names = [h for h in quick_ocr_hints if len(h.split()) <= 3 and h[0].isupper()][:10]
            web_terms = [v.get("term", "") for v in web_vocab] if web_vocab else []

            initial_prompt = build_initial_prompt(
                topic_vocab=topic_suggested,
                custom_vocab=custom_terms,
                ocr_names=ocr_names,
                web_vocab=web_terms,
            )
            logger.info("Step 4.5: Whisper initial_prompt (%d chars): %s", len(initial_prompt), initial_prompt[:200])

            enhanced, whisper_results = run_whisper_pass2(
                video_url=video_url,
                original_transcript=transcript,
                enhanced_transcript=enhanced,
                report=report,
                initial_prompt=initial_prompt,
                config=config,
            )

            # Update report with Whisper info
            agreements = sum(1 for r in whisper_results if r.agreement == "agree")
            disagreements = sum(1 for r in whisper_results if r.agreement == "disagree")
            unclear = sum(1 for r in whisper_results if r.agreement == "unclear")
            report.whisper_pass2_info = {
                "segments_retranscribed": len(set((r.segment_start, r.segment_end) for r in whisper_results)),
                "corrections_checked": len(whisper_results),
                "agreements": agreements,
                "disagreements_reverted": disagreements,
                "unclear_kept": unclear,
            }
        except ImportError:
            logger.warning("Step 4.5: faster-whisper not installed — skipping Whisper Pass 2")
        except Exception as e:
            logger.warning("Step 4.5: Whisper Pass 2 failed — %s", e)
    else:
        if not config.whisper_pass2_enabled:
            logger.info("Step 4.5: Whisper Pass 2 disabled")
        elif not video_url:
            logger.info("Step 4.5: No video URL — skipping Whisper Pass 2")
        else:
            logger.info("Step 4.5: No corrections to verify — skipping Whisper Pass 2")

    # Step 5: AVSR pass on flagged segments
    avsr_needs = [a for a in flagged if a.needs_avsr]
    if avsr_provider and video_url and avsr_needs and not config.dry_run:
        from .corrector import _fetch_ocr_hints
        from .model import build_prompt, run_inference

        avsr_needs = avsr_needs[:config.avsr_max_segments]
        logger.info("Step 5: AVSR on %d segments (mode=%s)", len(avsr_needs), config.avsr_mode)

        avsr_applied = 0
        for i, analysis in enumerate(avsr_needs):
            for candidate in analysis.candidates:
                try:
                    hint = avsr_provider.analyze_segment(
                        video_url, candidate.timestamp_start, candidate.timestamp_end
                    )
                    if hint is None:
                        continue

                    lip_hint_str = hint.to_prompt_hint()
                    ocr_hints = _fetch_ocr_hints(candidate, file_id, targeted_ocr_provider, config)

                    prompt = build_prompt(
                        candidate.context, [candidate.term], candidate.category,
                        ocr_hints, lip_hint=lip_hint_str,
                    )
                    new_result = run_inference(
                        prompt, config.system_prompt, model, tokenizer, config.max_tokens
                    )

                    new_confidence = new_result.get("confidence", 0.0)
                    new_changes = new_result.get("changes", [])
                    if new_confidence >= config.confidence_threshold and new_changes:
                        import re
                        pattern = re.compile(
                            r'\b' + re.escape(candidate.error_found) + r'\b',
                            re.IGNORECASE,
                        )
                        text = enhanced.get("text", "")
                        new_text = pattern.sub(candidate.term, text, count=1)
                        if new_text != text:
                            enhanced["text"] = new_text
                            for seg in enhanced.get("segments", []):
                                seg_new = pattern.sub(candidate.term, seg.get("text", ""), count=1)
                                if seg_new != seg.get("text", ""):
                                    seg["text"] = seg_new
                                    break
                            avsr_applied += 1
                except Exception as e:
                    logger.warning("AVSR failed for '%s': %s", candidate.term, e)

        logger.info("Step 5 complete: AVSR applied %d corrections", avsr_applied)
        report.corrections_applied += avsr_applied
    else:
        logger.info("Step 5: AVSR skipped (provider=%s, video=%s, segments=%d)",
                    bool(avsr_provider), bool(video_url), len(avsr_needs) if avsr_needs else 0)

    # Step 6: Collect data for future training
    if config.collect_data and report.results:
        logger.info("Step 6: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )

    # Add selector metadata to report
    report.selector_info = {
        "total_segments": len(analyses),
        "flagged_segments": len(flagged),
        "needs_ocr": len(quick_ocr_hints),
        "needs_avsr": len(avsr_needs) if avsr_needs else 0,
        "candidates_found": len(candidates),
    }
    report.topic_info = topic_info

    total_ms = (time.time() - pipeline_start) * 1000
    report.processing_time_ms = total_ms
    logger.info("Pipeline complete in %.0fms (selector → targeted OCR → correction)", total_ms)

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


def _enrich_vocab_from_web(topic_info, model, tokenizer):
    """Step 2.6: Search web using topic description, extract vocab with Qwen."""
    import json as _json
    import re as _re
    from .model import run_inference_raw

    description = topic_info.get("description", "")
    field = topic_info.get("field", "")
    topic = topic_info.get("topic", "")

    if not description or model is None or tokenizer is None:
        logger.info("Step 2.6: Skipped — no description or no model")
        return []

    # Search DuckDuckGo
    logger.info("=" * 60)
    logger.info("Step 2.6: WEB VOCAB ENRICHMENT")

    snippets = []
    try:
        from ddgs import DDGS

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

    except ImportError:
        logger.warning("  ddgs not installed — skipping web search")
        logger.info("=" * 60)
        return []

    if not snippets:
        logger.info("  No search results found")
        logger.info("=" * 60)
        return []

    web_context = " ".join(snippets)[:2500]
    logger.info("  Got %d chars of web context from %d snippets", len(web_context), len(snippets))

    # Send to Qwen to extract vocab
    extract_prompt = (
        f"This meeting is about: {description}\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n\n"
        f"Web search context about this domain:\n{web_context}\n\n"
        "Based on this, list domain-specific terms that ASR (speech-to-text) commonly gets wrong.\n\n"
        "Return ONLY a JSON array:\n"
        '[{"term": "Groq", "category": "company_name", "known_errors": ["Grok", "GROC"]}, ...]\n\n'
        "Categories: product_name, tech_term, person_name, company_name, domain_term, tech_acronym, business_term\n"
        "Give 10-20 terms. Focus on phonetically ambiguous terms. Pure JSON array, nothing else."
    )

    system_prompt = "You extract domain-specific vocabulary from web search results. Return only a JSON array."

    try:
        raw_response = run_inference_raw(
            extract_prompt, system_prompt, model, tokenizer, max_tokens=512,
        )

        # Parse JSON array from response
        json_match = _re.search(r'\[.*\]', raw_response, _re.DOTALL)
        if json_match:
            vocab = _json.loads(json_match.group())
            logger.info("  Extracted %d vocab terms:", len(vocab))
            for v in vocab:
                logger.info("    %s (%s) — errors: %s",
                           v.get("term", "?"), v.get("category", "?"),
                           v.get("known_errors", []))
            logger.info("=" * 60)
            return vocab
        else:
            logger.warning("  No JSON array found in model response")
            logger.info("  Raw response: %s", raw_response[:300])
    except Exception as e:
        logger.warning("  Vocab extraction failed: %s", e)

    logger.info("=" * 60)
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
