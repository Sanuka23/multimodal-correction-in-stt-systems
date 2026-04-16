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

    # Step 2.7: Per-candidate web search validation
    # When the LLM proposes a correction target that isn't in the vocab,
    # web-search for the error word to find a better-known entity.
    # E.g. "Post-Sog" → search → find "PostHog" → override "Post-SOC"
    _validate_candidates_via_web(candidates, vocab_terms, topic_info)

    # Step 3: Smart OCR — targeted frames at error timestamps + evenly-spaced fallback
    targeted_ocr_provider = ocr_provider
    quick_ocr_hints = []

    if not ocr_provider and video_url and not config.dry_run:
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

    # Step 3.5: Extract structured vocab from OCR text (names, products, companies)
    ocr_vocab = []
    if quick_ocr_hints and model and not config.dry_run:
        ocr_vocab = _extract_vocab_from_ocr(quick_ocr_hints, model, tokenizer)

    # Step 4: Whisper Pass 2 — re-transcribe flagged segments with vocab hints
    whisper_segments = {}
    if video_url and flagged and not config.dry_run:
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

    # Step 5: LLM Reconciliation — compare original vs Whisper, pick best words
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
            )
        except Exception as e:
            logger.warning("Step 5: Reconciliation failed — %s", e)
            enhanced = dict(transcript)
    else:
        logger.info("Step 5: No Whisper segments to reconcile — returning original")

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
    if config.collect_data and report.results:
        logger.info("Step 6: Collecting %d results for training data", len(report.results))
        collect_correction_data(
            report.results, config.system_prompt, config.data_output_dir
        )

    # Add metadata to report
    report.selector_info = {
        "total_segments": len(analyses),
        "flagged_segments": len(flagged),
        "ocr_hints": len(quick_ocr_hints),
        "whisper_segments": len(whisper_segments),
        "candidates_found": len(candidates),
    }
    report.topic_info = topic_info

    total_ms = (time.time() - pipeline_start) * 1000
    report.processing_time_ms = total_ms
    logger.info("Pipeline complete in %.0fms", total_ms)

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

    # Send to Qwen to extract vocab (simplified format — no known_errors to save tokens)
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

    system_prompt = "You extract domain-specific vocabulary. Return only a compact JSON array."

    try:
        raw_response = run_inference_raw(
            extract_prompt, system_prompt, model, tokenizer, max_tokens=1024,
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
