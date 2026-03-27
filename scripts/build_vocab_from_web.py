#!/usr/bin/env python3
"""
scripts/build_vocab_from_web.py

Automatically build domain-specific vocabulary for ASR correction training
by searching the web based on meeting topic classifications.

Uses:
  - ddgs (DuckDuckGo search, FREE, no API key)
  - wikipedia (FREE, structured domain knowledge)
  - Claude API or local Qwen (to extract + format vocab terms)

Install:
  pip install ddgs wikipedia-api requests --break-system-packages

Usage:
  python scripts/build_vocab_from_web.py
  python scripts/build_vocab_from_web.py --video aws_migration
  python scripts/build_vocab_from_web.py --all --output data/vocab/web_vocab.json
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

# ── Meeting topic definitions ──────────────────────────────────────────────
# Each video's field/topic/description — source of truth for vocab generation

MEETING_TOPICS = {
    "aws_migration": {
        "field": "Cloud Computing / SaaS",
        "topic": "AWS migration strategy, cloud infrastructure, marketplace listing",
        "description": (
            "ScreenApp meeting with AWS SMB account team discussing migration from GCP to AWS, "
            "SOC 2 compliance, ISV Accelerate program, AWS Marketplace listing, "
            "Bedrock LLMs, containerized infrastructure, B2B growth strategy."
        ),
        "key_entities": ["AWS", "GCP", "SOC 2", "ISV Accelerate", "Bedrock", "CloudCompass",
                         "Ingram", "NextGen", "ATLA", "ScreenApp"],
    },
    "followup_julien": {
        "field": "SaaS / Product Onboarding",
        "topic": "ScreenApp product onboarding, API integration, mobile platforms",
        "description": (
            "Onboarding call with Julien discussing ScreenApp features: transcription, "
            "knowledge base, API webhooks, iOS/Android integration, Notion/Confluence/Google Drive "
            "integrations, Growth plan pricing, APAC market."
        ),
        "key_entities": ["ScreenApp", "Notion", "Confluence", "Google Drive", "Julien",
                         "webhooks", "SDK", "iOS", "Android", "APAC"],
    },
    "screenapp_migration_kimi": {
        "field": "AI/ML / SaaS Infrastructure",
        "topic": "AI model comparison, transcription platform infrastructure, LLM evaluation",
        "description": (
            "ScreenApp team meeting discussing LLM providers (Groq, Kimi K2, Gemini Flash, Fireworks), "
            "transcription features (diarization, VAD, Whisper), infrastructure (Cloudflare, Qdrant, "
            "Elasticsearch), NDCG metrics, canary deployment, WhatsApp integration."
        ),
        "key_entities": ["Groq", "Kimi", "Gemini", "Fireworks", "Whisper", "Cloudflare",
                         "Qdrant", "Elasticsearch", "VAD", "diarization", "pyannote",
                         "NDCG", "ScreenApp", "Vicky", "Dinuka", "Sheldon"],
    },
    "troubleshooting_dimiter": {
        "field": "SaaS / Customer Support",
        "topic": "ScreenApp transcription troubleshooting, Bulgarian accent support",
        "description": (
            "Troubleshooting call with Dimiter Ivanov discussing inconsistent ScreenApp transcripts, "
            "Bulgarian accent recognition, monitoring evaluation, Growth plan features, "
            "transcript/summary quality issues."
        ),
        "key_entities": ["ScreenApp", "Dimitar", "Bulgarian", "monitoring evaluation",
                         "Growth plan", "transcript", "summary", "Arisa"],
    },
    "compliance_discussion": {
        "field": "SaaS / Compliance & Security",
        "topic": "SOC 2, HIPAA, GDPR compliance for SaaS meeting tools",
        "description": (
            "Compliance discussion between Matt and Tim covering SOC 2 Type 2, HIPAA, GDPR, "
            "ISO 27001 certifications, Vanta compliance platform, B2B SaaS security requirements, "
            "Atlassian, GitLab, Fireflies, WeWork, Barangaroo office."
        ),
        "key_entities": ["SOC 2", "HIPAA", "GDPR", "ISO", "Vanta", "Atlassian",
                         "GitLab", "Fireflies", "Bitbucket", "Coviu", "Anthropic",
                         "Martin Place", "Barangaroo"],
    },
    "gcp_security": {
        "field": "Cloud Security",
        "topic": "GCP Security Command Center, CNAPP, cloud compliance",
        "description": (
            "Meeting about Google Cloud Platform Security Command Center features: "
            "CNAPP, Wiz comparison, multi-cloud security, SOC 2, HIPAA compliance, "
            "ring-fence architecture, SCC setup."
        ),
        "key_entities": ["GCP", "Security Command Center", "SCC", "CNAPP", "Wiz",
                         "HIPAA", "SOC 2", "Azure", "AWS", "Stefan", "Shaquille", "Erin"],
    },
    "onboarding_andre": {
        "field": "SaaS / Business Operations",
        "topic": "ScreenApp onboarding, product analytics, churn management",
        "description": (
            "Onboarding meeting with Andre, Bud, and Avindi covering ScreenApp product features, "
            "PostHog analytics, ChartMogul metrics, Churnkey, MRR tracking, Stripe payments, "
            "Flutter mobile app, sprint planning, SSO, API integration."
        ),
        "key_entities": ["ScreenApp", "PostHog", "ChartMogul", "Churnkey", "Stripe",
                         "Flutter", "MRR", "SSO", "SEO", "Google Chat", "Front",
                         "Arvindi", "Callum", "Arissa"],
    },
    "project_update": {
        "field": "AI/ML / Software Development",
        "topic": "Vector search, RAG pipeline, Kubernetes deployment",
        "description": (
            "Developer team project update covering RAG implementation with Typesense/Qdrant, "
            "semantic ranker, re-ranker, Vertex AI, Gemini integration, Kubernetes on GCP/ECS, "
            "markdown rendering, PR reviews, meeting bot feature."
        ),
        "key_entities": ["Typesense", "Qdrant", "Vertex AI", "Gemini", "Kubernetes",
                         "RAG", "re-ranker", "semantic ranker", "NDCG", "embeddings",
                         "GCP", "ECS", "AWS", "CLI", "Dilushan", "Arju"],
    },
    "business_discussion": {
        "field": "Business Strategy / SaaS",
        "topic": "SaaS business strategy, video editing tools, startup ecosystem",
        "description": (
            "Business strategy discussion between Jim and Matt covering SaaS metrics "
            "(LTV, CAC, churn, MRR), video editing tools (Adobe Premiere, DaVinci Resolve, "
            "Final Cut, Canva), AI companies (Leonardo AI, Cursor, Antler), "
            "Australian startup ecosystem, landing page optimization."
        ),
        "key_entities": ["Adobe Premiere", "DaVinci Resolve", "Final Cut", "Canva",
                         "Leonardo AI", "Cursor", "Antler", "Deloitte", "ReachOut",
                         "Eucalyptus", "Juniper", "LTV", "CAC", "SIP", "ISO",
                         "Sri Lanka", "Philippines", "Melbourne", "Sydney"],
    },
    "zachary_onboarding": {
        "field": "SaaS / Consumer",
        "topic": "ScreenApp consumer onboarding, YouTube content creation, ASR features",
        "description": (
            "Zachary Jacobson onboarding call covering ScreenApp YouTube integration, "
            "Whisper/Gemini transcription, video analyzer feature, SEO for content creators, "
            "Australia/Philippines market, ChatGPT comparison, HIPAA compliance questions, "
            "Primitive Technology channel, My Self-Reliance."
        ),
        "key_entities": ["ScreenApp", "YouTube", "Whisper", "Gemini", "ChatGPT",
                         "Google Gemini", "HIPAA", "Zachary", "Logan",
                         "Primitive Technology", "My Self-Reliance", "Australia"],
    },
}


# ── Web search ─────────────────────────────────────────────────────────────

def search_ddg(query: str, max_results: int = 10) -> list[dict]:
    """Search DuckDuckGo. Returns list of {title, href, body} dicts."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except ImportError:
        print("  ddgs not installed. Run: pip install ddgs --break-system-packages")
        return []
    except Exception as e:
        print(f"  DDG search failed: {e}")
        return []


def search_wikipedia(topic: str) -> str:
    """Get Wikipedia summary for a topic."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='ScreenApp-ASR-VocabBuilder/1.0'
        )
        page = wiki.page(topic)
        if page.exists():
            # Return first 1000 chars of summary
            return page.summary[:1000]
    except ImportError:
        try:
            import wikipedia
            result = wikipedia.summary(topic, sentences=5, auto_suggest=True)
            return result
        except Exception:
            pass
    except Exception:
        pass
    return ""


def search_for_topic(topic_info: dict) -> str:
    """Run multiple targeted searches and compile a context blob."""
    field = topic_info["field"]
    topic = topic_info["topic"]
    entities = topic_info.get("key_entities", [])

    context_parts = []

    # Search 1: general domain terms
    query1 = f"{field} technical terminology jargon acronyms"
    results1 = search_ddg(query1, max_results=5)
    for r in results1:
        context_parts.append(r.get("body", ""))

    # Search 2: specific topic
    query2 = f"{topic} common terms vocabulary"
    results2 = search_ddg(query2, max_results=5)
    for r in results2:
        context_parts.append(r.get("body", ""))

    # Search 3: key entities for ASR confusion
    if entities:
        entity_sample = ", ".join(entities[:8])
        query3 = f"speech recognition errors {entity_sample}"
        results3 = search_ddg(query3, max_results=3)
        for r in results3:
            context_parts.append(r.get("body", ""))

    # Wikipedia for field overview
    wiki_text = search_wikipedia(field.split("/")[0].strip())
    if wiki_text:
        context_parts.append(wiki_text)

    # Combine and deduplicate
    combined = " ".join(context_parts)
    # Truncate to avoid token limits
    return combined[:3000]


# ── LLM vocab extraction ───────────────────────────────────────────────────

VOCAB_PROMPT = """I have a meeting transcript that has been classified as:
Field: {field}
Topic: {topic}
Description: {description}

Additional context from web search:
{web_context}

Key entities already identified in this meeting:
{entities}

The ASR (speech-to-text) system commonly misrecognizes domain-specific terms in this field.
Based on this meeting's field, topic, and the web context above, give me a comprehensive list of:

1. **Product/Company Names** commonly discussed in this domain
   (e.g., specific tools, platforms, services, startups, companies)
2. **Technical Terms** specific to this field
   (e.g., jargon, acronyms, methodologies, protocols)
3. **Person Names** commonly referenced in this space
   (e.g., founders, CEOs, researchers, thought leaders)
4. **Common ASR Misrecognitions** for each term
   (what the speech-to-text system might mishear them as)

Return ONLY a JSON array. No explanation, no markdown. Example format:
[
  {{"term": "Groq", "category": "company_name", "known_errors": ["Grok", "GROC", "Proc", "Grop"]}},
  {{"term": "diarization", "category": "tech_term", "known_errors": ["diarisation", "diary station", "die realization"]}},
  {{"term": "SOC 2", "category": "compliance", "known_errors": ["SOC", "sock 2", "sock too"]}}
]

Categories must be one of: product_name, tech_term, person_name, company_name, domain_term, compliance, tech_acronym, business_term, feature, cloud_service

Focus on terms that are phonetically ambiguous or likely to be misheard. Give me 30-50 terms.
Include the key entities listed above plus additional relevant terms from the web context.
"""


def extract_vocab_with_claude_api(topic_info: dict, web_context: str) -> list[dict]:
    """Use Claude API (claude-haiku-4-5) to extract vocab terms."""
    import requests

    prompt = VOCAB_PROMPT.format(
        field=topic_info["field"],
        topic=topic_info["topic"],
        description=topic_info["description"],
        web_context=web_context[:2000],
        entities=", ".join(topic_info.get("key_entities", [])),
    )

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _get_api_key(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        data = response.json()
        text = data["content"][0]["text"].strip()

        # Parse JSON from response
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return json.loads(text)

    except Exception as e:
        print(f"  Claude API failed: {e}")
        return []


def extract_vocab_with_local_model(topic_info: dict, web_context: str) -> list[dict]:
    """Use local Qwen model via MLX as fallback."""
    try:
        import subprocess
        import tempfile

        prompt = VOCAB_PROMPT.format(
            field=topic_info["field"],
            topic=topic_info["topic"],
            description=topic_info["description"],
            web_context=web_context[:1500],
            entities=", ".join(topic_info.get("key_entities", [])),
        )

        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        result = subprocess.run(
            ["python", "-m", "mlx_lm.generate",
             "--model", "mlx-community/Qwen2.5-7B-Instruct-4bit",
             "--max-tokens", "2000",
             "--prompt", prompt],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT)
        )

        output = result.stdout.strip()
        # Extract JSON from output
        json_match = re.search(r'\[.*\]', output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []

    except Exception as e:
        print(f"  Local model failed: {e}")
        return []


def extract_vocab_rule_based(topic_info: dict) -> list[dict]:
    """Simple rule-based fallback — just return key_entities with guessed errors."""
    vocab = []
    for entity in topic_info.get("key_entities", []):
        # Guess category
        if entity.isupper() and len(entity) <= 5:
            category = "tech_acronym"
        elif any(c.isupper() for c in entity[1:]) or " " in entity:
            category = "product_name"
        elif entity.istitle():
            category = "person_name"
        else:
            category = "tech_term"

        # Guess common ASR errors (simple phonetic rules)
        errors = []
        if entity.isupper():
            errors.append(entity.lower())
        if "2" in entity:
            errors.append(entity.replace("2", " two").strip())
            errors.append(entity.replace("2", "").strip())

        vocab.append({
            "term": entity,
            "category": category,
            "known_errors": errors[:3],
        })
    return vocab


def _get_api_key() -> Optional[str]:
    """Load Anthropic API key from .env."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    import os
    return os.environ.get("ANTHROPIC_API_KEY")


# ── Training pair builder ──────────────────────────────────────────────────

def build_training_pairs_from_vocab(vocab: list[dict], topic_info: dict,
                                     video_name: str) -> list[dict]:
    """
    Convert vocab terms into training pairs for fine-tuning.

    For each term that has known ASR errors:
    - Create a training example showing the error → correction
    - Add web search context as OCR hint (simulating slide/screen context)
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from training.prepare_data import (
        SYSTEM_PROMPT, build_user_prompt, build_assistant_response
    )

    pairs = []
    field = topic_info["field"]
    topic = topic_info["topic"]

    # Build a fake context sentence for each term
    context_templates = [
        "We should integrate {term} into our workflow for better results.",
        "The {term} approach has shown significant improvements in this area.",
        "Let me share my screen to show the {term} dashboard configuration.",
        "According to the latest {term} documentation, this is the recommended way.",
        "Our team has been using {term} extensively for the past few months.",
        "I think {term} would be the best solution for this use case.",
        "The {term} API allows us to process this data efficiently.",
        "We need to discuss the {term} integration before the next sprint.",
    ]

    import random
    random.seed(42)

    for vocab_entry in vocab:
        term = vocab_entry["term"]
        category = vocab_entry["category"]
        errors = vocab_entry.get("known_errors", [])

        if not errors:
            continue

        # Create one training pair per known error
        for error in errors[:3]:  # max 3 pairs per term
            template = random.choice(context_templates)
            gt_text = template.format(term=term)           # ground truth
            asr_text = template.format(term=error)          # what ASR heard

            if gt_text == asr_text:  # skip if error is same as term
                continue

            # Build the training pair
            # Vocab list = other terms from same meeting for context
            other_terms = [v["term"] for v in vocab if v["term"] != term][:15]
            vocab_list = [term] + other_terms[:14]

            # Simulate OCR hint (field/topic context as if on a slide)
            ocr_hints = [f"{field} — {topic[:50]}"]

            user_prompt = build_user_prompt(
                asr_text=asr_text,
                vocab=vocab_list,
                category=category,
                ocr_hints=ocr_hints,
                lip_hint=None,
            )

            assistant_response = build_assistant_response(
                corrected=gt_text,
                changes=[f"{error} → {term}"],
                confidence=0.95,
                need_lip=False,
            )

            pair = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                "metadata": {
                    "source": f"web_vocab_{video_name}",
                    "video": video_name,
                    "term": term,
                    "category": category,
                    "error_found": error,
                    "field": field,
                    "generated_by": "web_search",
                    "applied": True,
                }
            }
            pairs.append(pair)

    # Also add 20% negative examples (no correction needed)
    n_negatives = max(1, len(pairs) // 5)
    for vocab_entry in vocab[:n_negatives]:
        term = vocab_entry["term"]
        category = vocab_entry["category"]
        template = random.choice(context_templates)
        clean_text = template.format(term=term)
        other_terms = [v["term"] for v in vocab if v["term"] != term][:15]
        vocab_list = [term] + other_terms[:14]

        user_prompt = build_user_prompt(
            asr_text=clean_text,
            vocab=vocab_list,
            category=category,
            ocr_hints=[f"{field} — {topic[:50]}"],
        )
        assistant_response = build_assistant_response(
            corrected=clean_text,
            changes=[],
            confidence=0.99,
            need_lip=False,
        )
        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "source": f"web_vocab_{video_name}",
                "video": video_name,
                "term": term,
                "category": category,
                "error_found": None,
                "field": field,
                "generated_by": "web_search",
                "applied": False,
            }
        }
        pairs.append(pair)

    return pairs


# ── Main pipeline ──────────────────────────────────────────────────────────

def process_video(video_name: str, extraction_method: str = "claude_api",
                  output_dir: Path = None) -> dict:
    """
    Full pipeline for one video:
    1. Search web for domain context
    2. Extract vocab with LLM or rules
    3. Build training pairs
    4. Save vocab + pairs
    """
    if video_name not in MEETING_TOPICS:
        print(f"Unknown video: {video_name}. Available: {list(MEETING_TOPICS.keys())}")
        return {}

    topic_info = MEETING_TOPICS[video_name]
    output_dir = output_dir or ROOT / "data/training_pairs/web_vocab"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"Field: {topic_info['field']}")
    print(f"Topic: {topic_info['topic']}")
    print(f"{'='*60}")

    # Step 1: Web search
    print(f"\n[1/3] Searching web for domain context...")
    web_context = search_for_topic(topic_info)
    print(f"  Got {len(web_context)} chars of context")
    time.sleep(1)  # be polite to DDG

    # Step 2: Extract vocab
    print(f"\n[2/3] Extracting vocab terms ({extraction_method})...")
    vocab = []

    if extraction_method == "claude_api" and _get_api_key():
        vocab = extract_vocab_with_claude_api(topic_info, web_context)
        print(f"  Claude API: {len(vocab)} terms extracted")

    if not vocab and extraction_method in ("local_model", "claude_api"):
        print("  Trying local Qwen model...")
        vocab = extract_vocab_with_local_model(topic_info, web_context)
        print(f"  Local model: {len(vocab)} terms extracted")

    if not vocab:
        print("  Using rule-based fallback (key entities only)...")
        vocab = extract_vocab_rule_based(topic_info)
        print(f"  Rule-based: {len(vocab)} terms")

    # Save vocab to JSON
    vocab_path = output_dir / f"{video_name}_vocab.json"
    vocab_path.write_text(json.dumps({
        "video": video_name,
        "field": topic_info["field"],
        "topic": topic_info["topic"],
        "terms": vocab,
        "web_context_snippet": web_context[:500],
    }, indent=2))
    print(f"  Saved vocab: {vocab_path}")

    # Step 3: Build training pairs
    print(f"\n[3/3] Building training pairs...")
    pairs = build_training_pairs_from_vocab(vocab, topic_info, video_name)
    print(f"  Generated {len(pairs)} pairs ({sum(1 for p in pairs if p['metadata']['applied'])} positive, "
          f"{sum(1 for p in pairs if not p['metadata']['applied'])} negative)")

    # Save training pairs
    pairs_path = output_dir / f"{video_name}_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  Saved pairs: {pairs_path}")

    return {
        "video": video_name,
        "vocab_count": len(vocab),
        "pairs_count": len(pairs),
        "vocab_path": str(vocab_path),
        "pairs_path": str(pairs_path),
    }


def merge_all_web_vocab_pairs(output_dir: Path) -> Path:
    """Merge all per-video JSONL files into one combined file."""
    all_pairs = []
    for jsonl_file in sorted(output_dir.glob("*_pairs.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))

    merged_path = output_dir / "web_vocab_all.jsonl"
    with open(merged_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_pairs)} pairs → {merged_path}")
    return merged_path


def inject_into_training_data(web_vocab_path: Path, upsample: int = 2):
    """
    Inject web vocab pairs into the main training dataset.
    Upsamples by 2× since these are synthetic but domain-relevant.
    """
    collected_dir = ROOT / "data/collected_data"
    train_path = collected_dir / "train.jsonl"

    if not train_path.exists():
        print(f"Train file not found: {train_path}")
        return

    # Load existing
    existing = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                existing.append(line.strip())

    # Load web vocab pairs
    web_pairs = []
    with open(web_vocab_path) as f:
        for line in f:
            if line.strip():
                web_pairs.append(line.strip())

    # Add upsampled web pairs
    combined = existing + web_pairs * upsample

    import random
    random.seed(42)
    random.shuffle(combined)

    # Save
    train_path.write_text("\n".join(combined) + "\n")
    print(f"\nInjected {len(web_pairs) * upsample} web vocab pairs into train.jsonl")
    print(f"New total: {len(combined)} examples ({len(existing)} original + {len(web_pairs) * upsample} web vocab)")

    # Update metadata
    metadata_path = collected_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["train_count"] = len(combined)
        metadata["web_vocab_pairs"] = len(web_pairs)
        metadata["web_vocab_upsample"] = upsample
        metadata_path.write_text(json.dumps(metadata, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build web-sourced vocab for ASR training")
    parser.add_argument("--video", type=str, help="Process a single video by name")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--method", type=str, default="claude_api",
                        choices=["claude_api", "local_model", "rules"],
                        help="Vocab extraction method")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for vocab + pairs")
    parser.add_argument("--inject", action="store_true",
                        help="Inject into main training data after building")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test web search without LLM extraction")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else ROOT / "data/training_pairs/web_vocab"

    if args.dry_run:
        print("=== DRY RUN: Testing web search ===")
        topic = MEETING_TOPICS["screenapp_migration_kimi"]
        context = search_for_topic(topic)
        print(f"Got {len(context)} chars")
        print(context[:500])
        return

    if args.video:
        result = process_video(args.video, args.method, output_dir)
        print(f"\nResult: {result}")
    elif args.all:
        results = []
        for video_name in MEETING_TOPICS:
            result = process_video(video_name, args.method, output_dir)
            results.append(result)
            time.sleep(2)  # polite delay between videos

        # Merge all
        merged_path = merge_all_web_vocab_pairs(output_dir)

        if args.inject:
            inject_into_training_data(merged_path, upsample=2)

        # Summary
        print("\n=== Summary ===")
        total_vocab = sum(r.get("vocab_count", 0) for r in results)
        total_pairs = sum(r.get("pairs_count", 0) for r in results)
        print(f"Videos processed: {len(results)}")
        print(f"Total vocab terms: {total_vocab}")
        print(f"Total training pairs: {total_pairs}")
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Test web search only:")
        print("  python scripts/build_vocab_from_web.py --dry-run")
        print("")
        print("  # Process one video using Claude API:")
        print("  python scripts/build_vocab_from_web.py --video aws_migration")
        print("")
        print("  # Process all videos and inject into training data:")
        print("  python scripts/build_vocab_from_web.py --all --inject")
        print("")
        print("  # Use local Qwen model (no API cost):")
        print("  python scripts/build_vocab_from_web.py --all --method local_model")
        print("")
        print("  # Rule-based only (fastest, no LLM):")
        print("  python scripts/build_vocab_from_web.py --all --method rules --inject")


if __name__ == "__main__":
    main()
