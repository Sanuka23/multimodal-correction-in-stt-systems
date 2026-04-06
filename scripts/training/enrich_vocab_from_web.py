#!/usr/bin/env python3
"""
scripts/enrich_vocab_from_web.py

Enrich domain_vocab.json with web-sourced vocabulary for ASR correction.

Searches the web (DuckDuckGo + Wikipedia) for domain-specific terminology,
uses Claude Haiku to extract terms with likely ASR misrecognitions,
and merges them into asr_correction/domain_vocab.json.

Install:
  pip install ddgs wikipedia-api --break-system-packages

Usage:
  python scripts/enrich_vocab_from_web.py --dry-run
  python scripts/enrich_vocab_from_web.py --video screenapp_migration_kimi
  python scripts/enrich_vocab_from_web.py --all
  python scripts/enrich_vocab_from_web.py --all --no-api
"""

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Meeting topic definitions ──────────────────────────────────────────────

MEETING_TOPICS = {
    "aws_migration": {
        "field": "Cloud Computing / SaaS",
        "topic": "AWS migration strategy, cloud infrastructure, marketplace listing",
        "description": "ScreenApp meeting with AWS SMB team discussing GCP to AWS migration, SOC 2, ISV Accelerate, Bedrock LLMs, containerized infrastructure, B2B growth.",
        "known_terms": ["AWS", "GCP", "SOC 2", "ISV Accelerate", "Bedrock", "CloudCompass", "Ingram", "NextGen", "ATLA", "ScreenApp", "Kieran", "Mimi", "Bud"],
    },
    "followup_julien": {
        "field": "SaaS / Product Onboarding",
        "topic": "ScreenApp product onboarding, API integration, mobile platforms",
        "description": "Onboarding call with Julien: transcription, knowledge base, API webhooks, iOS/Android, Notion, Confluence, Google Drive, Growth plan, APAC.",
        "known_terms": ["ScreenApp", "Notion", "Confluence", "Google Drive", "webhooks", "SDK", "iOS", "Android", "APAC", "Julien", "Philippines"],
    },
    "screenapp_migration_kimi": {
        "field": "AI/ML / SaaS Infrastructure",
        "topic": "LLM model benchmarking, transcription platform, diarization",
        "description": "ScreenApp team discussing Groq, Kimi K2, Gemini Flash, Fireworks, Whisper, diarization, VAD, Cloudflare, Qdrant, Elasticsearch, NDCG, canary deployment.",
        "known_terms": ["Groq", "Kimi", "Gemini", "Fireworks", "Whisper", "Cloudflare", "Qdrant", "Elasticsearch", "VAD", "diarization", "NDCG", "pyannote", "LLM", "OCR", "Vicky", "Dinuka", "Sheldon"],
    },
    "troubleshooting_dimiter": {
        "field": "SaaS / Customer Support",
        "topic": "ASR troubleshooting, accent support, transcript quality",
        "description": "Troubleshooting call with Dimiter: inconsistent ScreenApp transcripts, Bulgarian accent recognition, Growth plan features.",
        "known_terms": ["ScreenApp", "Dimitar", "Bulgarian", "transcript", "summary", "Growth plan", "Arissa", "monitoring evaluation"],
    },
    "compliance_discussion": {
        "field": "SaaS / Compliance and Security",
        "topic": "SOC 2, HIPAA, GDPR compliance for SaaS meeting tools",
        "description": "Compliance discussion: SOC 2 Type 2, HIPAA, GDPR, ISO 27001, Vanta, Atlassian, GitLab, Fireflies, WeWork, Australian office locations.",
        "known_terms": ["SOC 2", "HIPAA", "GDPR", "ISO", "Vanta", "Atlassian", "GitLab", "Fireflies", "Bitbucket", "Coviu", "Anthropic", "Martin Place", "Barangaroo", "VISO"],
    },
    "gcp_security": {
        "field": "Cloud Security",
        "topic": "GCP Security Command Center, CNAPP, multi-cloud compliance",
        "description": "GCP Security Command Center, CNAPP comparison with Wiz, multi-cloud security, SOC 2, HIPAA, ring-fence architecture.",
        "known_terms": ["GCP", "Security Command Center", "SCC", "CNAPP", "Wiz", "HIPAA", "SOC 2", "Azure", "AWS", "ring-fence", "Erin"],
    },
    "onboarding_andre": {
        "field": "SaaS / Business Operations",
        "topic": "Product analytics, churn management, mobile development",
        "description": "Onboarding: PostHog, ChartMogul, Churnkey, MRR, Stripe, Flutter, sprint planning, SSO, API, SEO, Google Chat, Front.",
        "known_terms": ["PostHog", "ChartMogul", "Churnkey", "Stripe", "Flutter", "MRR", "SSO", "SEO", "Google Chat", "Front", "ScreenApp", "Avindi", "Callum"],
    },
    "project_update": {
        "field": "AI/ML / Software Development",
        "topic": "Vector search, RAG pipeline, Kubernetes, semantic ranking",
        "description": "Team update: RAG with Typesense/Qdrant, semantic ranker, re-ranker, Vertex AI, Gemini, Kubernetes on GCP/ECS, markdown, PR reviews, meeting bot.",
        "known_terms": ["Typesense", "Qdrant", "Vertex AI", "Gemini", "Kubernetes", "RAG", "re-ranker", "semantic ranker", "NDCG", "embeddings", "GCP", "ECS", "CLI", "Dilushan"],
    },
    "business_discussion": {
        "field": "Business Strategy / SaaS",
        "topic": "SaaS metrics, video editing tools, startup ecosystem",
        "description": "Business discussion: LTV, CAC, churn, MRR, Adobe Premiere, DaVinci Resolve, Final Cut, Canva, Leonardo AI, Cursor, Antler, Deloitte, Australia startups.",
        "known_terms": ["Adobe Premiere", "DaVinci Resolve", "Final Cut", "Canva", "Leonardo AI", "Cursor", "Antler", "Deloitte", "LTV", "CAC", "SIP", "ISO", "Eucalyptus"],
    },
    "zachary_onboarding": {
        "field": "SaaS / Consumer / Content Creation",
        "topic": "YouTube integration, ASR transcription features, content creator tools",
        "description": "Zachary onboarding: YouTube integration, Whisper/Gemini transcription, video analyzer, SEO, ChatGPT comparison, HIPAA, Primitive Technology channel.",
        "known_terms": ["ScreenApp", "YouTube", "Whisper", "Gemini", "ChatGPT", "HIPAA", "SEO", "Zachary", "Logan", "Primitive Technology", "My Self-Reliance"],
    },
}

# ── LLM extraction prompt ─────────────────────────────────────────────────

EXTRACTION_PROMPT = """I have a meeting transcript classified as:
Field: {field}
Topic: {topic}
Description: {description}

Web search context about this domain:
{web_context}

Already known terms in this meeting:
{known_terms}

The ASR system commonly misrecognizes domain-specific terms.
Based on the field, topic, and web context above, give me a comprehensive vocabulary list.

Return ONLY a valid JSON array, no explanation, no markdown backticks.
Format:
[
  {{"term": "Groq", "category": "company_name", "known_errors": ["Grok", "GROC", "Proc"]}},
  {{"term": "diarization", "category": "tech_term", "known_errors": ["diary station", "die realization"]}}
]

Categories (use exactly these strings):
product_name, tech_term, person_name, company_name, domain_term,
compliance, tech_acronym, business_term, feature, cloud_service

Rules:
- Focus on terms that are phonetically ambiguous or hard for ASR
- Include ALL known_terms listed above (with their ASR errors)
- Add 20-30 more terms from the web context
- known_errors = what ASR would actually transcribe incorrectly
- Return 30-50 terms total
- Pure JSON array, nothing else
"""


# ── Web search ─────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 8) -> list:
    """Search DuckDuckGo. Returns list of {title, href, body} dicts."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except ImportError:
        print("  [!] ddgs not installed. Run: pip install ddgs --break-system-packages")
        return []
    except Exception as e:
        print(f"  [!] Search failed: {e}")
        return []


def wiki_summary(topic: str) -> str:
    """Get Wikipedia summary for a topic."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='ASR-VocabBuilder/1.0'
        )
        page = wiki.page(topic)
        return page.summary[:1000] if page.exists() else ""
    except ImportError:
        return ""
    except Exception:
        return ""


def get_search_context(topic_info: dict) -> str:
    """Run 3 targeted searches + Wikipedia, return combined text ≤3000 chars."""
    field = topic_info["field"]
    topic = topic_info["topic"]
    known = ", ".join(topic_info.get("known_terms", [])[:6])

    queries = [
        f"{field} technical terminology jargon acronyms 2024",
        f"{topic} common terms glossary",
        f"speech recognition errors {known}",
    ]

    snippets = []
    for q in queries:
        results = web_search(q, max_results=6)
        for r in results:
            snippets.append(r.get("body", ""))
        time.sleep(0.5)

    wiki = wiki_summary(field.split("/")[0].strip())
    if wiki:
        snippets.append(wiki)

    return " ".join(snippets)[:3000]


# ── LLM extraction (local Qwen) ───────────────────────────────────────────

def _load_qwen():
    """Load the local Qwen model (singleton)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from asr_correction.model import load_model
    return load_model()


def extract_vocab_with_qwen(topic_info: dict, web_context: str) -> list:
    """Use local Qwen model to extract vocab terms from web context."""
    import sys
    sys.path.insert(0, str(ROOT))
    from asr_correction.model import run_inference_raw

    model, tokenizer = _load_qwen()
    if model is None:
        print("  [!] Could not load Qwen model")
        return []

    prompt = EXTRACTION_PROMPT.format(
        field=topic_info["field"],
        topic=topic_info["topic"],
        description=topic_info["description"],
        web_context=web_context[:2500],
        known_terms=", ".join(topic_info.get("known_terms", [])),
    )

    system_prompt = "You extract domain-specific vocabulary from web search results. Return only a JSON array."

    try:
        raw = run_inference_raw(prompt, system_prompt, model, tokenizer, max_tokens=512)
        # Parse JSON array
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        print(f"  [!] No JSON array in response: {raw[:200]}")
        return []
    except Exception as e:
        print(f"  [!] Qwen extraction failed: {e}")
        return []


# ── Process one topic ─────────────────────────────────────────────────────

def process_topic(video_name: str, topic_info: dict, dry_run: bool = False):
    """Full pipeline: search → extract with Qwen → log results."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"Field: {topic_info['field']}")
    print(f"{'='*60}")

    # Step 1: Web search
    print(f"\n[1/3] Searching web...")
    web_context = get_search_context(topic_info)
    print(f"  Got {len(web_context)} chars of context")

    if dry_run:
        print(f"\n  [DRY RUN] Context preview:\n  {web_context[:300]}...")
        return

    # Step 2: Extract vocab with local Qwen
    print(f"\n[2/2] Extracting vocab with Qwen...")
    vocab = extract_vocab_with_qwen(topic_info, web_context)
    if vocab:
        print(f"  Extracted {len(vocab)} terms:")
        for v in vocab:
            print(f"    {v.get('term', '?')} ({v.get('category', '?')}) — errors: {v.get('known_errors', [])}")
    else:
        print("  No terms extracted")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich domain_vocab.json with web-sourced vocabulary")
    parser.add_argument("--video", type=str, help="Process a single video by name")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument("--dry-run", action="store_true", help="Test web search only")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN: Testing web search ===")
        topic = MEETING_TOPICS["screenapp_migration_kimi"]
        process_topic("screenapp_migration_kimi", topic, dry_run=True)
        return

    if args.video:
        if args.video not in MEETING_TOPICS:
            print(f"Unknown video: {args.video}")
            print(f"Available: {', '.join(MEETING_TOPICS.keys())}")
            return
        process_topic(args.video, MEETING_TOPICS[args.video])

    elif args.all:
        for name, topic in MEETING_TOPICS.items():
            process_topic(name, topic)
            time.sleep(1)

        print(f"\n{'='*60}")
        print(f"DONE — processed {len(MEETING_TOPICS)} videos")
        print(f"{'='*60}")

    else:
        parser.print_help()
        print(f"\nAvailable videos: {', '.join(MEETING_TOPICS.keys())}")


if __name__ == "__main__":
    main()
