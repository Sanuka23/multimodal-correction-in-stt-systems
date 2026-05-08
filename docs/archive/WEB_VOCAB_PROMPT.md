# Claude Code Prompt — Web Search Vocabulary Enrichment

## OBJECTIVE

Build a script `scripts/enrich_vocab_from_web.py` that:
1. Takes a meeting transcript + its topic classification as input
2. Searches the web for domain-specific terminology related to that topic
3. Uses an LLM to extract vocabulary terms + likely ASR misrecognitions from the search results
4. Outputs a clean vocab list in the exact format used by `asr_correction/domain_vocab.json`
5. Merges the new terms into `domain_vocab.json` so the correction pipeline uses them automatically

This is NOT about fine-tuning. This is purely about enriching the vocabulary dictionary that the ASR correction system uses at inference time.

---

## HOW THE VOCAB IS USED (read this first)

Read `asr_correction/vocabulary.py` — the `merge_vocabularies()` function combines:
- `domain_vocab.json` — the static vocabulary file on disk
- `custom_vocabulary` — per-request terms from ScreenApp

Read `asr_correction/domain_vocab.json` — this is the target format:
```json
{
  "version": "1.0.0",
  "terms": [
    {
      "term": "Groq",
      "category": "company_name",
      "known_errors": ["Grok", "GROC", "Proc", "Grop"]
    },
    {
      "term": "diarization",
      "category": "tech_term",
      "known_errors": ["diary station", "die realization", "diarisation"]
    }
  ]
}
```

The goal is to add more terms like this to `domain_vocab.json` by searching the web.

---

## INPUT FORMAT

The script receives a topic classification like this:

```python
topic_info = {
    "field": "AI/ML / SaaS Infrastructure",
    "topic": "LLM model benchmarking and transcription platform",
    "description": "ScreenApp team meeting discussing Groq, Kimi, Gemini Flash, Fireworks, diarization, VAD, Whisper, Cloudflare, Qdrant, NDCG, canary deployment.",
    "known_terms": ["Groq", "Kimi", "Gemini", "Fireworks", "Whisper", "Cloudflare", "Qdrant", "VAD", "diarization"]
}
```

These topic classifications already exist for all 10 ScreenApp meeting videos. Define them as a dict inside the script — no need to read from any file.

---

## WEB SEARCH IMPLEMENTATION

### Library to use: `ddgs` (DuckDuckGo Search)
- **Free, no API key required**
- Install: `pip install ddgs --break-system-packages`
- Latest version as of 2025: `ddgs 8.x` or `ddgs 9.x`

### How to use ddgs:
```python
from ddgs import DDGS

def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Returns list of {title, href, body} dicts."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []
```

### Wikipedia as secondary source (also free):
```python
import wikipediaapi

def wiki_summary(topic: str) -> str:
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='ASR-VocabBuilder/1.0'
    )
    page = wiki.page(topic)
    return page.summary[:1000] if page.exists() else ""
```
Install: `pip install wikipedia-api --break-system-packages`

### Search queries to run per topic (run all 3):
```python
def get_search_context(topic_info: dict) -> str:
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
        time.sleep(0.5)  # be polite to DDG

    # Also try Wikipedia for the field
    wiki = wiki_summary(field.split("/")[0].strip())
    if wiki:
        snippets.append(wiki)

    return " ".join(snippets)[:3000]  # cap at 3000 chars
```

---

## LLM EXTRACTION

After getting search context, use the **Claude API** (claude-haiku-4-5) to extract vocab terms.

### The prompt to send:
```python
EXTRACTION_PROMPT = """
I have a meeting transcript classified as:
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
```

### API call (read API key from .env):
```python
import os, json, re, requests
from pathlib import Path

def get_api_key() -> str:
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    return os.environ.get("ANTHROPIC_API_KEY", "")

def extract_vocab_with_llm(topic_info: dict, web_context: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        field=topic_info["field"],
        topic=topic_info["topic"],
        description=topic_info["description"],
        web_context=web_context,
        known_terms=", ".join(topic_info.get("known_terms", [])),
    )

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": get_api_key(),
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

    text = response.json()["content"][0]["text"].strip()
    # Strip any accidental markdown
    text = re.sub(r"```json\s*|\s*```", "", text).strip()
    return json.loads(text)
```

### Fallback (no API key):
If no API key, just return the `known_terms` as basic vocab entries with empty `known_errors`. The web search still runs — just no LLM extraction.

---

## MERGING INTO domain_vocab.json

After extracting new terms, merge them into `asr_correction/domain_vocab.json`:

```python
def merge_into_domain_vocab(new_terms: list[dict], vocab_path: Path):
    """Add new terms to domain_vocab.json without overwriting existing entries."""

    vocab = json.loads(vocab_path.read_text())
    existing = {t["term"].lower(): t for t in vocab["terms"]}

    added = 0
    updated = 0

    for term_info in new_terms:
        term = term_info.get("term", "").strip()
        if not term:
            continue
        key = term.lower()

        if key in existing:
            # Only add NEW known_errors to existing entry
            existing_errors = set(existing[key].get("known_errors", []))
            new_errors = [e for e in term_info.get("known_errors", [])
                         if e and e.lower() != key and e not in existing_errors]
            if new_errors:
                existing[key]["known_errors"].extend(new_errors)
                updated += 1
        else:
            # Add completely new term
            existing[key] = {
                "term": term,
                "category": term_info.get("category", "unknown"),
                "known_errors": [e for e in term_info.get("known_errors", []) if e],
            }
            added += 1

    # Write back sorted by term
    vocab["terms"] = sorted(existing.values(), key=lambda x: x["term"].lower())
    vocab_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))

    print(f"  Added {added} new terms, updated {updated} existing terms")
    print(f"  Total terms now: {len(vocab['terms'])}")
```

---

## MEETING TOPIC DEFINITIONS

Define these inside the script as a constant dict `MEETING_TOPICS`:

```python
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
```

---

## FULL SCRIPT STRUCTURE

```
scripts/enrich_vocab_from_web.py
```

Functions to implement:

```python
def web_search(query: str, max_results: int = 8) -> list[dict]
    # ddgs search, returns [{title, href, body}]

def wiki_summary(topic: str) -> str
    # wikipedia-api summary, returns first 1000 chars

def get_search_context(topic_info: dict) -> str
    # runs 3 targeted queries + wikipedia, returns combined text ≤3000 chars

def extract_vocab_with_llm(topic_info: dict, web_context: str) -> list[dict]
    # calls claude-haiku-4-5 with EXTRACTION_PROMPT, parses JSON response

def extract_vocab_fallback(topic_info: dict) -> list[dict]
    # if no API key: returns known_terms as basic entries with empty known_errors

def merge_into_domain_vocab(new_terms: list[dict], vocab_path: Path)
    # merges new terms into domain_vocab.json, never overwrites existing known_errors

def process_topic(video_name: str, topic_info: dict, dry_run: bool = False)
    # full pipeline: search → extract → merge → print summary

def main()
    # argparse: --video, --all, --dry-run, --no-api
```

---

## CLI USAGE

```bash
# Install dependencies
pip install ddgs wikipedia-api --break-system-packages

# Test web search only (no LLM, no write)
python scripts/enrich_vocab_from_web.py --dry-run

# Enrich vocab for one video
python scripts/enrich_vocab_from_web.py --video screenapp_migration_kimi

# Enrich all 10 videos (adds to domain_vocab.json)
python scripts/enrich_vocab_from_web.py --all

# Run without Claude API (just use known_terms as vocab, still searches web for context)
python scripts/enrich_vocab_from_web.py --all --no-api
```

---

## EXPECTED OUTPUT

After running `--all`:

Terminal output per video:
```
==============================
Processing: screenapp_migration_kimi
Field: AI/ML / SaaS Infrastructure
==============================
[1/3] Searching web...
  Got 2847 chars of context
[2/3] Extracting vocab with LLM...
  Extracted 38 terms
[3/3] Merging into domain_vocab.json...
  Added 24 new terms, updated 6 existing terms
  Total terms now: 187
```

New entries added to `asr_correction/domain_vocab.json`:
```json
{
  "term": "Qdrant",
  "category": "tech_term",
  "known_errors": ["quadrant", "Q-drant", "Qdrant"]
},
{
  "term": "pyannote",
  "category": "tech_term",
  "known_errors": ["piano note", "py annotate", "pee annote"]
},
{
  "term": "NDCG",
  "category": "tech_acronym",
  "known_errors": ["end DCG", "NDSG", "N D C G"]
}
```

---

## QUALITY RULES (enforce in the script)

Before adding any term to domain_vocab.json:
- Skip if `term` is empty or less than 2 characters
- Skip if `term` is a pure stop word (the, a, an, in, of, to, is, it)
- Skip if `term.lower()` already exists in domain_vocab.json (only update known_errors)
- Validate `category` is one of the allowed values — default to `"unknown"` if invalid
- Remove any `known_error` that is identical to the term (case-insensitive)
- Remove any `known_error` shorter than 2 characters
- Cap `known_errors` list at 6 entries per term

---

## WHAT NOT TO DO

- Do NOT generate training pairs (that's a separate step)
- Do NOT modify any files except `asr_correction/domain_vocab.json`
- Do NOT require any API key for the script to run (use fallback)
- Do NOT scrape full web pages — just use ddgs text snippets
- Do NOT hardcode any terms — all terms must come from web search + LLM
  (the `known_terms` in MEETING_TOPICS are seeds for the LLM prompt only)
