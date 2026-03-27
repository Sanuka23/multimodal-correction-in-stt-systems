#!/usr/bin/env python3
"""Manual test: Send transcript + vocab to model, see what it detects.

No OCR, no AVSR, no segment selector — just raw model detection.

Usage:
    python scripts/test_detection.py
    python scripts/test_detection.py --file-id 5c925cfa-1dbe-42ca-8e61-d585c4f75f43
    python scripts/test_detection.py --text "we use quadrant for vector search" --vocab "Qdrant,RAG,ScreenApp"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asr_correction.model import load_model, unload_model, run_inference
from asr_correction.batch_corrector import build_batch_prompt, BATCH_SYSTEM_PROMPT


CHUNK_SIZE = 2000


def chunk_text(text, size=CHUNK_SIZE):
    if len(text) <= size:
        return [text]
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            for sep in ['. ', '? ', '! ', '\n']:
                bp = text.rfind(sep, pos + size - 300, end)
                if bp > pos:
                    end = bp + len(sep)
                    break
        chunks.append(text[pos:end])
        pos = end
    return chunks


def load_from_screenapp(file_id: str) -> tuple:
    """Load transcript + vocab from ScreenApp API."""
    import requests, os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    api_url = os.getenv("SCREENAPP_API_URL")
    token = os.getenv("SCREENAPP_PAT_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(f"{api_url}/files/{file_id}", headers=headers)
    if resp.status_code != 200:
        print(f"API error: {resp.status_code}")
        return None, None

    file_data = resp.json().get("data", {}).get("file", {})
    td = file_data.get("textData", {})

    # Get transcript
    transcript_url = td.get("transcriptUrl", "")
    if transcript_url:
        t_resp = requests.get(transcript_url)
        t_data = t_resp.json()
        segments = t_data.get("segments", [])
        text = " ".join(s.get("text", "").strip() for s in segments if s.get("text"))
    else:
        text = td.get("transcriptText", "")

    # Get vocab from file options or team settings
    vocab = []
    # Try loading domain vocab
    from asr_correction.vocabulary import load_domain_vocab
    domain = load_domain_vocab()
    vocab = [t["term"] for t in domain.get("terms", domain) if isinstance(t, dict)]

    return text, vocab


def main():
    parser = argparse.ArgumentParser(description="Test model detection on transcript + vocab")
    parser.add_argument("--file-id", help="ScreenApp file ID to load")
    parser.add_argument("--text", help="Direct transcript text")
    parser.add_argument("--vocab", help="Comma-separated vocab terms")
    parser.add_argument("--adapter", default="asr_correction/adapters",
                        help="Adapter path (default: v1)")
    args = parser.parse_args()

    # Load transcript + vocab
    if args.text:
        text = args.text
        vocab = args.vocab.split(",") if args.vocab else []
    elif args.file_id:
        text, vocab = load_from_screenapp(args.file_id)
        if not text:
            print("Failed to load transcript")
            return
    else:
        # Load last correction request from the log output
        print("No --file-id or --text provided.")
        print("Usage: python scripts/test_detection.py --file-id YOUR_FILE_ID")
        print("   or: python scripts/test_detection.py --text 'transcript here' --vocab 'term1,term2'")
        return

    print(f"=== TRANSCRIPT ({len(text)} chars) ===")
    print(text[:200] + "..." if len(text) > 200 else text)
    print(f"\n=== VOCAB ({len(vocab)} terms) ===")
    print(vocab[:20])

    # Load model
    print(f"\n=== Loading model (adapters: {args.adapter}) ===")
    unload_model()
    model, tokenizer = load_model(
        adapter_path=args.adapter,
        model_path="asr_correction/model_weights",
    )

    # Chunk and send to model
    chunks = chunk_text(text)
    print(f"\n=== DETECTION ({len(chunks)} chunks) ===\n")

    all_changes = []
    for i, chunk in enumerate(chunks):
        prompt = build_batch_prompt(chunk, vocab[:50])
        result = run_inference(prompt, BATCH_SYSTEM_PROMPT, model, tokenizer, max_tokens=256)

        changes = result.get("changes", [])
        confidence = result.get("confidence", 0)
        raw = result.get("corrected", "")

        print(f"--- Chunk {i+1}/{len(chunks)} ({len(chunk)} chars) ---")
        print(f"  Confidence: {confidence}")
        print(f"  Changes: {changes}")
        if raw and changes:
            print(f"  Raw: {str(raw)[:200]}")
        print()

        if changes:
            for c in changes:
                all_changes.append(c)

    print(f"=== SUMMARY ===")
    print(f"Total changes found: {len(all_changes)}")
    for c in all_changes:
        print(f"  {c}")


if __name__ == "__main__":
    main()
