#!/usr/bin/env python3
"""Step 4: Build complete multimodal training pairs from ScreenApp videos.

For each of the 11 private meeting videos, combines:
- ScreenApp ASR hypothesis (from comparison_output/)
- ElevenLabs ground truth (from elevenlabs_output/)
- Domain vocabulary (from tter_vocab.json)
- OCR hints (from data/ocr_cache/)
- AVSR hints (from data/avsr_cache/)

Uses jiwer alignment to find substitution errors and generates ChatML pairs.
"""

import json
import random
import sys
from pathlib import Path

import jiwer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.prepare_data import SYSTEM_PROMPT, build_user_prompt, build_assistant_response

FYP = Path("/Users/sanukathamuditha/Desktop/FYP/Tests")
PROJECT = Path("/Users/sanukathamuditha/Documents/ScreenApp/multimodal-correction-in-stt-systems")

# Video name → (comparison_output key, accent)
VIDEO_MAP = {
    "aws_migration": ("aws_migration", "south_asian"),
    "followup_julien": ("followup_julien", "american"),
    "screenapp_migration_kimi": ("screenapp_migration_kimi", "south_asian"),
    "troubleshooting_dimiter": ("troubleshooting_dimiter", "european"),
    "compliance_discussion": ("compliance_discussion", "american"),
    "gcp_security": ("gcp_security", "american"),
    "onboarding_andre": ("onboarding_andre", "south_asian"),
    "project_update": ("project_update", "south_asian"),
    "business_discussion": ("business_discussion", "american"),
    "zachary_onboarding": ("zachary_onboarding", "american"),
    "test_video": ("test_video", "south_asian"),
}

# Stop words to skip as correction targets
STOP_WORDS = {
    "the", "a", "an", "in", "of", "to", "for", "is", "it", "was",
    "are", "be", "been", "and", "or", "but", "if", "on", "at", "by",
    "from", "with", "as", "that", "this", "have", "has", "had", "do",
    "did", "not", "no", "so", "up", "out", "can", "will", "than",
    "then", "its", "i", "you", "we", "he", "she", "they", "me", "my",
    "your", "our", "just", "like", "um", "uh", "yeah", "okay", "ok",
    "right", "well", "what", "how", "when", "where", "who", "which",
}

CONTEXT_WINDOW = 80
random.seed(42)


def load_vocab(video_key: str) -> list:
    """Load vocabulary terms for a video from tter_vocab.json."""
    vocab_path = FYP / "tter_vocab.json"
    with open(vocab_path) as f:
        data = json.load(f)

    videos = data.get("videos", data)
    if video_key in videos:
        vdata = videos[video_key]
        if isinstance(vdata, dict):
            return vdata.get("terms", [])
        elif isinstance(vdata, list):
            return vdata
    return []


def load_ocr_cache(video_key: str) -> dict:
    """Load OCR cache for a video."""
    cache_path = PROJECT / "data" / "ocr_cache" / f"{video_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def load_avsr_cache(video_key: str) -> dict:
    """Load AVSR cache for a video."""
    cache_path = PROJECT / "data" / "avsr_cache" / f"{video_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def get_ocr_hints_at_timestamp(ocr_cache: dict, timestamp: float, window: float = 30.0) -> list:
    """Get OCR text lines near a timestamp."""
    hints = []
    seen = set()
    for ts_str, lines in ocr_cache.items():
        ts = float(ts_str)
        if abs(ts - timestamp) <= window:
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    hints.append(line)
    return hints[:15]


def get_avsr_hint_at_timestamp(avsr_cache: dict, timestamp: float, window: float = 10.0) -> str:
    """Get AVSR hint string near a timestamp."""
    best_dist = float("inf")
    best_hint = "null"
    for ts_str, hint in avsr_cache.items():
        ts = float(ts_str)
        dist = abs(ts - timestamp)
        if dist <= window and dist < best_dist:
            best_dist = dist
            best_hint = hint
    return best_hint


def find_timestamp_for_word(segments: list, word_idx: int, asr_words: list) -> float:
    """Find the approximate timestamp for a word at position word_idx."""
    # Walk through segments to find which one contains this word
    running_idx = 0
    for seg in segments:
        seg_words = seg.get("words", [])
        if not seg_words:
            # Estimate from segment text
            seg_word_count = len(seg.get("text", "").split())
            if running_idx + seg_word_count > word_idx:
                # This segment contains our word
                ratio = (word_idx - running_idx) / max(seg_word_count, 1)
                return seg.get("start", 0) + ratio * (seg.get("end", 0) - seg.get("start", 0))
            running_idx += seg_word_count
        else:
            if running_idx + len(seg_words) > word_idx:
                local_idx = word_idx - running_idx
                if local_idx < len(seg_words):
                    return seg_words[local_idx].get("start", seg.get("start", 0))
                return seg.get("start", 0)
            running_idx += len(seg_words)

    # Fallback to last segment
    if segments:
        return segments[-1].get("end", 0)
    return 0.0


def extract_context(text: str, word_position: int, window: int = CONTEXT_WINDOW) -> str:
    """Extract context window around a character position."""
    start = max(0, word_position - window)
    end = min(len(text), word_position + window)
    return text[start:end]


def build_pair(
    asr_context: str,
    gt_context: str,
    error_word: str,
    correct_word: str,
    vocab_terms: list,
    category: str,
    ocr_hints: list,
    avsr_hint: str,
    source: str,
    accent: str,
    timestamp: float,
    is_negative: bool = False,
) -> dict:
    """Build a single ChatML training pair."""
    # Build vocab list for prompt
    vocab_names = [t["term"] for t in vocab_terms[:20]]
    if correct_word not in vocab_names and not is_negative:
        vocab_names.insert(0, correct_word)

    # Build user prompt
    user_content = build_user_prompt(
        asr_context, vocab_names, category,
        ocr_hints=ocr_hints if ocr_hints else None,
        lip_hint=avsr_hint if avsr_hint != "null" else None,
    )

    # Build assistant response
    if is_negative:
        assistant_content = build_assistant_response(
            asr_context, [], 0.99, False
        )
    else:
        changes = [f"{error_word} → {correct_word}"]
        assistant_content = build_assistant_response(
            gt_context, changes, 0.95, False
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": source,
            "accent": accent,
            "term": correct_word if not is_negative else error_word,
            "category": category,
            "error_found": error_word,
            "timestamp": round(timestamp, 2),
            "has_ocr": bool(ocr_hints),
            "has_avsr": avsr_hint != "null",
            "applied": not is_negative,
            "is_negative": is_negative,
        },
    }


def process_video(video_key: str, accent: str) -> list:
    """Process one video into training pairs."""
    comp_dir = FYP / "comparison_output" / video_key
    gt_dir = FYP / "elevenlabs_output" / video_key

    # Load ASR transcript
    asr_json_path = comp_dir / "screenapp_transcript.json"
    if not asr_json_path.exists():
        print(f"  SKIP — no ASR transcript")
        return []

    with open(asr_json_path) as f:
        asr_data = json.load(f)
    asr_text = asr_data.get("text", "")
    segments = asr_data.get("segments", [])

    # Load ground truth
    gt_path = gt_dir / "reference_transcript.txt"
    if not gt_path.exists():
        print(f"  SKIP — no ground truth")
        return []

    gt_text = gt_path.read_text().strip()

    if not asr_text or not gt_text:
        print(f"  SKIP — empty text")
        return []

    # Load vocab, OCR, AVSR
    vocab_terms = load_vocab(video_key)
    ocr_cache = load_ocr_cache(video_key)
    avsr_cache = load_avsr_cache(video_key)

    print(f"  ASR: {len(asr_text)} chars | GT: {len(gt_text)} chars | "
          f"Vocab: {len(vocab_terms)} | OCR: {len(ocr_cache)} timestamps | "
          f"AVSR: {len(avsr_cache)} timestamps")

    # jiwer alignment
    asr_words = asr_text.split()
    gt_words = gt_text.split()

    try:
        alignment = jiwer.process_words(gt_text, asr_text)
    except Exception as e:
        print(f"  SKIP — jiwer alignment failed: {e}")
        return []

    pairs = []
    neg_candidates = []

    # Process alignment chunks
    for chunk in alignment.alignments[0]:
        if chunk.type == "substitute":
            # Substitution error — positive training example
            # chunk covers ref[ref_start_idx:ref_end_idx] and hyp[hyp_start_idx:hyp_end_idx]
            n_ref = chunk.ref_end_idx - chunk.ref_start_idx
            n_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx

            for i in range(min(n_ref, n_hyp)):
                ref_i = chunk.ref_start_idx + i
                hyp_i = chunk.hyp_start_idx + i

                if ref_i < len(gt_words) and hyp_i < len(asr_words):
                    error_word = asr_words[hyp_i]
                    correct_word = gt_words[ref_i]

                    # Quality filters
                    if error_word.lower() in STOP_WORDS:
                        continue
                    if correct_word.lower() in STOP_WORDS:
                        continue
                    if len(error_word) < 3 or len(correct_word) < 3:
                        continue
                    if error_word.lower() == correct_word.lower():
                        continue

                    # Find timestamp
                    timestamp = find_timestamp_for_word(segments, hyp_i, asr_words)

                    # Find character position for context
                    char_pos = sum(len(w) + 1 for w in asr_words[:hyp_i])
                    asr_context = extract_context(asr_text, char_pos)
                    gt_char_pos = sum(len(w) + 1 for w in gt_words[:ref_i])
                    gt_context = extract_context(gt_text, gt_char_pos)

                    # Get OCR and AVSR hints at this timestamp
                    ocr_hints = get_ocr_hints_at_timestamp(ocr_cache, timestamp)
                    avsr_hint = get_avsr_hint_at_timestamp(avsr_cache, timestamp)

                    # Determine category from vocab
                    category = "content_word"
                    for vt in vocab_terms:
                        if vt["term"].lower() == correct_word.lower():
                            category = vt.get("category", "content_word")
                            break
                        if error_word.lower() in [e.lower() for e in vt.get("known_errors", [])]:
                            category = vt.get("category", "content_word")
                            break

                    pair = build_pair(
                        asr_context, gt_context, error_word, correct_word,
                        vocab_terms, category, ocr_hints, avsr_hint,
                        f"screenapp_{video_key}", accent, timestamp,
                    )
                    pairs.append(pair)

        elif chunk.type == "equal":
            # Equal — candidate for negative example
            for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                hyp_i = chunk.hyp_start_idx + i
                if hyp_i < len(asr_words):
                    word = asr_words[hyp_i]
                    if len(word) >= 4 and word.lower() not in STOP_WORDS:
                        neg_candidates.append(hyp_i)

    # Sample ~20% negative examples
    n_negatives = max(1, len(pairs) // 4)
    neg_sample = random.sample(neg_candidates, min(n_negatives, len(neg_candidates)))

    for hyp_idx in neg_sample:
        word = asr_words[hyp_idx]
        timestamp = find_timestamp_for_word(segments, hyp_idx, asr_words)
        char_pos = sum(len(w) + 1 for w in asr_words[:hyp_idx])
        context = extract_context(asr_text, char_pos)
        ocr_hints = get_ocr_hints_at_timestamp(ocr_cache, timestamp)
        avsr_hint = get_avsr_hint_at_timestamp(avsr_cache, timestamp)

        # Pick a random vocab term to test against
        if vocab_terms:
            test_term = random.choice(vocab_terms)
            category = test_term.get("category", "content_word")
        else:
            category = "content_word"

        pair = build_pair(
            context, context, word, word,
            vocab_terms, category, ocr_hints, avsr_hint,
            f"screenapp_{video_key}", accent, timestamp,
            is_negative=True,
        )
        pairs.append(pair)

    return pairs


def main():
    output_path = PROJECT / "data" / "training_pairs" / "screenapp_final.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    print("=== Step 4: Building ScreenApp Training Pairs ===\n")

    for video_key, (comp_key, accent) in VIDEO_MAP.items():
        print(f"[{video_key}] accent={accent}")
        pairs = process_video(comp_key, accent)
        print(f"  Generated: {len(pairs)} pairs "
              f"({sum(1 for p in pairs if p['metadata']['applied'])} positive, "
              f"{sum(1 for p in pairs if p['metadata']['is_negative'])} negative)")
        all_pairs.extend(pairs)
        print()

    # Save
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"=== Complete ===")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"  Positive: {sum(1 for p in all_pairs if p['metadata']['applied'])}")
    print(f"  Negative: {sum(1 for p in all_pairs if p['metadata']['is_negative'])}")
    print(f"  With OCR: {sum(1 for p in all_pairs if p['metadata']['has_ocr'])}")
    print(f"  With AVSR: {sum(1 for p in all_pairs if p['metadata']['has_avsr'])}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
