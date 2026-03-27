"""Generate hard negative training examples for ASR correction.

Hard negatives are examples where:
- A correction was PROPOSED but is WRONG
- The model should learn to output NO CHANGE for these

Three types of hard negatives:
1. Real term → different real term (TypeSense ≠ Qdrant, Sheldon ≠ Tim)
2. Correct word → vocab-biased replacement (screen ≠ transcript)
3. Mined from existing training data — flip positive examples where original was correct
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an ASR transcript correction model. Given a noisy ASR transcript "
    "segment and context signals, detect errors in context-critical terms and "
    "output the corrected transcript with changes noted."
)

# ─── Type 1: Real observed bad corrections from production runs ───
# These were actually proposed by the model and rejected or identified as wrong
OBSERVED_BAD_CORRECTIONS = [
    # (transcript_context, wrong_replacement, correct_word, vocab_that_caused_it, category)
    # Meeting 1: LLM comparison meeting
    ("Sheldon and Andre are working on the integration", "Tim", "Sheldon", "Sheldon", "person_name"),
    ("Sheldon presented the results to the team", "Sean", "Sheldon", "Sheldon", "person_name"),
    ("Andre will take care of the deployment", "Andrei", "Andre", "Andre", "person_name"),
    ("Dinuka will handle the backend changes", "Kieran", "Dinuka", "Dinuka", "person_name"),
    ("Dinuka, how about this approach?", "Dilushan", "Dinuka", "Dinuka", "person_name"),
    ("Junuka is working on the frontend", "Kieran", "Junuka", "Junuka", "person_name"),
    ("let me quickly share my screen and show you", "transcript", "screen", "ScreenApp", "content_word"),
    ("we're using TypeSense for our search", "Qdrant", "TypeSense", "Qdrant", "tech_term"),
    ("the GPT-2 model benchmark results", "Gemini 2.5 Pro", "GPT-2", "Gemini", "ai_model"),
    ("GPT-OSS performed well in our tests", "DeepSeek", "GPT-OSS", "DeepSeek", "ai_model"),
    ("Maverick is the new model from Meta", "Kimi", "Maverick", "Kimi", "ai_model"),
    ("NavRick showed strong performance", "Kimi", "NavRick", "Kimi", "ai_model"),
    ("Kimi is already performing well here", "Mimi", "Kimi", "Kimi", "ai_model"),
    ("Why Kimmy results are better than expected", "Mimi", "Kimmy", "Kimi", "person_name"),
    ("the hash function needs optimization", "HALLUCINATIONS", "hash", "hallucination", "tech_term"),
    ("we need to manage cash flow better", "cache flow", "cash flow", "cache", "business_term"),
    ("walk through the process step by step", "churning", "walk", "churn", "content_word"),
    ("Meantime, let's move to the next topic", "Mimi", "Meantime", "Kimi", "content_word"),
    ("Ask AI feature is working great", "AskK", "Ask AI", "Ask AI", "feature"),
    ("the cool thing about this approach", "cut", "cool", "cut", "content_word"),
    ("looking at the Grok API performance", "Groq", "Grok", "Groq", "ai_model"),
    ("28 million tokens processed per day", "28M", "28 million", "million", "content_word"),
    ("pull out them from the database", "pull them out", "pull out them", "pull", "content_word"),
    ("Flashlight is the smallest Gemini model", "Flash", "Flashlight", "Flash", "ai_model"),
    # Meeting 2: Qdrant/Vector search meeting
    ("TypeSense handles full text search well", "Qdrant", "TypeSense", "Qdrant", "tech_term"),
    ("TypeScript Cloud functions are deployed", "Vertex AI", "TypeScript Cloud", "Vertex AI", "tech_term"),
    ("the queue is backing up with requests", "queries", "queue", "query", "tech_term"),
    ("running multiple Experiments in parallel", "Experiment", "Experiments", "Experiment", "content_word"),
    ("mind-on approach to vector search", "multi-vector", "mind-on", "multi-vector", "tech_term"),
]

# ─── Type 2: Common ASR words that should NOT be replaced ───
# These are real English words that sound like vocab terms
COMMON_WORDS_NOT_TO_REPLACE = [
    ("we need to cache the results properly", "cash", "cache", ["cache", "Groq"], "tech_term"),
    ("the screen recording is saving", "screen", "screen", ["ScreenApp", "transcript"], "content_word"),
    ("login to your account first", "login", "login", ["log in", "Groq"], "content_word"),
    ("we can evaluate the model performance", "evaluate", "evaluate", ["eval", "evaluation"], "content_word"),
    ("the parallel processing is working", "parallel", "parallel", ["parallelly", "pipeline"], "tech_term"),
    ("check the index for that document", "index", "index", ["indexed", "Elasticsearch"], "tech_term"),
    ("this is a compliance requirement", "compliance", "compliance", ["SOC 2", "GDPR"], "compliance"),
    ("the piano music was playing in the background", "piano", "piano", ["Plan", "Pipeline"], "content_word"),
    ("she walked into the meeting room", "walked", "walked", ["churning", "walking"], "content_word"),
    ("the clock shows it's almost noon", "clock", "clock", ["compute", "cluster"], "content_word"),
    ("I have a fear that this might break", "fear", "fear", ["fix", "feature"], "content_word"),
    ("this is a cool feature we built", "cool", "cool", ["cut", "call"], "content_word"),
]


def make_hard_negative(transcript: str, wrong_word: str, correct_word: str,
                       vocab: list, category: str) -> dict:
    """Create a training example that teaches: don't make this replacement."""
    if isinstance(vocab, str):
        vocab = [vocab]

    user_prompt = (
        "Correct this ASR transcript segment using the provided context.\n"
        "IMPORTANT: Only change words that are clearly wrong. If a word is "
        "already correct, do NOT change it. Use OCR screen text to verify "
        "the correct spelling of domain terms.\n\n"
        f"ASR transcript: {transcript}\n"
        f"Custom vocabulary: {json.dumps(vocab)}\n"
        f"Category: {category}\n"
        "OCR hints: none available\n"
        "Lip reading hint: null"
    )

    # The correct output: NO CHANGE — the word is already correct
    assistant_response = json.dumps({
        "corrected": transcript,
        "changes": [],
        "confidence": 0.99,
        "need_lip": False,
    })

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": {
            "source": "hard_negative_generated",
            "is_negative": True,
            "is_hard_negative": True,
            "applied": False,
            "term": correct_word,
            "category": category,
            "error_found": correct_word,
            "wrong_replacement": wrong_word,
            "reason": f"Model incorrectly proposed '{correct_word}' → '{wrong_word}'. Original word is correct.",
        }
    }


def generate_all():
    ROOT = Path(__file__).resolve().parent.parent
    output_dir = ROOT / "data" / "hard_negatives"
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []

    # Type 1: Observed bad corrections
    print("Generating Type 1: Observed bad corrections...")
    for transcript, wrong, correct, vocab, category in OBSERVED_BAD_CORRECTIONS:
        ex = make_hard_negative(transcript, wrong, correct, vocab, category)
        examples.append(ex)
    print(f"  Generated {len(OBSERVED_BAD_CORRECTIONS)} examples")

    # Type 2: Common words not to replace
    print("Generating Type 2: Common words not to replace...")
    for transcript, word, correct, vocab, category in COMMON_WORDS_NOT_TO_REPLACE:
        ex = make_hard_negative(transcript, word, correct, vocab, category)
        examples.append(ex)
    print(f"  Generated {len(COMMON_WORDS_NOT_TO_REPLACE)} examples")

    # Type 3: Mine from existing training data — find cases where the correction
    # was dubious (error_found is a real common word)
    print("Generating Type 3: Mining existing data for dubious corrections...")
    train_path = ROOT / "data" / "collected_data" / "train.jsonl"
    if train_path.exists():
        with open(train_path) as f:
            all_train = [json.loads(l) for l in f if l.strip()]

        # Common English words that shouldn't be replaced
        common_words = {
            "going", "the", "and", "but", "have", "will", "can", "did",
            "around", "about", "want", "need", "time", "know", "think",
            "look", "come", "make", "take", "give", "call", "put",
            "screen", "cash", "walk", "cool", "fear", "piano", "clock",
            "queue", "login", "parallel", "index", "sort", "types",
        }

        mined_count = 0
        for ex in all_train:
            m = ex.get("metadata", {})
            if not m.get("applied"):
                continue
            error = m.get("error_found", "").lower().strip(".,!?'\" ")
            term = m.get("term", "")
            # If the "error" is actually a common English word, this might be a bad correction
            if error in common_words and error != term.lower():
                # Create a hard negative: the original word was fine
                user_content = ex["messages"][1]["content"]
                # Extract the ASR transcript from the prompt
                asr_start = user_content.find("ASR transcript: ")
                if asr_start >= 0:
                    asr_text = user_content[asr_start + len("ASR transcript: "):]
                    asr_text = asr_text.split("\n")[0].strip()
                    neg = make_hard_negative(
                        asr_text, term, error,
                        [term], m.get("category", "content_word")
                    )
                    examples.append(neg)
                    mined_count += 1
                    if mined_count >= 200:  # Cap to avoid overwhelming
                        break
        print(f"  Mined {mined_count} examples from existing training data")

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.85)
    train_examples = examples[:split]
    valid_examples = examples[split:]

    # Save
    with open(output_dir / "hard_negatives.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal hard negatives: {len(examples)}")
    print(f"  Train: {len(train_examples)}")
    print(f"  Valid: {len(valid_examples)}")
    print(f"  Saved to: {output_dir}/")

    # Show breakdown
    types = {}
    for ex in examples:
        cat = ex["metadata"]["category"]
        types[cat] = types.get(cat, 0) + 1
    print(f"\nBy category: {json.dumps(types, indent=2)}")

    return examples


if __name__ == "__main__":
    generate_all()
