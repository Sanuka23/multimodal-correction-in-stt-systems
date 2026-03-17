"""Generate correction pairs from LibriSpeech dataset.

Reads LibriSpeech directory structure:
    LibriSpeech/train-clean-100/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.flac
    LibriSpeech/train-clean-100/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}.trans.txt

Usage:
    python training/generate_pairs_librispeech.py \
        --librispeech-dir data/datasets/LibriSpeech/train-clean-100 \
        --transcripts data/transcripts/librispeech.jsonl \
        --output data/training_pairs/librispeech_pairs.jsonl
"""

import argparse, json, logging, sys
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))
from generate_pairs import generate_pairs_from_alignment, save_pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_librispeech_transcripts(librispeech_dir: str) -> dict:
    """Load ground truth transcripts from LibriSpeech directory.

    Returns: dict mapping filename (without extension) -> transcript text
    """
    transcripts = {}
    base = Path(librispeech_dir)

    for trans_file in base.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    transcripts[utterance_id] = text

    logger.info("Loaded %d LibriSpeech ground truth transcripts", len(transcripts))
    return transcripts


def load_screenapp_transcripts(jsonl_path: str) -> dict:
    """Load ScreenApp ASR transcripts from JSONL output.

    Returns: dict mapping file_name -> transcript_text
    """
    transcripts = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            name = Path(entry.get("file_name", "")).stem
            transcripts[name] = entry.get("transcript_text", "")

    logger.info("Loaded %d ScreenApp transcripts", len(transcripts))
    return transcripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-dir", required=True)
    parser.add_argument("--transcripts", required=True, help="ScreenApp transcripts JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ground_truth = load_librispeech_transcripts(args.librispeech_dir)
    screenapp = load_screenapp_transcripts(args.transcripts)

    all_pairs = []
    matched = 0

    for utterance_id, gt_text in ground_truth.items():
        if utterance_id in screenapp:
            asr_text = screenapp[utterance_id]
            pairs = generate_pairs_from_alignment(
                asr_text=asr_text,
                ground_truth=gt_text,
                source="librispeech",
                file_id=utterance_id,
            )
            all_pairs.extend(pairs)
            matched += 1

    logger.info("Matched %d/%d utterances, generated %d pairs",
                matched, len(ground_truth), len(all_pairs))
    save_pairs(all_pairs, args.output)


if __name__ == "__main__":
    main()
