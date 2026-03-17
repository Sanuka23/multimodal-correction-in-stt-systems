"""Generate correction pairs from AMI Meeting Corpus.

Parses AMI word-level annotations (XML format) and compares with
ScreenApp ASR transcripts of the meeting audio.

Usage:
    python training/generate_pairs_ami.py \
        --ami-dir data/datasets/ami \
        --transcripts data/transcripts/ami.jsonl \
        --output data/training_pairs/ami_pairs.jsonl
"""

import argparse, json, logging, sys, xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_pairs import generate_pairs_from_alignment, save_pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_ami_transcripts(ami_dir: str) -> dict:
    """Load AMI meeting transcripts from word-level XML annotations.

    Returns: dict mapping meeting_id -> full transcript text
    """
    transcripts = {}
    words_dir = Path(ami_dir) / "words"

    if not words_dir.exists():
        # Try alternative path
        words_dir = Path(ami_dir) / "ami_public_manual_1.6.2" / "words"

    if not words_dir.exists():
        logger.error("AMI words directory not found at %s", words_dir)
        return transcripts

    for xml_file in sorted(words_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            words = []
            for w in root.iter("w"):
                text = w.text
                if text and text.strip():
                    words.append(text.strip())

            meeting_id = xml_file.stem.rsplit(".", 1)[0]  # e.g., "ES2002a.A"
            base_meeting = meeting_id.split(".")[0]  # e.g., "ES2002a"

            if base_meeting not in transcripts:
                transcripts[base_meeting] = []
            transcripts[base_meeting].extend(words)
        except ET.ParseError:
            logger.warning("Failed to parse %s", xml_file)

    # Join words into full transcripts
    result = {k: " ".join(v) for k, v in transcripts.items()}
    logger.info("Loaded %d AMI meeting transcripts", len(result))
    return result


def load_screenapp_transcripts(jsonl_path: str) -> dict:
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
    parser.add_argument("--ami-dir", required=True)
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ground_truth = load_ami_transcripts(args.ami_dir)
    screenapp = load_screenapp_transcripts(args.transcripts)

    all_pairs = []
    matched = 0

    for meeting_id, gt_text in ground_truth.items():
        if meeting_id in screenapp:
            asr_text = screenapp[meeting_id]
            pairs = generate_pairs_from_alignment(
                asr_text=asr_text,
                ground_truth=gt_text,
                source="ami",
                file_id=meeting_id,
            )
            all_pairs.extend(pairs)
            matched += 1

    logger.info("Matched %d/%d meetings, generated %d pairs",
                matched, len(ground_truth), len(all_pairs))
    save_pairs(all_pairs, args.output)


if __name__ == "__main__":
    main()
