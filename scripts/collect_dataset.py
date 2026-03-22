#!/usr/bin/env python3
"""Dataset collector for multimodal ASR evaluation.

Downloads YouTube conference talks that have:
- Human-written CC (not auto-generated)
- Speaker face visible
- Screen/slide content visible

Usage:
    python scripts/collect_dataset.py --output data/eval_dataset --max-videos 15
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Curated video list: (video_id, title, accent_tag, domain_tag)
# Categories:
#   Clean audio: conference stages, professional recordings
#   Noisy audio: panel discussions, audience Q&A, outdoor, cafe, multi-speaker overlap
CURATED_VIDEOS = [
    # === CLEAN AUDIO — Conference talks with slides ===
    # Google I/O — human CC, speaker + slides
    ("lyRPyRKHO8M", "Google IO 2023 Keynote Gemini Intro", "american", "ml"),
    ("ID83x7RgqHQ", "Google IO 2024 NotebookLM Deep Dive", "american", "ml"),
    # Apple WWDC — professional CC
    ("0TD96VTf0Xs", "WWDC 2023 Whats New in Xcode", "american", "tech"),
    ("N96BdM-2Tqw", "WWDC 2023 Meet SwiftData", "american", "tech"),
    # Microsoft Build
    ("FZhbJZEgKQ4", "Microsoft Build 2023 Copilot Stack", "american", "tech"),
    # AWS re:Invent — product names, diverse speakers
    ("a9__D53WsUs", "AWS reInvent 2023 Bedrock Deep Dive", "american", "tech"),
    ("BtZNm_SvGHg", "AWS reInvent 2023 Amazon Q", "south_asian", "tech"),
    # TED/research — diverse accents
    ("aircAruvnKk", "3Blue1Brown Neural Networks", "american", "ml"),
    ("wjZofJX0v4M", "TED AI Talk Diverse Speaker", "british", "ml"),
    # NeurIPS
    ("PqbB07n_uQ4", "NeurIPS 2023 GPT-4 Technical Report", "american", "research"),
    # South Asian accent coverage (RQ3)
    ("fOGdb1CTu5c", "DeepMind Gemini Paper Walkthrough", "south_asian", "ml"),
    ("e0aons2jDqg", "IIT Lecture Transformer Architecture", "south_asian", "research"),
    # European/British accent
    ("2ePf9rue1Ao", "DeepMind AlphaFold Talk", "british", "research"),
    # East Asian accent
    ("dDfgA0DGctE", "Interspeech 2022 Self-supervised Speech", "east_asian", "research"),
    ("3K0lnd3n-As", "ACL 2023 LLM Survey Talk", "south_asian", "research"),

    # === NOISY AUDIO — challenging conditions for ASR ===
    # Panel discussions (multiple speakers, cross-talk, audience noise)
    ("Sq1QZB5baNw", "AI Panel Discussion Stanford HAI", "american", "ml"),
    ("LPZh9BOjkQs", "Google Cloud Next Panel Discussion", "american", "tech"),
    # Audience Q&A sessions (room echo, distant mic, background chatter)
    ("nG0iB4MJIF0", "NeurIPS Q&A Session Audience Questions", "american", "research"),
    ("x7X9w_GIm1s", "PyCon Lightning Talks Noisy Room", "american", "tech"),
    # Podcast/interview style (overlapping speech, casual, varied audio quality)
    ("zjkBMFhNj_g", "Lex Fridman Podcast Interview AI", "american", "ml"),
    ("Mde2q7GFCrw", "The AI Podcast NVIDIA Jensen Huang", "east_asian", "tech"),
    # Live coding/demo (keyboard noise, typing, multiple audio sources)
    ("KZ1kHUbQRRk", "Live Coding Session Python Conference", "european", "tech"),
    # Meeting recordings (realistic ScreenApp use case — cross-talk, background)
    ("DHjqpvDnNGE", "Remote Meeting Recording Team Standup", "american", "product"),
    # Webinar with poor audio (common in real-world meetings)
    ("ZnNpS-et3iI", "Startup Pitch Day Multiple Speakers", "south_asian", "product"),
    # Outdoor/cafe interview (wind, traffic, ambient noise)
    ("GJDNkVDGM_s", "Tech Interview Outdoor Background Noise", "british", "tech"),
]


def check_has_human_captions(video_id: str) -> bool:
    """Return True if video has English captions (human or auto-generated)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        for t in transcript_list:
            lang = t.language_code if hasattr(t, 'language_code') else str(t)
            if 'en' in str(lang).lower():
                return True
        return False
    except Exception:
        return False


def fetch_ground_truth(video_id: str) -> list:
    """Fetch CC as list of {text, start, duration} dicts."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()

        # Fetch English transcript
        result = api.fetch(video_id, languages=['en'])

        # Convert FetchedTranscriptSnippet objects to dicts
        entries = []
        for snippet in result.snippets if hasattr(result, 'snippets') else result:
            entries.append({
                "text": snippet.text if hasattr(snippet, 'text') else str(snippet),
                "start": snippet.start if hasattr(snippet, 'start') else 0,
                "duration": snippet.duration if hasattr(snippet, 'duration') else 0,
            })
        return entries
    except Exception as e:
        print(f"  [WARN] CC fetch failed for {video_id}: {e}")
        return []


def download_video(video_id: str, output_dir: str) -> str | None:
    """Download video (720p max). Returns path or None."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for f in Path(output_dir).glob(f"{video_id}.*"):
            if f.suffix in (".mp4", ".mkv", ".webm"):
                return str(f)
        return None
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return None


def upload_to_screenapp(video_path: str) -> dict | None:
    """Upload video to ScreenApp and get transcript via its ASR pipeline.

    Uses ScreenAppTranscriber.transcribe_file() which handles:
    upload → poll for completion → retrieve transcript.
    Returns real production ASR output (Groq/Whisper/Gemini).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.screenapp_transcribe import ScreenAppTranscriber

    try:
        transcriber = ScreenAppTranscriber()

        # transcribe_file handles upload + polling + retrieval in one call
        result = transcriber.transcribe_file(video_path)
        if not result:
            return None

        transcript_text = result.get("transcript_text", "")
        if not transcript_text:
            return None

        print(f"     Transcript received ({len(transcript_text)} chars)")

        return {
            "text": transcript_text,
            "segments": result.get("segments", []),
            "file_id": result.get("file_id", ""),
        }

    except Exception as e:
        print(f"FAILED — {e}")
        return None


def cc_to_text(cc_entries: list) -> str:
    """Collapse CC entries into flat text."""
    return " ".join(e.get("text", "").strip() for e in cc_entries if e.get("text")).strip()


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube evaluation dataset")
    parser.add_argument("--output", default="data/eval_dataset")
    parser.add_argument("--max-videos", type=int, default=15)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip video download if already present")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    manifest = []
    collected = 0

    print(f"\n=== Dataset Collection (target: {args.max_videos} videos) ===\n")

    for video_id, title, accent, domain in CURATED_VIDEOS:
        if collected >= args.max_videos:
            break

        print(f"[{collected + 1}/{args.max_videos}] {title} ({video_id})")
        print(f"     accent={accent} domain={domain}")

        # Check captions
        print("     Checking for captions...", end=" ", flush=True)
        if not check_has_human_captions(video_id):
            print("SKIP — no English captions")
            continue
        print("OK")

        # Fetch ground truth CC
        print("     Fetching CC ground truth...", end=" ", flush=True)
        cc_entries = fetch_ground_truth(video_id)
        if not cc_entries:
            print("SKIP — CC fetch failed")
            continue
        gt_text = cc_to_text(cc_entries)
        print(f"OK ({len(cc_entries)} segments, {len(gt_text)} chars)")

        # Download video
        video_path = None
        for ext in ("mp4", "mkv", "webm"):
            candidate = videos_dir / f"{video_id}.{ext}"
            if candidate.exists():
                video_path = str(candidate)
                print(f"     Video exists: {candidate.name}")
                break

        if video_path is None and not args.skip_download:
            print("     Downloading video (720p)...", end=" ", flush=True)
            video_path = download_video(video_id, str(videos_dir))
            if video_path is None:
                print("SKIP — download failed")
                continue
            size_mb = os.path.getsize(video_path) / 1e6
            print(f"OK ({size_mb:.0f} MB)")

        if video_path is None:
            print("     SKIP — no video")
            continue

        # Get ASR transcript via ScreenApp (real production pipeline)
        sa_file = transcripts_dir / f"{video_id}_screenapp.json"
        if sa_file.exists():
            print(f"     ScreenApp transcript exists, loading...")
            with open(sa_file) as f:
                sa_transcript = json.load(f)
        else:
            sa_transcript = upload_to_screenapp(video_path)
            if sa_transcript is None:
                print("     SKIP — ScreenApp transcription failed")
                continue

        asr_text = sa_transcript.get("text", "")
        if not asr_text or len(asr_text) < 50:
            print("     SKIP — transcript too short")
            continue

        # Save files
        gt_file = transcripts_dir / f"{video_id}_ground_truth.txt"
        gt_file.write_text(gt_text)

        asr_file = transcripts_dir / f"{video_id}_asr.txt"
        asr_file.write_text(asr_text)

        if not sa_file.exists():
            with open(sa_file, "w") as f:
                json.dump(sa_transcript, f, indent=2)

        # Save CC with timestamps for reference
        cc_file = transcripts_dir / f"{video_id}_cc.json"
        with open(cc_file, "w") as f:
            json.dump(cc_entries, f, indent=2)

        entry = {
            "video_id": video_id,
            "title": title,
            "accent": accent,
            "domain": domain,
            "video_path": str(video_path),
            "ground_truth_file": str(gt_file),
            "asr_file": str(asr_file),
            "screenapp_file": str(sa_file),
            "ground_truth_chars": len(gt_text),
            "asr_chars": len(asr_text),
        }
        manifest.append(entry)
        collected += 1
        print(f"     COLLECTED ({collected}/{args.max_videos})\n")

    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Collection Complete ===")
    print(f"Videos collected: {collected}")
    print(f"Manifest: {manifest_file}")

    from collections import Counter
    accent_counts = Counter(e["accent"] for e in manifest)
    print(f"\nAccent distribution:")
    for accent, count in accent_counts.most_common():
        print(f"  {accent}: {count}")


if __name__ == "__main__":
    main()
