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


def collect_transcripts_from_screenapp(output_dir: Path):
    """Pass 2: Collect transcripts from ScreenApp for already-uploaded files.

    Queries ScreenApp folder for all files, matches by video_id in filename,
    and downloads transcripts for any that are ready.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.screenapp_transcribe import ScreenAppTranscriber

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    transcriber = ScreenAppTranscriber()

    # Get all files from ScreenApp folder
    print("\n=== Collecting Transcripts from ScreenApp ===\n")
    print("Fetching file list from ScreenApp folder...")

    import requests
    headers = transcriber._get_headers() if hasattr(transcriber, '_get_headers') else {
        "Authorization": f"Bearer {transcriber.pat_token}"
    }

    all_files = []
    offset = 0
    limit = 100
    while True:
        resp = requests.get(
            f"{transcriber.api_url}/files",
            params={"parentId": transcriber.folder_id, "limit": limit, "offset": offset},
            headers=headers,
        )
        if resp.status_code != 200:
            print(f"  ERROR: API returned {resp.status_code}")
            break
        data = resp.json().get("data", {})
        files = data.get("fileSystem", data.get("files", []))
        if not files:
            break
        all_files.extend(files)
        offset += len(files)
        if len(files) < limit:
            break

    print(f"Found {len(all_files)} files in ScreenApp folder")

    # Match files to curated video IDs
    collected = 0
    for f in all_files:
        file_id = f.get("_id", "")
        file_name = f.get("name", "")

        # Check if this file has a transcript
        transcript_data = f.get("textData", {})
        transcript_text = transcript_data.get("transcriptText", "")

        if not transcript_text or len(transcript_text) < 50:
            continue

        # Try to match to a video_id from our curated list
        # ScreenApp renames files, so check if any video_id appears in providerKey
        provider_key = f.get("providerKey", "")
        matched_video_id = None
        for vid_id, title, accent, domain in CURATED_VIDEOS:
            # Check if file was uploaded with this video's filename
            if vid_id in file_name or vid_id in provider_key:
                matched_video_id = vid_id
                break

        if not matched_video_id:
            # Try matching by file_id from upload tracking
            upload_tracking = output_dir / "upload_tracking.json"
            if upload_tracking.exists():
                with open(upload_tracking) as tf:
                    tracking = json.load(tf)
                matched_video_id = tracking.get(file_id)

        if not matched_video_id:
            continue

        # Save transcript
        sa_file = transcripts_dir / f"{matched_video_id}_screenapp.json"
        if sa_file.exists():
            print(f"  {matched_video_id}: already collected, skipping")
            continue

        segments = transcript_data.get("segments", [])
        sa_transcript = {
            "text": transcript_text,
            "segments": [
                {
                    "id": i,
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    "text": s.get("text", "").strip(),
                    "words": s.get("words", []),
                }
                for i, s in enumerate(segments)
            ],
            "file_id": file_id,
        }

        with open(sa_file, "w") as sf:
            json.dump(sa_transcript, sf, indent=2)

        # Also save ASR text
        asr_file = transcripts_dir / f"{matched_video_id}_asr.txt"
        asr_file.write_text(transcript_text)

        collected += 1
        print(f"  {matched_video_id}: collected ({len(transcript_text)} chars, {len(segments)} segments)")

    print(f"\nCollected {collected} new transcripts")
    return collected


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube evaluation dataset")
    parser.add_argument("--output", default="data/eval_dataset")
    parser.add_argument("--max-videos", type=int, default=15)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip video download if already present")
    parser.add_argument("--upload-only", action="store_true",
                        help="Pass 1: Download videos + fetch CC + upload to ScreenApp (don't wait for transcripts)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Pass 2: Collect transcripts from ScreenApp for already-uploaded files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # Pass 2: Just collect transcripts from ScreenApp
    if args.collect_only:
        collected = collect_transcripts_from_screenapp(output_dir)
        # Rebuild manifest from collected files
        _rebuild_manifest(output_dir, transcripts_dir, videos_dir)
        return

    manifest = []
    upload_tracking = {}  # screenapp_file_id → video_id
    collected = 0

    print(f"\n=== Dataset Collection (target: {args.max_videos} videos) ===")
    if args.upload_only:
        print("=== MODE: Upload only (transcripts collected later with --collect-only) ===")
    print()

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

        # Save ground truth immediately
        gt_file = transcripts_dir / f"{video_id}_ground_truth.txt"
        gt_file.write_text(gt_text)
        cc_file = transcripts_dir / f"{video_id}_cc.json"
        with open(cc_file, "w") as f:
            json.dump(cc_entries, f, indent=2)

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

        if args.upload_only:
            # Upload to ScreenApp but don't wait for transcript
            sa_file = transcripts_dir / f"{video_id}_screenapp.json"
            if sa_file.exists():
                print(f"     ScreenApp transcript already exists")
            else:
                print(f"     Uploading to ScreenApp (no wait)...", end=" ", flush=True)
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from training.screenapp_transcribe import ScreenAppTranscriber
                    transcriber = ScreenAppTranscriber()

                    import requests
                    headers = {"Authorization": f"Bearer {transcriber.pat_token}"}
                    file_name = Path(video_path).name
                    content_type = "video/mp4"

                    # Step 1: Get pre-signed upload URL (same endpoint as transcribe_file)
                    resp = requests.post(
                        f"{transcriber.api_url}/files/upload/urls/{transcriber.team_id}/{transcriber.folder_id}",
                        json=[{"fileName": file_name, "contentType": content_type}],
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        upload_params = resp.json().get("data", {}).get("uploadParams", [])
                        if not upload_params:
                            print("FAILED — no upload params")
                        else:
                            file_id = upload_params[0]["fileId"]
                            upload_url = upload_params[0]["uploadUrl"]

                            # Step 2: Upload to S3
                            with open(video_path, "rb") as vf:
                                requests.put(upload_url, data=vf, headers={"Content-Type": content_type})

                            # Step 3: Finalize (triggers transcription)
                            requests.post(
                                f"{transcriber.api_url}/files/upload/finalize/{transcriber.team_id}/{transcriber.folder_id}",
                                json={"fileIds": [file_id]},
                                headers=headers,
                            )

                            upload_tracking[file_id] = video_id
                            print(f"OK (file_id={file_id[:8]}...)")
                    else:
                        print(f"FAILED (status={resp.status_code})")
                except Exception as e:
                    print(f"FAILED — {e}")
        else:
            # Full mode: upload and wait for transcript
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

            asr_file = transcripts_dir / f"{video_id}_asr.txt"
            asr_file.write_text(asr_text)

            with open(sa_file, "w") as f:
                json.dump(sa_transcript, f, indent=2)

        entry = {
            "video_id": video_id,
            "title": title,
            "accent": accent,
            "domain": domain,
            "video_path": str(video_path),
            "ground_truth_file": str(gt_file),
            "ground_truth_chars": len(gt_text),
        }

        # Add ASR info if transcript is available
        asr_file = transcripts_dir / f"{video_id}_asr.txt"
        sa_file = transcripts_dir / f"{video_id}_screenapp.json"
        if asr_file.exists():
            entry["asr_file"] = str(asr_file)
            entry["asr_chars"] = len(asr_file.read_text())
        if sa_file.exists():
            entry["screenapp_file"] = str(sa_file)

        manifest.append(entry)
        collected += 1
        print(f"     {'UPLOADED' if args.upload_only else 'COLLECTED'} ({collected}/{args.max_videos})\n")

    # Save upload tracking for collect-only pass
    if upload_tracking:
        tracking_file = output_dir / "upload_tracking.json"
        with open(tracking_file, "w") as f:
            json.dump(upload_tracking, f, indent=2)

    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== {'Upload' if args.upload_only else 'Collection'} Complete ===")
    print(f"Videos processed: {collected}")
    print(f"Manifest: {manifest_file}")

    if args.upload_only:
        print(f"\nNext step: wait for ScreenApp to finish transcribing, then run:")
        print(f"  python scripts/collect_dataset.py --output {args.output} --collect-only")

    from collections import Counter
    accent_counts = Counter(e["accent"] for e in manifest)
    print(f"\nAccent distribution:")
    for accent, count in accent_counts.most_common():
        print(f"  {accent}: {count}")


def _rebuild_manifest(output_dir: Path, transcripts_dir: Path, videos_dir: Path):
    """Rebuild manifest.json from collected files on disk."""
    manifest = []

    for gt_file in sorted(transcripts_dir.glob("*_ground_truth.txt")):
        video_id = gt_file.stem.replace("_ground_truth", "")

        # Find matching files
        sa_file = transcripts_dir / f"{video_id}_screenapp.json"
        asr_file = transcripts_dir / f"{video_id}_asr.txt"
        video_path = None
        for ext in ("mp4", "mkv", "webm"):
            candidate = videos_dir / f"{video_id}.{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break

        if not sa_file.exists() or not asr_file.exists():
            continue

        # Find video info from curated list
        accent = "unknown"
        domain = "unknown"
        title = video_id
        for vid_id, t, a, d in CURATED_VIDEOS:
            if vid_id == video_id:
                title, accent, domain = t, a, d
                break

        entry = {
            "video_id": video_id,
            "title": title,
            "accent": accent,
            "domain": domain,
            "video_path": video_path or "",
            "ground_truth_file": str(gt_file),
            "asr_file": str(asr_file),
            "screenapp_file": str(sa_file),
            "ground_truth_chars": len(gt_file.read_text()),
            "asr_chars": len(asr_file.read_text()),
        }
        manifest.append(entry)
        print(f"  Added to manifest: {video_id} ({title[:40]})")

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest rebuilt: {len(manifest)} videos")
    print(f"Saved to: {manifest_file}")


if __name__ == "__main__":
    main()
