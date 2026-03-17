"""Upload audio files to ScreenApp and retrieve ASR transcripts.

Uses ScreenApp's own transcription pipeline (Groq/Whisper/Gemini) to generate
ASR outputs that match real production errors.

Usage:
    python training/screenapp_transcribe.py \
        --audio-dir data/datasets/LibriSpeech/train-clean-100/19/198/  \
        --output data/transcripts/librispeech_19_198.jsonl \
        --max-files 50

Environment variables (or .env file):
    SCREENAPP_API_URL=http://localhost:8081/api/v2
    SCREENAPP_PAT_TOKEN=<your_pat>
    SCREENAPP_TEAM_ID=<team_id>
    SCREENAPP_FOLDER_ID=<folder_id>
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class ScreenAppTranscriber:
    """Upload audio to ScreenApp API and retrieve transcripts."""

    def __init__(
        self,
        api_url: str = None,
        pat_token: str = None,
        team_id: str = None,
        folder_id: str = None,
        poll_interval: float = 5.0,
        max_poll_time: float = 300.0,
    ):
        self.api_url = (api_url or os.environ.get("SCREENAPP_API_URL", "http://localhost:8081/v2")).rstrip("/")
        self.pat_token = pat_token or os.environ.get("SCREENAPP_PAT_TOKEN")
        self.team_id = team_id or os.environ.get("SCREENAPP_TEAM_ID")
        self.folder_id = folder_id or os.environ.get("SCREENAPP_FOLDER_ID")
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time

        if not self.pat_token:
            raise ValueError("SCREENAPP_PAT_TOKEN is required")
        if not self.team_id:
            raise ValueError("SCREENAPP_TEAM_ID is required")
        if not self.folder_id:
            raise ValueError("SCREENAPP_FOLDER_ID is required")

        # Support both PAT and JWT auth
        if self.pat_token.startswith("eyJ"):
            # Looks like a JWT token
            self.headers = {"Authorization": f"Bearer {self.pat_token}"}
        else:
            # PAT token
            self.headers = {"x-screenapp-token": self.pat_token}
        logger.info("ScreenApp API: %s (team=%s, folder=%s)", self.api_url, self.team_id, self.folder_id)

    def _get_content_type(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
        }.get(ext, "audio/wav")

    def transcribe_file(self, audio_path: str) -> Optional[dict]:
        """Upload a single audio file to ScreenApp and wait for transcript.

        Returns dict with keys: file_id, transcript_text, segments, or None on failure.
        """
        audio_path = str(audio_path)
        file_name = Path(audio_path).name
        content_type = self._get_content_type(audio_path)

        try:
            # Step 1: Get pre-signed upload URL
            resp = requests.post(
                f"{self.api_url}/files/upload/urls/{self.team_id}/{self.folder_id}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"files": [{"contentType": content_type, "name": file_name}]},
            )
            resp.raise_for_status()
            data = resp.json()
            upload_params = data.get("data", {}).get("uploadParams", [])
            if not upload_params:
                logger.error("No upload params returned for %s", file_name)
                return None

            file_id = upload_params[0]["fileId"]
            upload_url = upload_params[0]["uploadUrl"]
            logger.info("Got upload URL for %s (file_id=%s)", file_name, file_id)

            # Step 2: Upload audio file to S3
            with open(audio_path, "rb") as f:
                put_resp = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": content_type},
                )
                put_resp.raise_for_status()
            logger.info("Uploaded %s to S3", file_name)

            # Step 3: Finalize upload (triggers transcription)
            finalize_resp = requests.post(
                f"{self.api_url}/files/upload/finalize/{self.team_id}/{self.folder_id}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={
                    "file": {
                        "fileId": file_id,
                        "contentType": content_type,
                        "name": file_name,
                    }
                },
            )
            finalize_resp.raise_for_status()
            logger.info("Finalized %s — transcription started", file_name)

            # Step 4: Poll for transcript completion
            start_time = time.time()
            while time.time() - start_time < self.max_poll_time:
                time.sleep(self.poll_interval)
                file_resp = requests.get(
                    f"{self.api_url}/files/{file_id}",
                    headers=self.headers,
                )
                if file_resp.status_code != 200:
                    continue

                file_data = file_resp.json()
                transcript = file_data.get("transcript")
                if transcript and transcript.get("text"):
                    logger.info("Transcript ready for %s (%d chars)",
                                file_name, len(transcript["text"]))
                    return {
                        "file_id": file_id,
                        "file_name": file_name,
                        "transcript_text": transcript.get("text", ""),
                        "segments": transcript.get("segments", []),
                    }

            logger.warning("Transcript timeout for %s after %.0fs", file_name, self.max_poll_time)
            return None

        except Exception as e:
            logger.error("Failed to transcribe %s: %s", audio_path, e)
            return None

    def _upload_file(self, audio_path: str) -> Optional[dict]:
        """Upload a single file and trigger transcription. Returns file_id or None."""
        audio_path = str(audio_path)
        file_name = Path(audio_path).name
        content_type = self._get_content_type(audio_path)

        try:
            # Get upload URL
            resp = requests.post(
                f"{self.api_url}/files/upload/urls/{self.team_id}/{self.folder_id}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"files": [{"contentType": content_type, "name": file_name}]},
            )
            resp.raise_for_status()
            upload_params = resp.json().get("data", {}).get("uploadParams", [])
            if not upload_params:
                return None

            file_id = upload_params[0]["fileId"]
            upload_url = upload_params[0]["uploadUrl"]

            # Upload to S3
            with open(audio_path, "rb") as f:
                requests.put(upload_url, data=f, headers={"Content-Type": content_type}).raise_for_status()

            # Finalize (triggers transcription)
            requests.post(
                f"{self.api_url}/files/upload/finalize/{self.team_id}/{self.folder_id}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"file": {"fileId": file_id, "contentType": content_type, "name": file_name}},
            ).raise_for_status()

            return {"file_id": file_id, "file_name": file_name, "audio_path": str(audio_path)}

        except Exception as e:
            logger.error("Upload failed for %s: %s", file_name, e)
            return None

    def _poll_transcript(self, file_id: str) -> Optional[dict]:
        """Poll for a single file's transcript."""
        try:
            resp = requests.get(f"{self.api_url}/files/{file_id}", headers=self.headers)
            if resp.status_code != 200:
                return None
            body = resp.json()
            # Response structure: {success, data: {file, transcript, ...}}
            data = body.get("data", body)
            transcript = data.get("transcript")
            if transcript and transcript.get("text"):
                return {
                    "transcript_text": transcript.get("text", ""),
                    "segments": transcript.get("segments", []),
                }
        except Exception:
            pass
        return None

    def transcribe_batch(self, audio_files: list, output_path: str = None, batch_size: int = 20) -> list:
        """Fast batch transcription: upload all files first, then poll all in parallel.

        Strategy:
        1. Upload files in batches (batch_size at a time)
        2. After each batch upload, start polling all pending files
        3. Save results as they come in
        """
        results = []
        total = len(audio_files)
        pending = {}  # file_id → {file_name, audio_path, upload_time}

        # Phase 1: Upload all files in batches
        logger.info("=== Phase 1: Uploading %d files (batch_size=%d) ===", total, batch_size)
        for i, audio_path in enumerate(audio_files):
            logger.info("Uploading %d/%d: %s", i + 1, total, Path(audio_path).name)
            info = self._upload_file(audio_path)
            if info:
                pending[info["file_id"]] = {
                    **info,
                    "upload_time": time.time(),
                }

            # After each batch, do a quick poll round
            if (i + 1) % batch_size == 0 or i == total - 1:
                logger.info("Batch uploaded. Pending: %d. Quick poll round...", len(pending))
                self._poll_round(pending, results, output_path)

            # Small delay to avoid rate limiting
            time.sleep(0.2)

        # Phase 2: Poll remaining pending files
        logger.info("=== Phase 2: Polling %d remaining transcripts ===", len(pending))
        poll_start = time.time()
        while pending and (time.time() - poll_start) < self.max_poll_time:
            time.sleep(self.poll_interval)
            self._poll_round(pending, results, output_path)
            if pending:
                logger.info("  Still waiting for %d transcripts...", len(pending))

        if pending:
            logger.warning("Timed out waiting for %d transcripts", len(pending))

        logger.info("=== Complete: %d/%d files transcribed ===", len(results), total)
        return results

    def _poll_round(self, pending: dict, results: list, output_path: str = None):
        """Poll all pending files once and move completed ones to results."""
        completed_ids = []
        for file_id, info in pending.items():
            transcript = self._poll_transcript(file_id)
            if transcript:
                result = {**info, **transcript}
                results.append(result)
                completed_ids.append(file_id)
                logger.info("  Transcript ready: %s (%d chars)",
                            info["file_name"], len(transcript["transcript_text"]))
                if output_path:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

        for fid in completed_ids:
            del pending[fid]


def find_audio_files(directory: str, extensions=(".wav", ".flac", ".mp3", ".m4a")) -> list:
    """Find all audio files in a directory recursively."""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(files)


def collect_existing_transcripts(transcriber, folder_id: str, output_path: str, max_files: int = 0):
    """Collect transcripts for files already uploaded to a ScreenApp folder.

    Lists files in the folder and fetches their transcripts.
    """
    logger.info("=== Collecting existing transcripts from folder %s ===", folder_id)

    # List files in folder (paginate to get all)
    resp = requests.get(
        f"{transcriber.api_url}/files?parentId={folder_id}&limit=1000&sort=createdAt&order=desc",
        headers=transcriber.headers,
    )
    if resp.status_code != 200:
        logger.error("Failed to list files: %d", resp.status_code)
        return []

    body = resp.json()
    files_data = body.get("data", body)

    # Handle different response formats
    if isinstance(files_data, dict):
        file_list = files_data.get("fileSystem", files_data.get("files", files_data.get("items", [])))
    elif isinstance(files_data, list):
        file_list = files_data
    else:
        file_list = []

    # Filter to only File type (not folders)
    file_list = [f for f in file_list if f.get("type") == "File"]

    if max_files > 0:
        file_list = file_list[:max_files]

    logger.info("Found %d files in folder", len(file_list))

    results = []
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(file_list):
        file_id = f.get("_id", f.get("id", ""))
        file_name = f.get("name", "unknown")

        if not file_id:
            continue

        transcript = transcriber._poll_transcript(file_id)
        if transcript:
            result = {
                "file_id": file_id,
                "file_name": file_name,
                **transcript,
            }
            results.append(result)
            with open(output_path, "a") as fout:
                fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("  Collected %d/%d transcripts", len(results), len(file_list))

    logger.info("=== Collected %d/%d transcripts ===", len(results), len(file_list))
    return results


def main():
    parser = argparse.ArgumentParser(description="Upload audio to ScreenApp for transcription")
    parser.add_argument("--audio-dir", help="Directory containing audio files (for upload mode)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--limit", type=int, default=0, help="Alias for --max-files")
    parser.add_argument("--extensions", default=".wav,.flac", help="Audio file extensions (comma-separated)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Skip upload, just collect transcripts for already-uploaded files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already present in the output JSONL")
    parser.add_argument("--skip-from", nargs="*", default=[],
                        help="Additional JSONL files to check for already-transcribed files")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of files to upload per batch (default: 20)")
    args = parser.parse_args()

    # --limit is alias for --max-files
    max_files = args.limit or args.max_files

    transcriber = ScreenAppTranscriber()

    if args.collect_only:
        collect_existing_transcripts(transcriber, transcriber.folder_id, args.output, max_files)
        return

    if not args.audio_dir:
        parser.error("--audio-dir is required unless using --collect-only")

    extensions = tuple(args.extensions.split(","))
    audio_files = find_audio_files(args.audio_dir, extensions)

    # Skip files already transcribed (check output file + any --skip-from files)
    if args.skip_existing:
        existing_names = set()
        check_files = [args.output] + args.skip_from
        for check_path in check_files:
            if Path(check_path).exists():
                with open(check_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            existing_names.add(entry.get("file_name", ""))
        before = len(audio_files)
        audio_files = [f for f in audio_files if f.name not in existing_names]
        logger.info("Skipping %d already-transcribed files (%d remaining)",
                     before - len(audio_files), len(audio_files))

    if max_files > 0:
        audio_files = audio_files[:max_files]

    logger.info("Found %d audio files to process in %s", len(audio_files), args.audio_dir)

    if not audio_files:
        logger.info("No audio files to process")
        return

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    transcriber.transcribe_batch(audio_files, output_path=args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
