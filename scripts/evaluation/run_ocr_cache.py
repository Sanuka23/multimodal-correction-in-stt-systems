#!/usr/bin/env python3
"""Run OCR on SlideAVSR videos and cache the results for Phase 2 evaluation.

For each video in data/eval_dataset/slideavsr/, runs PaddleOCR extraction via
the existing asr_correction.ocr_extractor module and stores the raw XML
alongside the transcript files. Skips videos that already have a cached
result. Re-running is safe and resumable.

Run with the conda Python:
    /opt/homebrew/Caskroom/miniforge/base/bin/python3 scripts/evaluation/run_ocr_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OCR_CACHE_DIR = PROJECT_ROOT / "data/eval_dataset/slideavsr/ocr_cache"
MANIFEST_PATH = PROJECT_ROOT / "data/eval_dataset/slideavsr/manifest.json"


def main() -> int:
    from asr_correction.ocr_extractor import _extract_ocr_sync

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Running OCR on {len(manifest)} SlideAVSR videos ===\n")

    for i, entry in enumerate(manifest, 1):
        file_id = entry["file_id"]
        video_path = entry.get("video_path", "")

        if not video_path or not Path(video_path).exists():
            print(f"[{i}/{len(manifest)}] {file_id} — SKIP (no video)")
            continue

        cache_path = OCR_CACHE_DIR / f"{file_id}.xml"
        if cache_path.exists():
            print(f"[{i}/{len(manifest)}] {file_id} — cached ({cache_path.stat().st_size} bytes)")
            continue

        print(f"[{i}/{len(manifest)}] {file_id} — running OCR on {Path(video_path).name}...",
              flush=True)
        t0 = time.time()
        try:
            xml = _extract_ocr_sync(
                video_path,
                interval_s=10.0,
                max_frames=40,
                min_confidence=0.5,
                dedup_threshold=0.85,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        elapsed = time.time() - t0
        if xml is None:
            print(f"    no text found ({elapsed:.0f}s)")
            # Write empty cache to avoid re-running
            cache_path.write_text("<ocr-extraction></ocr-extraction>")
            continue

        cache_path.write_text(xml)
        print(f"    saved {len(xml)} chars ({elapsed:.0f}s)")

    print("\n=== Summary ===")
    cached = sorted(OCR_CACHE_DIR.glob("*.xml"))
    total_size = sum(p.stat().st_size for p in cached)
    print(f"Cached OCR files: {len(cached)}")
    print(f"Total cache size: {total_size / 1024:.1f} KB")
    print(f"Cache location: {OCR_CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
