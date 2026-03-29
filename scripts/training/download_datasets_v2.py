#!/usr/bin/env python3
"""Download external datasets for accent-diverse ASR correction training.

Downloads:
  - L2-ARCTIC: Non-native English (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese)
  - Svarah: Indian English accent (9.6h, 117 speakers)

Usage:
    python scripts/download_datasets_v2.py
"""

import os
import subprocess
import sys
from pathlib import Path


def download_svarah(output_dir: str):
    """Download Svarah Indian English dataset from HuggingFace."""
    print("\n=== Downloading Svarah (Indian English) ===")
    svarah_dir = Path(output_dir) / "svarah"

    if svarah_dir.exists() and any(svarah_dir.iterdir()):
        print(f"  Already exists at {svarah_dir}, skipping")
        return

    try:
        from datasets import load_dataset
        print("  Loading from HuggingFace (ai4bharat/Svarah)...")
        ds = load_dataset("ai4bharat/Svarah", split="test")
        ds.save_to_disk(str(svarah_dir))
        print(f"  Saved {len(ds)} examples to {svarah_dir}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Try: pip install datasets")


def download_l2arctic(output_dir: str):
    """Download L2-ARCTIC non-native English corpus."""
    print("\n=== Downloading L2-ARCTIC (Non-native English) ===")
    l2_dir = Path(output_dir) / "l2arctic"
    l2_dir.mkdir(parents=True, exist_ok=True)

    # L2-ARCTIC accent groups with download URLs
    accent_groups = {
        "hindi": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_hindi.zip",
        "korean": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_korean.zip",
        "mandarin": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_mandarin.zip",
        "spanish": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_spanish.zip",
        "arabic": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_arabic.zip",
        "vietnamese": "https://psi.engr.tamu.edu/l2-arctic-corpus/data/arctic_l2_vietnamese.zip",
    }

    for accent, url in accent_groups.items():
        accent_dir = l2_dir / f"arctic_l2_{accent}"
        if accent_dir.exists() and any(accent_dir.iterdir()):
            print(f"  {accent}: already exists, skipping")
            continue

        zip_file = l2_dir / f"arctic_l2_{accent}.zip"
        print(f"  {accent}: downloading...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["wget", "-q", "-O", str(zip_file), url],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                # Try curl as fallback
                result = subprocess.run(
                    ["curl", "-sL", "-o", str(zip_file), url],
                    capture_output=True, text=True, timeout=600,
                )

            if zip_file.exists() and zip_file.stat().st_size > 1000:
                subprocess.run(
                    ["unzip", "-q", "-o", str(zip_file), "-d", str(l2_dir)],
                    capture_output=True, timeout=300,
                )
                zip_file.unlink()
                print(f"OK")
            else:
                print(f"FAILED (empty/missing zip)")
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")
        except Exception as e:
            print(f"FAILED ({e})")


def main():
    datasets_dir = "data/datasets"
    os.makedirs(datasets_dir, exist_ok=True)

    print("=== Dataset Download v2 ===")
    print(f"Output: {datasets_dir}/\n")

    # Download Svarah first (smaller, faster)
    download_svarah(datasets_dir)

    # Download L2-ARCTIC
    download_l2arctic(datasets_dir)

    # Summary
    print("\n=== Download Complete ===")
    for name in ["svarah", "l2arctic"]:
        path = Path(datasets_dir) / name
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
            print(f"  {name}: {size:.0f} MB")
        else:
            print(f"  {name}: NOT DOWNLOADED")


if __name__ == "__main__":
    main()
