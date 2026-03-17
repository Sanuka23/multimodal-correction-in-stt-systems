#!/bin/bash
# Download training datasets for ASR correction finetuning
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/datasets"

mkdir -p "$DATA_DIR"

echo "=== Dataset Download ==="

# LibriSpeech clean-100 (~6GB, 100 hours of read speech)
if [ ! -d "$DATA_DIR/LibriSpeech/train-clean-100" ]; then
    echo "Downloading LibriSpeech train-clean-100..."
    wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz -P "$DATA_DIR/"
    echo "Extracting..."
    tar xzf "$DATA_DIR/train-clean-100.tar.gz" -C "$DATA_DIR/"
    rm "$DATA_DIR/train-clean-100.tar.gz"
    echo "LibriSpeech clean-100 ready."
else
    echo "LibriSpeech clean-100 already exists."
fi

# AMI Meeting Corpus - just transcripts + sample audio
AMI_DIR="$DATA_DIR/ami"
if [ ! -d "$AMI_DIR" ]; then
    mkdir -p "$AMI_DIR"
    echo "Downloading AMI Meeting Corpus annotations..."
    # Download word-level annotations
    wget -c http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip -P "$AMI_DIR/"
    cd "$AMI_DIR" && unzip -o ami_public_manual_1.6.2.zip && rm ami_public_manual_1.6.2.zip
    echo "AMI annotations ready."
    echo ""
    echo "NOTE: AMI audio files must be downloaded separately due to license."
    echo "Visit: https://groups.inf.ed.ac.uk/ami/download/"
    echo "Download headset mix audio and place in: $AMI_DIR/audio/"
else
    echo "AMI corpus already exists."
fi

echo ""
echo "=== Download Complete ==="
echo "LibriSpeech: $DATA_DIR/LibriSpeech/train-clean-100/"
echo "AMI:         $AMI_DIR/"
