#!/bin/bash
# Setup script for Auto-AVSR lip reading model
#
# This downloads the Auto-AVSR repository and VSR model weights.
# Run this once to enable the "auto_avsr" AVSR mode.
#
# Usage:
#   cd multimodal-correction-in-stt-systems
#   bash scripts/setup_auto_avsr.sh
#
# After setup, set in .env or config:
#   AVSR_MODE=auto_avsr

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
AVSR_DIR="$MODELS_DIR/auto_avsr"

echo "=== Auto-AVSR Setup ==="
echo "Project: $PROJECT_DIR"
echo "Models:  $MODELS_DIR"
echo ""

# Step 1: Install Python dependencies
echo "Step 1: Installing Python dependencies..."
pip install pytorch-lightning sentencepiece av mediapipe
echo "Done."
echo ""

# Step 2: Clone Auto-AVSR repo
if [ -d "$AVSR_DIR" ]; then
    echo "Step 2: Auto-AVSR directory already exists at $AVSR_DIR"
    echo "  Pulling latest changes..."
    cd "$AVSR_DIR" && git pull || true
else
    echo "Step 2: Cloning Auto-AVSR repository..."
    mkdir -p "$MODELS_DIR"
    git clone https://github.com/mpc001/auto_avsr.git "$AVSR_DIR"
fi
echo "Done."
echo ""

# Step 3: Download VSR model weights
echo "Step 3: Downloading VSR model weights..."
echo "  This may take a few minutes (~1GB download)"
cd "$AVSR_DIR"

# Check if model weights exist
if [ -f "$AVSR_DIR/benchmarks/LRS3/models/vsr_trlrs3vox2_base.pth" ]; then
    echo "  Model weights already downloaded."
else
    echo "  Please download the VSR model manually:"
    echo "  1. Visit: https://github.com/mpc001/auto_avsr#pre-trained-models"
    echo "  2. Download the LRS3 VSR model (vsr_trlrs3vox2_base.pth)"
    echo "  3. Place it in: $AVSR_DIR/benchmarks/LRS3/models/"
    echo ""
    echo "  Alternatively, the model will be downloaded automatically"
    echo "  on first use if huggingface-hub is installed."
fi
echo ""

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '$AVSR_DIR')
try:
    from pipelines.pipeline import InferencePipeline
    print('  InferencePipeline: OK')
except ImportError as e:
    print(f'  InferencePipeline: FAILED ({e})')

try:
    import mediapipe
    print('  MediaPipe: OK')
except ImportError:
    print('  MediaPipe: FAILED')

try:
    import pytorch_lightning
    print('  PyTorch Lightning: OK')
except ImportError:
    print('  PyTorch Lightning: FAILED')

try:
    import sentencepiece
    print('  SentencePiece: OK')
except ImportError:
    print('  SentencePiece: FAILED')
"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To enable Auto-AVSR, set in your config:"
echo "  avsr_mode: auto_avsr"
echo "  avsr_model_dir: $AVSR_DIR"
echo ""
echo "Or set environment variable:"
echo "  export AUTO_AVSR_DIR=$AVSR_DIR"
