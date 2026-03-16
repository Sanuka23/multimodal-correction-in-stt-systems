#!/bin/bash
# Run the ASR Correction Evaluation Dashboard
#
# Usage:
#   ./run_dashboard.sh                    # Uses default data directory
#   ASR_EVAL_DATA_DIR=/path/to/data ./run_dashboard.sh  # Custom data directory
#
# The dashboard runs on port 8501 by default.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default data directory (FYP/Tests folder)
export ASR_EVAL_DATA_DIR="${ASR_EVAL_DATA_DIR:-/Users/sanukathamuditha/Desktop/FYP/Tests}"

# Load .env from python services root if exists
if [ -f "../../.env" ]; then
    export $(grep -v '^#' ../../.env | xargs)
fi

echo "Starting ASR Correction Evaluation Dashboard..."
echo "Data directory: $ASR_EVAL_DATA_DIR"
echo "Dashboard URL: http://localhost:8501"
echo ""

# Run Streamlit
streamlit run dashboard.py --server.port 8501 --server.headless true
