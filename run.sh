#!/bin/bash
set -e

PORT=${API_PORT:-8000}

echo "Starting ASR Correction API on port $PORT..."
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
