#!/bin/bash
set -e

echo "🚀 Starting Accurate Parser Service..."

# Ensure magic-pdf.json exists
if [ ! -f "/root/magic-pdf.json" ]; then
    echo "❌ /root/magic-pdf.json not found!"
    exit 1
fi

echo "✅ Configuration found at /root/magic-pdf.json"

# Check if models are already available (pre-downloaded during build or in volume)
MODEL_DIR="/root/.cache/huggingface/hub"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "✅ Models found in cache (pre-downloaded during build)"
else
    echo "⚠️ Models not found in image. This should not happen if build completed successfully."
    echo "   Models should have been pre-downloaded during Docker build."
    echo "   Attempting fallback download (this may take 10-20 minutes)..."
    
    # Fallback: download models if somehow missing
    python3 -m mineru.cli.models_download -s huggingface -m all || \
    python3 -c "from mineru.cli.models_download import download_models; download_models('huggingface', 'all')" || \
    echo "⚠️ Fallback download failed. Models will auto-download on first parse request."
fi

# Set MINERU_MODEL_SOURCE=local after models are downloaded (matches MinerU official Dockerfiles)
# This tells MinerU to use the pre-downloaded models instead of trying to download again
export MINERU_MODEL_SOURCE=local

# Start the application (main.py follows industry standard naming convention)
echo "🎯 Starting Uvicorn server on port 8005..."
exec python3 -m uvicorn two_tier_parser.accurate.main:app \
    --host 0.0.0.0 \
    --port 8005 \
    --timeout-keep-alive 600

