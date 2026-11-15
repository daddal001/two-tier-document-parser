#!/bin/sh
set -eu

# Validate configured model roots, download missing models, and fail on errors.
# A nonempty Hugging Face cache alone is not evidence that models are ready.
python3 -m two_tier_parser.accurate.prepare_models
export MINERU_MODEL_SOURCE=local
exec python3 -m uvicorn two_tier_parser.accurate.main:app \
    --host 0.0.0.0 --port 8005 --timeout-keep-alive "${UVICORN_TIMEOUT_KEEP_ALIVE:-600}"
