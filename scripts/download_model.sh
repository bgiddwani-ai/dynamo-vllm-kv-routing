#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?Set MODEL_DIR to the model directory on the filesystem exposed by your PVC}"
mkdir -p "$MODEL_DIR"
.venv/bin/hf download nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \
  --revision 04822723e77e036ddf2d24e83c6d469d3b009252 \
  --local-dir "$MODEL_DIR"
.venv/bin/python scripts/verify_model.py "$MODEL_DIR" | tee model-validation.json
