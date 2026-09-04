#!/usr/bin/env bash
# 用法: MODEL_NAME=xxx bash run_lf_eval_1010.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 "${DIR}/test/generate_model_answers.py" \
  --qa "${DIR}/data/combined_three_documents_qa_minimal_1010_v1.json" \
  --out-dir "${OUT_DIR:-${DIR}/data/eval_finetuned_model}" \
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model "${MODEL_NAME:?请设置 MODEL_NAME}" \
  --workers "${WORKERS:-2}" \
  ${LIMIT:+--limit "${LIMIT}"}
