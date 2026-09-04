#!/usr/bin/env bash
# 前台运行 TechDoc QA Pipeline V2（用户指南 converted 全量数据集）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/wugk/.conda/envs/dataflow/bin/python" ]]; then
    PYTHON="/home/wugk/.conda/envs/dataflow/bin/python"
  else
    PYTHON="python3"
  fi
fi

# ── 输入 / 输出（默认 converted 221 chunks）──
export QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-/home/wugk/finetune/finetune/data/用户指南}"
export QA_CACHE_DIR="${QA_CACHE_DIR:-${QA_OUTPUT_DIR}/qa_pipeline_cache}"
export QA_FILE_PREFIX="${QA_FILE_PREFIX:-user_guide_converted_qa}"
export TECHDOC_INPUT="${TECHDOC_INPUT:-${QA_OUTPUT_DIR}/text_chunks_20251119_232705_converted.json}"

# ── 本地 llama-server（与故障管理相同配置）──
export DF_API_KEY="${DF_API_KEY:-local}"
export DF_API_URL="${DF_API_URL:-http://127.0.0.1:19000/v1/chat/completions}"
export DF_MODEL_NAME="${DF_MODEL_NAME:-Qwen3.6-27B-Q5_K_M.gguf}"
export DF_MAX_WORKERS="${DF_MAX_WORKERS:-1}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-4}"
export QA_RESUME="${QA_RESUME:-1}"
export DATASET_LABEL="${DATASET_LABEL:-用户指南}"
export DISTILL_TAG="${DISTILL_TAG:-运维}"

exec "$PYTHON" "$SCRIPT_DIR/脚本07_QA管道主程序_Python入口.py" "$@"
