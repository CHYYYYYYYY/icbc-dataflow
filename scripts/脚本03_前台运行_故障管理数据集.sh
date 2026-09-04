#!/usr/bin/env bash
# 前台运行 TechDoc QA Pipeline V2（故障管理 converted 全量数据集）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/wugk/.conda/envs/dataflow/bin/python" ]]; then
    PYTHON="/home/wugk/.conda/envs/dataflow/bin/python"
  else
    PYTHON="python3"
  fi
fi

# ── 输入 / 输出（默认 converted 584 chunks；小数据集试跑请改环境变量）──
export QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-/home/wugk/finetune/finetune/data/故障管理}"
export QA_CACHE_DIR="${QA_CACHE_DIR:-${QA_OUTPUT_DIR}/qa_pipeline_cache}"
export QA_FILE_PREFIX="${QA_FILE_PREFIX:-fault_mgmt_converted_qa}"
export TECHDOC_INPUT="${TECHDOC_INPUT:-${QA_OUTPUT_DIR}/text_chunks_20260113_150815_converted.json}"

# ── 阿里云 token-plan OpenAI 兼容 API ───────────────────────────────────────
export DF_API_KEY="${DF_API_KEY:-}"
export DF_API_URL="${DF_API_URL:-https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions}"
export DF_MODEL_NAME="${DF_MODEL_NAME:-qwen3.6-plus}"
export DF_MAX_WORKERS="${DF_MAX_WORKERS:-20}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-4}"

exec "$PYTHON" "$SCRIPT_DIR/脚本07_QA管道主程序_Python入口.py" "$@"
