#!/usr/bin/env bash
# 后台运行 TechDoc QA Pipeline V2 — 01-01 告警处理（本地 llama-server）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-/home/wugk/finetune/finetune/data/01-01 告警处理}"
LOG_DIR="${LOG_DIR:-${QA_OUTPUT_DIR}/logs}"
mkdir -p "$LOG_DIR"

export QA_OUTPUT_DIR
export QA_CACHE_DIR="${QA_CACHE_DIR:-${QA_OUTPUT_DIR}/qa_pipeline_cache}"
export QA_FILE_PREFIX="${QA_FILE_PREFIX:-alarm_handling_qa}"
export TECHDOC_INPUT="${TECHDOC_INPUT:-${QA_OUTPUT_DIR}/text_chunks_20260401_160730_converted.json}"
export DATASET_LABEL="${DATASET_LABEL:-01-01 告警处理}"
export DISTILL_TAG="${DISTILL_TAG:-告警处理}"

export DF_API_URL="${DF_API_URL:-http://127.0.0.1:19000/v1/chat/completions}"
export DF_API_KEY="${DF_API_KEY:-local}"
export DF_MODEL_NAME="${DF_MODEL_NAME:-Qwen3.6-27B-Q5_K_M.gguf}"
export DF_MAX_WORKERS="${DF_MAX_WORKERS:-2}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-4}"
export QA_RESUME="${QA_RESUME:-1}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/wugk/.conda/envs/dataflow/bin/python" ]]; then
    PYTHON="/home/wugk/.conda/envs/dataflow/bin/python"
  else
    PYTHON="python3"
  fi
fi
export PYTHON

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qa_pipeline_local_${TS}.log"
PID_FILE="${LOG_DIR}/qa_pipeline.pid"
LATEST_LOG_LINK="${LOG_DIR}/qa_pipeline_latest.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] 已有任务在运行 (PID=$OLD_PID)"
    exit 1
  fi
fi

echo "启动后台任务（本地 llama-server）— $DATASET_LABEL"
echo "  输入: $TECHDOC_INPUT"
echo "  API : $DF_API_URL"
echo "  并发: DF_MAX_WORKERS=$DF_MAX_WORKERS"
echo "  断点续跑: QA_RESUME=$QA_RESUME"
echo "  日志: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/脚本07_QA管道主程序_Python入口.py" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"

echo "已启动 PID=$(cat "$PID_FILE")"
