#!/usr/bin/env bash
# 后台运行单文档跨 chunk 多跳 QA（旁路，不改动 V2 单跳管道）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DORADO_DIR="/home/wugk/finetune/finetune/data/OceanStor Dorado V6系列 6.1.x & V700R001 性能监控指南"

export QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-$DORADO_DIR}"
export OCR_SOURCE="${OCR_SOURCE:-$DORADO_DIR/text_chunks_20260401_144243.json}"
export MH_FILE_PREFIX="${MH_FILE_PREFIX:-dorado_perf_multihop}"
export DATASET_LABEL="${DATASET_LABEL:-OceanStor Dorado 性能监控指南}"
export LOG_DIR="${LOG_DIR:-${QA_OUTPUT_DIR}/logs}"
mkdir -p "$LOG_DIR"

export DF_API_URL="${DF_API_URL:-http://127.0.0.1:19000/v1/chat/completions}"
export DF_API_KEY="${DF_API_KEY:-local}"
export DF_MODEL_NAME="${DF_MODEL_NAME:-Qwen3.6-27B-Q5_K_M.gguf}"
export DF_MAX_WORKERS="${DF_MAX_WORKERS:-2}"

export MH_MAX_CLUSTERS="${MH_MAX_CLUSTERS:-80}"
export MH_NUM_Q="${MH_NUM_Q:-1}"
export MH_MIN_COMPLEXITY="${MH_MIN_COMPLEXITY:-0.55}"
export MH_ENABLE_DEGENERACY="${MH_ENABLE_DEGENERACY:-1}"
export MH_DRY_RUN="${MH_DRY_RUN:-0}"
export MH_RESUME="${MH_RESUME:-1}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/wugk/.conda/envs/dataflow/bin/python" ]]; then
    PYTHON="/home/wugk/.conda/envs/dataflow/bin/python"
  else
    PYTHON="python3"
  fi
fi
export PYTHON

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/multihop_${TS}.log"
PID_FILE="${LOG_DIR}/multihop.pid"
LATEST_LOG_LINK="${LOG_DIR}/multihop_latest.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] 已有任务在运行 (PID=$OLD_PID)"
    exit 1
  fi
fi

echo "启动后台任务（多跳跨 chunk）— $DATASET_LABEL"
echo "  OCR_SOURCE : $OCR_SOURCE"
echo "  OUTPUT_DIR : $QA_OUTPUT_DIR"
echo "  PREFIX     : $MH_FILE_PREFIX"
echo "  MH_DRY_RUN : $MH_DRY_RUN"
echo "  API        : $DF_API_URL"
echo "  日志       : $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/../pipeline/multihop_cross_chunk.py" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"

echo "已启动 PID=$(cat "$PID_FILE")"
echo "监控: tail -f $LATEST_LOG_LINK"
