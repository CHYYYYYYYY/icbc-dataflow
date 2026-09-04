#!/usr/bin/env bash
# 后台运行 Pipeline（本地 llama-server + 用户指南 converted 221 chunks）
# 支持断点续跑：中断后重跑同一命令即可（QA_RESUME=1 默认开启）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 固定 converted 全量默认（专用脚本，避免 shell 残留环境变量污染）
export QA_OUTPUT_DIR="/home/wugk/finetune/finetune/data/用户指南"
export QA_CACHE_DIR="${QA_CACHE_DIR:-${QA_OUTPUT_DIR}/qa_pipeline_cache}"
export QA_FILE_PREFIX="${QA_FILE_PREFIX:-user_guide_converted_qa}"
export TECHDOC_INPUT="${TECHDOC_INPUT:-${QA_OUTPUT_DIR}/text_chunks_20251119_232705_converted.json}"

LOG_DIR="${LOG_DIR:-${QA_OUTPUT_DIR}/logs}"

# ── 本地 llama-server（与故障管理相同稳定配置）──
export DF_API_URL="${DF_API_URL:-http://127.0.0.1:19000/v1/chat/completions}"
export DF_API_KEY="${DF_API_KEY:-local}"
export DF_MODEL_NAME="${DF_MODEL_NAME:-Qwen3.6-27B-Q5_K_M.gguf}"
export DF_MAX_WORKERS="${DF_MAX_WORKERS:-1}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-4}"
export QA_RESUME="${QA_RESUME:-1}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qa_pipeline_local_${TS}.log"
PID_FILE="${LOG_DIR}/qa_pipeline.pid"
LATEST_LOG_LINK="${LOG_DIR}/qa_pipeline_latest.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] 已有任务在运行 (PID=$OLD_PID)"
    echo "  进度: QA_OUTPUT_DIR=$QA_OUTPUT_DIR $SCRIPT_DIR/脚本02_查看QA管道运行进度.sh --once"
    echo "  日志: tail -f $LATEST_LOG_LINK"
    echo "  如需强制重启: kill $OLD_PID && rm $PID_FILE"
    exit 1
  fi
fi

echo "启动后台任务（本地 llama-server）..."
echo "  输入: $TECHDOC_INPUT"
echo "  缓存: $QA_CACHE_DIR"
echo "  API : $DF_API_URL"
echo "  断点续跑: QA_RESUME=$QA_RESUME"
echo "  日志: $LOG_FILE"

nohup "$SCRIPT_DIR/脚本04_前台运行_用户指南数据集.sh" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"

echo "已启动 PID=$(cat "$PID_FILE")"
echo ""
echo "实时进展:"
echo "  QA_OUTPUT_DIR=$QA_OUTPUT_DIR $SCRIPT_DIR/脚本02_查看QA管道运行进度.sh"
echo ""
echo "实时日志:"
echo "  tail -f $LATEST_LOG_LINK"
