#!/usr/bin/env bash
# 下载 Qwen3.6-27B-Q5_K_M.gguf（ModelScope，国内网络友好）
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/sdd/models/Qwen3.6-27B-GGUF}"
LOG_DIR="${MODEL_DIR}/logs"
mkdir -p "$LOG_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/sdd/conda/miniconda3/bin/modelscope" ]]; then
    MODELSCOPE="/sdd/conda/miniconda3/bin/modelscope"
  else
    MODELSCOPE="modelscope"
  fi
else
  MODELSCOPE="${PYTHON} -m modelscope"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/download_ms_${TS}.log"
PID_FILE="${LOG_DIR}/download.pid"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] 已有下载任务在运行 (PID=$OLD_PID)"
    echo "  日志: $(ls -t ${LOG_DIR}/download_ms_*.log 2>/dev/null | head -1)"
    exit 1
  fi
fi

echo "启动 ModelScope 下载..."
echo "  模型: unsloth/Qwen3.6-27B-GGUF"
echo "  文件: Qwen3.6-27B-Q5_K_M.gguf (~18.2GB)"
echo "  目录: $MODEL_DIR"
echo "  日志: $LOG_FILE"

nohup $MODELSCOPE download \
  --model unsloth/Qwen3.6-27B-GGUF \
  --local_dir "$MODEL_DIR" \
  --include Qwen3.6-27B-Q5_K_M.gguf \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "已启动 PID=$(cat "$PID_FILE")"
echo ""
echo "查看进度:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看文件:"
echo "  ls -lh $MODEL_DIR/Qwen3.6-27B-Q5_K_M.gguf"
