#!/usr/bin/env bash
# 后台运行 Pipeline（nohup），SSH 断开后仍继续执行
# 支持断点续跑：中断后重跑同一命令（QA_RESUME=1 默认开启）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-/home/wugk/finetune/finetune/data/故障管理}"
LOG_DIR="${LOG_DIR:-${QA_OUTPUT_DIR}/logs}"
mkdir -p "$LOG_DIR"

export QA_RESUME="${QA_RESUME:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qa_pipeline_${TS}.log"
PID_FILE="${LOG_DIR}/qa_pipeline.pid"
LATEST_LOG_LINK="${LOG_DIR}/qa_pipeline_latest.log"

# 若已有运行中的任务，提示用户
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] 已有任务在运行 (PID=$OLD_PID)"
    echo "  进度: $SCRIPT_DIR/脚本02_查看QA管道运行进度.sh"
    echo "  日志: tail -f $LATEST_LOG_LINK"
    echo "  如需强制重启: kill $OLD_PID && rm $PID_FILE"
    exit 1
  fi
fi

echo "启动后台任务..."
echo "  日志: $LOG_FILE"
echo "  PID 文件: $PID_FILE"
echo "  断点续跑: QA_RESUME=$QA_RESUME"

nohup "$SCRIPT_DIR/脚本03_前台运行_故障管理数据集.sh" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"

echo "已启动 PID=$(cat "$PID_FILE")"
echo ""
echo "查看进度:"
echo "  $SCRIPT_DIR/脚本02_查看QA管道运行进度.sh"
echo ""
echo "查看日志:"
echo "  tail -f $LATEST_LOG_LINK"
echo ""
echo "查看是否在跑:"
echo "  ps -p \$(cat $PID_FILE) -o pid,cmd"
