#!/usr/bin/env bash
# 实时显示 TechDoc QA Pipeline 进展（step 缓存 + 日志摘要）
set -euo pipefail

QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-/home/wugk/finetune/finetune/data/故障管理}"
QA_CACHE_DIR="${QA_CACHE_DIR:-${QA_OUTPUT_DIR}/qa_pipeline_cache}"
QA_FILE_PREFIX="${QA_FILE_PREFIX:-fault_mgmt_converted_qa}"
LOG_DIR="${LOG_DIR:-${QA_OUTPUT_DIR}/logs}"
PID_FILE="${LOG_DIR}/qa_pipeline.pid"
LATEST_LOG="${LOG_DIR}/qa_pipeline_latest.log"
TOTAL_STEPS=21
INTERVAL="${WATCH_INTERVAL:-30}"

_latest_step() {
  local latest=0
  shopt -s nullglob
  for f in "$QA_CACHE_DIR/${QA_FILE_PREFIX}_step"*.json; do
    local base
    base="$(basename "$f")"
    if [[ "$base" =~ _step([0-9]+)\.json$ ]]; then
      local n="${BASH_REMATCH[1]}"
      if (( n > latest )); then latest=$n; fi
    fi
  done
  echo "$latest"
}

_row_count() {
  local step="$1"
  local f="$QA_CACHE_DIR/${QA_FILE_PREFIX}_step${step}.json"
  [[ -f "$f" ]] || { echo "-"; return; }
  python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?"
}

_show_once() {
  local now latest rows pct pid_status health
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  latest="$(_latest_step)"
  rows="$(_row_count "$latest")"
  if (( latest > 0 )); then
    pct=$(( latest * 100 / TOTAL_STEPS ))
  else
    pct=0
  fi

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    pid_status="运行中 PID=$(cat "$PID_FILE")"
  else
    pid_status="未检测到运行中的 Pipeline"
  fi

  health="$(curl -s -m 2 http://127.0.0.1:19000/health 2>/dev/null || echo 'N/A')"

  echo "============================================================"
  echo "[$now] TechDoc QA Pipeline 进展"
  echo "  状态     : $pid_status"
  echo "  llama    : $health"
  echo "  最新 step: step${latest}.json / step${TOTAL_STEPS}  (${pct}%)"
  echo "  当前行数 : ${rows}"
  echo "  缓存目录 : $QA_CACHE_DIR"
  echo "------------------------------------------------------------"
  if [[ -L "$LATEST_LOG" || -f "$LATEST_LOG" ]]; then
    echo "最近日志:"
    tail -5 "$LATEST_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^/  /'
  else
    local newest
    newest="$(ls -t "$LOG_DIR"/qa_pipeline_*.log 2>/dev/null | head -1 || true)"
    if [[ -n "$newest" ]]; then
      echo "最近日志 ($newest):"
      tail -5 "$newest" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^/  /'
    fi
  fi
  echo "============================================================"
}

if [[ "${1:-}" == "--once" ]]; then
  _show_once
  exit 0
fi

echo "每 ${INTERVAL}s 刷新一次；Ctrl+C 退出"
echo "单次查看: $0 --once"
echo ""

while true; do
  clear 2>/dev/null || true
  _show_once
  if [[ -f "$LATEST_LOG" ]] && grep -q "Pipeline 运行完成" "$LATEST_LOG" 2>/dev/null; then
    echo ""
    echo "Pipeline 已完成。"
    break
  fi
  sleep "$INTERVAL"
done
