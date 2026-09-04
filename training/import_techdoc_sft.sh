#!/usr/bin/env bash
# 将 QA Pipeline 导出的 *_sft.jsonl 接入 LLaMA-Factory data/ 目录（软链接，不复制）
set -euo pipefail

LF_DATA="/home/wugk/finetune/finetune/LLaMA-Factory/data"
FINETUNE_DATA="/home/wugk/finetune/finetune/data"

link_one() {
  local src="$1" dst_name="$2"
  if [[ ! -f "$src" ]]; then
    echo "[SKIP] 源文件不存在: $src"
    return 1
  fi
  ln -sf "$src" "${LF_DATA}/${dst_name}"
  local n
  n="$(wc -l < "$src")"
  echo "[OK] ${dst_name} <- $src  (${n} 条)"
}

link_one "${FINETUNE_DATA}/故障管理/sft_train.jsonl" "sft_train.jsonl" || true
link_one "${FINETUNE_DATA}/用户指南/user_guide_converted_qa_sft.jsonl" "user_guide_qa.jsonl" || true

# 可选：合并两个数据集为一个文件供 techdoc_qa_all 使用
if [[ -f "${LF_DATA}/sft_train.jsonl" ]] && [[ -f "${LF_DATA}/user_guide_qa.jsonl" ]]; then
  cat "${LF_DATA}/sft_train.jsonl" "${LF_DATA}/user_guide_qa.jsonl" > "${LF_DATA}/techdoc_qa_all.jsonl"
  echo "[OK] techdoc_qa_all.jsonl 已合并 $(wc -l < "${LF_DATA}/techdoc_qa_all.jsonl") 条"
fi

echo ""
echo "下一步："
echo "  cd /home/wugk/finetune/finetune/LLaMA-Factory"
echo "  llamafactory-cli train examples/train_lora/qwen3_lora_sft_techdoc.yaml"
echo ""
echo "合并训练时在 yaml 里改 dataset: sft_train,user_guide_qa"
