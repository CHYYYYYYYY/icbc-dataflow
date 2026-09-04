# 技术文档 QA 语料接入 LLaMA-Factory 微调指南

> 给程同学：DataFlow QA Pipeline 产出的语料 → LLaMA-Factory LoRA 微调，三步搞定。

---

## 1. 语料是什么格式？

Pipeline 导出（`--with-context`）的 `*_sft.jsonl` 已是 **Alpaca 格式**，LLaMA-Factory 直接可用：

```json
{
  "instruction": "问题（refined_reverse_question）",
  "input": "原文 chunk（raw_content）",
  "output": "答案（refined_answer）",
  "question_type": "default"
}
```

训练时 LLaMA-Factory 会把 `instruction + input` 拼成用户提示，`output` 作为模型学习目标。

| 数据集 | 语料文件 | 条数 |
|--------|----------|------|
| 故障管理 | `/home/wugk/finetune/finetune/data/故障管理/fault_mgmt_converted_qa_sft.jsonl` | 1544 |
| 用户指南 | `/home/wugk/finetune/finetune/data/用户指南/user_guide_converted_qa_sft.jsonl` | Pipeline 跑完后生成 |

---

## 2. 放入 LLaMA-Factory（推荐软链接）

```bash
/home/wugk/finetune/finetune/LLaMA-Factory/import_techdoc_sft.sh
```

脚本会把 `*_sft.jsonl` 软链接到 `LLaMA-Factory/data/`：

- `fault_mgmt_qa.jsonl`
- `user_guide_qa.jsonl`（存在时）
- `techdoc_qa_all.jsonl`（两者合并，可选）

`data/dataset_info.json` 已注册三个数据集名：`fault_mgmt_qa`、`user_guide_qa`、`techdoc_qa_all`。

---

## 3. 启动微调

```bash
cd /home/wugk/finetune/finetune/LLaMA-Factory

# 单数据集（故障管理 1544 条）
llamafactory-cli train examples/train_lora/qwen3_lora_sft_techdoc.yaml

# 或 WebUI
./run.sh   # 浏览器选 dataset=fault_mgmt_qa, template=qwen3
```

**合并两个数据集训练**：改 yaml 里 `dataset` 行：

```yaml
dataset: fault_mgmt_qa,user_guide_qa
```

关键参数说明（已在 `qwen3_lora_sft_techdoc.yaml` 配好）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `template` | `qwen3` | 与 Qwen3 系列对齐 |
| `cutoff_len` | `8192` | input 含长文档 chunk，需足够长 |
| `finetuning_type` | `lora` | LoRA 微调，省显存 |
| `output_dir` | `saves/qwen3-8b/lora/sft_techdoc` | 与旧 fm_qa 训练目录分开 |

---

## 4. 合并 LoRA 权重（训练完成后）

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

把 `adapter_name_or_path` 改成 `saves/qwen3-8b/lora/sft_techdoc` 即可。

---

## 5. Docker 场景说明

之前打的 **`dataflow-qa` 镜像**是跑 QA 生成 Pipeline 的，**不是**微调用的。

微调有两种方式：

**A. 宿主机直接跑**（当前环境已有 LLaMA-Factory）

```bash
cd /home/wugk/finetune/finetune/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen3_lora_sft_techdoc.yaml
```

**B. LLaMA-Factory Docker**（`docker/docker-cuda/docker-compose.yml`）

挂载语料和代码：

```bash
docker run --gpus all -it --rm \
  -v /home/wugk/finetune/finetune/LLaMA-Factory:/app \
  -v /home/wugk/finetune/finetune/data:/data \
  -w /app \
  llamafactory \
  llamafactory-cli train examples/train_lora/qwen3_lora_sft_techdoc.yaml
```

容器内需先执行 `import_techdoc_sft.sh` 或手动 ln -s。

---

## 6. 与旧 `fm_qa` 数据集的区别

| | 旧 `fm_qa.json` | 新 `fault_mgmt_qa.jsonl` |
|--|-----------------|--------------------------|
| 来源 | 早期手工/试跑 | Pipeline V2 全量 21 步过滤 |
| input | 多为空 | 含 raw_content 原文 |
| 条数 | ~3000+ | 1544（质量更高） |
| yaml | `qwen3_lora_sft.yaml` | `qwen3_lora_sft_techdoc.yaml` |

建议新项目用 `fault_mgmt_qa`，旧 `fm_qa` 保留作对比。

---

## 7. 单跳 + 多跳合并微调

多跳语料由 [文档16](文档16_单文档跨chunk多跳QA技术路线.plan.md) / `multihop_cross_chunk.py` 产出，文件名形如 `*_multihop_sft.jsonl`。

### 7.1 字段差异

| 字段 | 单跳 `*_sft.jsonl` | 多跳 `*_multihop_sft.jsonl` |
|------|---------------------|------------------------------|
| `instruction` | 单 chunk 可答的问题 | 必须跨 2~3 chunk |
| `input` | 通常 1 个 chunk | `【片段N \| 节号】` 拼接多 chunk |
| `output` | 答案 | 答案（可选拼接推理链，见下） |
| 额外 | `question_type` 等 | `question_type=多跳`, `multihop_type`, `reasoning_steps`, `complexity_score` |

LLaMA-Factory **最少只需** `instruction` + `input` + `output` 三字段即可训练。

### 7.2 合并示例（比例建议 单跳 70% : 多跳 30%）

```bash
FINETUNE_DATA=/home/wugk/finetune/finetune/data
DORADO="$FINETUNE_DATA/OceanStor Dorado V6系列 6.1.x & V700R001 性能监控指南"

# 单跳（Dorado conv102 清洗版）
SINGLE="$DORADO/sft_train.jsonl"
# 多跳
MULTI="$DORADO/dorado_perf_multihop_multihop_sft.jsonl"

# 全库合并（若已有 sft_train_all.jsonl）
MERGED="$FINETUNE_DATA/sft_train_all_with_multihop.jsonl"
cat "$FINETUNE_DATA/sft_train_all.jsonl" "$MULTI" > "$MERGED"

# 或仅 Dorado 单跳+多跳
cat "$SINGLE" "$MULTI" > "$DORADO/dorado_sft_single_and_multihop.jsonl"
```

### 7.3 推理链训练（可选方案 B）

若希望模型学习多步推理，可将多跳 `output` 预处理为：

```
【推理过程】
1. （step1 text）
2. （step2 text）
【最终答案】
（answer）
```

单跳数据保持原 `output` 不变，合并后一起训练。

### 7.4 dataset_info.json 注册

在 `LLaMA-Factory/data/dataset_info.json` 增加：

```json
"dorado_multihop_qa": {
  "file_name": "dorado_multihop_qa.jsonl",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

软链接：

```bash
ln -sf "$MULTI" /home/wugk/finetune/finetune/LLaMA-Factory/data/dorado_multihop_qa.jsonl
```

训练时：`dataset: techdoc_qa_all,dorado_multihop_qa` 或单独 `dorado_multihop_qa`。

### 7.5 注意

- 多跳 `input` 更长，确保 yaml 中 `cutoff_len` ≥ 8192（必要时 16384）
- 合并前检查多跳条数，避免小样本过拟合（单文档全量预期约 25~40 条/文档）
- 多跳与单跳 instruction 语义去重：同一章节事实不宜重复出太多相似问法

---

## 快速检查清单

- [ ] Pipeline 跑完，`*_sft.jsonl` 已生成
- [ ] 执行 `import_techdoc_sft.sh`
- [ ] `dataset_info.json` 里有对应数据集名
- [ ] yaml 里 `dataset` 和 `cutoff_len` 正确
- [ ] 基座模型路径存在（当前：`/home/wugk/finetune/QA/models/Qwen/Qwen3-8B`）
- [ ] 训练完成后 merge LoRA → 推理验证
