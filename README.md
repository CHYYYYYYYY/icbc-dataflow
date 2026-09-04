# ICBC DataFlow 技术文档 QA 流水线

本仓库收录了从 OCR 技术文档生成高质量 QA/SFT 语料的完整代码、运行脚本、技术文档和数据集。核心流程为：蒸馏出题 → 初答生成 → 答案溯源/净化 → 问答双向精炼 → Rubric 评分 → MetaFilter 托底 → SFT 导出。

## 目录

- `DataFlow/`：DataFlow 完整源码快照，含本项目扩展的技术文档 QA 算子、Prompt、服务和存储层。
- `pipeline/`：业务编排、多跳 QA、OCR 转换、SFT 导出/清洗及评测脚本。
- `scripts/`：前台/后台运行、进度监控及模型下载脚本。
- `training/`：LLaMA-Factory 数据接入脚本和 Qwen3 LoRA 配置。
- `docs/`：方案、演进记录、运行手册和完整技术报告。
- `data/`：训练集、OceanStor 多跳测试集及对应 Rubric。

建议先阅读 [`docs/文档15_DataFlow技术文档QA流水线完整技术报告.plan.md`](docs/%E6%96%87%E6%A1%A315_DataFlow%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3QA%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%AE%8C%E6%95%B4%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.plan.md)，文档索引见 [`docs/00_目录说明与文件索引.md`](docs/00_%E7%9B%AE%E5%BD%95%E8%AF%B4%E6%98%8E%E4%B8%8E%E6%96%87%E4%BB%B6%E7%B4%A2%E5%BC%95.md)。

## 数据集

| 文件 | 用途 | 规模 |
|---|---|---:|
| `data/sft_train.jsonl` | Alpaca 格式训练集 | 5,662 条 |
| `data/oceanstor_multihop_testset_v1.json` | OceanStor 多跳测试集 | 10 题 |
| `data/oceanstor_multihop_rubric_v1.json` | 测试集配套评分规则 | 10 组 item rubric |

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ./DataFlow

export DF_API_KEY="your-api-key"
export DF_API_URL="https://your-openai-compatible-endpoint/v1/chat/completions"
export DF_MODEL_NAME="your-model"
export TECHDOC_INPUT="/path/to/text_chunks.json"
export QA_OUTPUT_DIR="$PWD/outputs"

python scripts/脚本07_QA管道主程序_Python入口.py
```

运行脚本中的服务器路径仅是历史默认值，均可通过同名环境变量覆盖。仓库不包含 API Key、GitLab Token 或 SSH 凭据。

## 来源与许可

`DataFlow/` 基于 OpenDCAI/Open-DataFlow DataFlow 提交 `3a8a10b184ad843c84bcae531b4ea35eae079829`，并包含服务器上尚未提交的本项目扩展；快照日期为 2026-09-04。DataFlow 原始许可证见 `DataFlow/LICENSE`。
