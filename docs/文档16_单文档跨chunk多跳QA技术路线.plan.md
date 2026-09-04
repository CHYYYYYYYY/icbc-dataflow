# 单文档跨 Chunk 多跳 QA 技术路线

> 旁路于 V2 单跳管道（脚本07），通用脚本 + 环境变量驱动；主程序：`finetune/test/multihop_cross_chunk.py`，后台启动：脚本13。

---

## 1. 目标与边界

**目标**：同一文档内，综合 **2~3 个相关章节 chunk** 才能回答的多跳 QA，产出带 `reasoning_steps` 的 SFT 语料。

**非目标（本阶段）**：跨文档多跳、改写 V2 主链路、重跑 OCR。

与 V2 关系：

```
OCR chunks ──→ V2 单跳（脚本07）──→ *_qa_sft.jsonl
           └──→ 多跳旁路（脚本13）──→ *_multihop_sft.jsonl
                                    └──→ 合并微调（见文档14 §7）
```

---

## 2. 主程序与脚本

| 文件 | 说明 |
|------|------|
| `finetune/test/multihop_cross_chunk.py` | 聚类 → StepA 构图 → StepB 出题 → L1~L4 质检 → SFT |
| `脚本13_后台运行_单文档多跳跨chunk_本地llama-server.sh` | 通用后台启动（Dorado 为默认测试文档） |

### 2.1 环境变量

| 变量 | 必填 | 默认 | 说明 |
|------|:----:|------|------|
| `OCR_SOURCE` | 是 | — | 原始 `text_chunks_*.json`（**不用** `*_converted.json`） |
| `QA_OUTPUT_DIR` | 否 | OCR 父目录 | 产出根目录 |
| `MH_FILE_PREFIX` | 否 | `multihop_qa` | 缓存/产出前缀 |
| `MH_MAX_CLUSTERS` | 否 | 80 | 最多处理聚类数 |
| `MH_DRY_RUN` | 否 | 0 | 1=只聚类不调 LLM |
| `MH_RESUME` | 否 | 1 | 1=跳过 raw 中已有 cluster |
| `MH_MIN_COMPLEXITY` | 否 | 0.55 | L4 阈值 |
| `MH_ENABLE_DEGENERACY` | 否 | 1 | L3 退化检测 |
| `MH_MAX_LLM_RETRIES` | 否 | 1 | StepA/B 失败重试次数 |
| `DF_API_URL` | 否 | `127.0.0.1:19000` | llama-server |
| `DF_MAX_WORKERS` | 否 | 2 | 并发 |
| `TMPDIR` | 建议 | — | 根分区满时设为 `/sdd/home/wugk/tmp` |

### 2.2 常用命令

```bash
# 阶段0：只聚类
export TMPDIR=/sdd/home/wugk/tmp
OCR_SOURCE='.../text_chunks_20260401_144243.json' \
QA_OUTPUT_DIR='.../OceanStor Dorado...' \
MH_FILE_PREFIX='dorado_perf_multihop' \
MH_DRY_RUN=1 \
python finetune/test/multihop_cross_chunk.py

# 阶段1：后台全量（Resume 续跑）
MH_MAX_CLUSTERS=80 MH_RESUME=1 \
/home/wugk/.cursor/plans/脚本13_后台运行_单文档多跳跨chunk_本地llama-server.sh

# 监控
tail -f '<QA_OUTPUT_DIR>/logs/multihop_latest.log'
```

---

## 3. 流水线设计

### 3.1 预处理

1. 加载 OCR `result[]`（`id`, `content`, `title`）
2. 过滤：过短 chunk、表格占比 ≥60%、目录型 chunk
3. 章节号抽取（`2.1.1`、`### 1.3` 等）
4. 同 parent 章节滑窗 size=2/3，要求 ≥2 个不同 section

Dorado dry-run：**64 可用 chunk → 61 clusters**。

### 3.2 两步生成

**Step A**：跨片段依赖图（`entities`, `edges`, `reasoning_path`, `path_type`）

- 2 片段时 path 长度=2 且 fragment 交替
- 失败可 `repair_dependency_graph()` 自动修正相邻同 fragment
- 校验失败时带原因重试 1 次

**Step B**：基于依赖链出题（`question`, `reasoning_steps`, `answer`, …）

- JSON 解析取**最后一个** `<answer>` 块（避免 thinking 内伪 answer）
- 解析失败重试 1 次

### 3.3 四层质检

| 层 | 内容 | 实现 |
|----|------|------|
| L0 | JSON 解析 | 程序 |
| L1 | 结构 S1~S7（步数、片段覆盖、泄题、步相似度等） | 程序 |
| L2 | 跨源 C1~C3（步骤 grounding、片段覆盖） | 程序；中文 2-gram token |
| L3 | 退化检测 | **程序化**（见 §4 与 Plan 差异） |
| L4 | complexity_score ≥ 0.55 | 程序 |

---

## 4. 与原始 Plan 的实现差异

| Plan 设计 | 当前实现 | 原因 |
|-----------|----------|------|
| L3 LLM 单片段可答性审计 | 程序化 L3：步骤已覆盖 ≥2 片段则通过；否则检查单片段自洽 | LLM 审计误杀率高（pilot 4/5）；更快 |
| L3 事实真子集可答性 | 未实现 | 成本高；阶段2可选 |
| MetaFilter 六维抽检 | 未接入 | 阶段2可选 |
| C2 supporting_facts 跨 chunk | 已改为以 reasoning_steps 跨片段为准 | 措辞相近时 facts 易全落同一 chunk |

Pilot（5 cluster）优化后通过率 **5/5**；全量 61 cluster 进行中。

---

## 5. 产出格式

路径：`<QA_OUTPUT_DIR>/<prefix>_multihop_sft.jsonl`

```json
{
  "instruction": "多跳问题",
  "input": "【片段1 | 节号 1.1】\\n...\\n\\n【片段2 | 节号 1.2】\\n...",
  "output": "最终答案",
  "question_type": "多跳",
  "multihop_type": "流程链|配置链|对比链|因果链|排障链",
  "reasoning_steps": [
    {"step": 1, "fragment": 1, "uses_facts": ["E1"], "text": "..."},
    {"step": 2, "fragment": 2, "uses_facts": ["E2"], "text": "..."}
  ],
  "supporting_facts": ["...", "..."],
  "reasoning_path": ["E1", "E2"],
  "complexity_score": 0.895,
  "source_sections": ["1.1", "1.2"],
  "source_chunk_ids": ["uuid-1", "uuid-2"],
  "dataset": "文档名"
}
```

中间缓存（`multihop_cache/`）：

- `<prefix>_clusters.json` — 聚类清单
- `<prefix>_raw.json` — 含 pass/reject 原因与 raw LLM 输出

**微调字段用法**见 [文档14 §7](文档14_QA语料接入LLaMA-Factory微调指南.plan.md)。

---

## 6. Dorado 实测路径

| 产物 | 路径 |
|------|------|
| OCR 输入 | `.../text_chunks_20260401_144243.json` |
| 多跳 SFT | `.../dorado_perf_multihop_multihop_sft.jsonl` |
| 单跳 SFT | `.../dorado_perf_qa_conv102_sft.jsonl` |
| 日志 | `.../logs/multihop_latest.log` |

---

## 7. 阶段2 可选增强

1. 恢复 LLM L3 + 子集事实退化检测（提高假多跳拦截，降低通过率）
2. MetaFilter 六维抽检
3. golden set 50 条人工标注，定期回归假多跳率
4. embedding 聚类突破「仅兄弟章节」

---

## 8. 验收清单

- [x] 通用脚本 + 环境变量
- [x] MH_DRY_RUN 聚类
- [x] 两步生成 + reasoning_path 跨 fragment
- [x] SFT 含 reasoning_steps / complexity_score
- [x] 日志拒绝统计
- [x] Dorado pilot 5/5 通过
- [ ] 全量 61 cluster 跑完并人工抽检假多跳率 < 15%
- [ ] 与单跳合并微调并验证推理能力

---

## 9. 相关文档

- 总体方案：[文档01](文档01_技术文档QA管道V2总体设计方案.plan.md)
- 微调接入：[文档14](文档14_QA语料接入LLaMA-Factory微调指南.plan.md) §7 单跳+多跳合并
- 实施 Plan 源稿：`单文档跨chunk多跳qa_4415a191.plan.md`
