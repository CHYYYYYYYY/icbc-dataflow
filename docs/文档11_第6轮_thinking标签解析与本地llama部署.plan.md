# TechDoc QA Pipeline V2 — Round 6 修复记录（qwen3.6-plus `<think>` 全链路）

本文件记录 2026-05-23 ~ 2026-05-24 对话中的全部代码改动、运行结论、本地 llama.cpp 部署与 Pipeline 联调说明。

---

## 背景

`qwen3.6-plus`（阿里云 token-plan）返回格式为：

```
<think>...模型思考过程...</think>
<answer>...实际输出...</answer>
```

多个算子在解析 LLM 响应时**未先剥离 thinking 块**，导致：
- 误命中 thinking 里的 prompt 模板占位符（如 `[Improved Question Start]你改进后的问题[Improved Question End]`）
- JSON / 分数 / 答案 / 问题提取失败
- 默认 FAIL / DROP / 空字段，漏斗大量丢行（曾出现 `Final rows: 0`）

---

## 核心工具函数

**文件**：`DataFlow/dataflow/utils/meta_label_filters.py`

新增 `extract_llm_answer_text(text)`：
1. 去掉 `<think>...</think>`
2. 优先提取 `<answer>...</answer>` 内容
3. 供各算子统一调用

---

## 代码改动清单

### 1. Pipeline 层 — 答案 strip + 过滤

**文件**：`finetune/test/test_filter.py`

| 改动 | 说明 |
|------|------|
| `storage_strip_llm_answer_columns()` | 新增工具：批量 strip 指定列的 thinking 标签 |
| AnswerGenerator 后 | strip `initial_answer` |
| AnswerRewriter 后 | strip `clean_answer` |
| AnswerRefiner 后 | strip `refined_answer` |
| RubricScorer 前 | strip `clean_answer`、`refined_answer` |
| `_initial_answer_valid` / `_refined_answer_valid` | 过滤前先 `extract_llm_answer_text`，避免 thinking 里出现 `OUTOFSCOPE` 误杀 |

### 2. JSON 解析 — Grounding / Rubric / Rewriter

**文件**：`DataFlow/dataflow/prompts/tech_doc_qa.py`

`parse_json_llm_response()` 增强：
- 候选顺序：`extract_llm_answer_text(text)` → 去 thinking 后的全文 → 原文
- 避免 thinking 块内 `{ }` 被误当 JSON 解析

**受益算子**（间接修复，无需改各自文件）：
- `AnswerGroundingFilter`
- `AnswerRewriter`
- `RubricScorer`

### 3. MetaFilter 评分解析

**文件**：`DataFlow/dataflow/operators/text_pt/eval/meta_sample_evaluator.py`

`get_score()`：
- 先 `extract_llm_answer_text`
- 在 cleaned 文本中搜索 `[num, num, ...]` 分数行
- 不再盲目取 raw 响应最后一行（可能是 `</answer>`）

### 4. Distill 问题生成

**文件**：`DataFlow/dataflow/operators/text_sft/generate/distill_question_generator.py`

`extract_json_array()` / `extract_json_object()`：
- 入口先 `extract_llm_answer_text`
- fallback 才用原始文本 regex

### 5. Answer Critique 解析

**文件**：`DataFlow/dataflow/operators/text_sft/tech_doc/answer_critique_evaluator.py`

`parse_answer_critique()` 入口加 `extract_llm_answer_text`。

### 6. QuestionRefiner 问题提取（Round 6 重点）

**文件**：`DataFlow/dataflow/operators/text_sft/refine/question_refiner.py`

`_extract_refined_question()` 修复：
- 入口 `extract_llm_answer_text`，所有策略只在 `<answer>` 内搜索
- 占位符黑名单：`'...'`、`'你改进后的问题'`、`'改进后的问题'`
- `_parse_fact_closure_check()` 入口同样 strip thinking

**修复前现象**（step21）：
- `refined_reverse_question = "你改进后的问题"`（命中 thinking 里模板）
- `refined_reverse_question = "..."`（占位符）
- 长段英文 thinking 正文被 Strategy 2 误提取

### 7. AnswerRefiner 答案提取（Round 6 重点）

**文件**：`DataFlow/dataflow/operators/text_sft/refine/answer_refiner.py`

`_extract_refined_answer()` 修复（与 QuestionRefiner 同模式）：
- 入口 `extract_llm_answer_text`
- 占位符黑名单：`'...'`、`'你改进后的答案'` 等
- `<answer></answer>` 空标签 → 返回 `None`（fallback 原答案）
- Strategy 4 长度阈值从 50 降至 10（适配中文短答案）

**修复前现象**（step18）：
- `refined_answer = "..."` → Rubric 收到空答案 → `rubric_raw=''` → 默认 FAIL
- 6 条合法问题被误杀（非 MCQ 选项类）

---

## 全链路审计结论（TechDocQAPipelineV2 用到的算子）

| 文件 | 解析函数 | thinking 处理 |
|------|----------|---------------|
| `distill_question_generator.py` | `extract_json_array/object` | ✅ |
| `answer_grounding_filter.py` | `parse_json_llm_response` | ✅ |
| `answer_rewriter.py` | `parse_json_llm_response` | ✅ |
| `answer_critique_evaluator.py` | `parse_answer_critique` | ✅ |
| `answer_refiner.py` | `_extract_refined_answer` | ✅ Round 6 |
| `rubric_scorer.py` | `parse_json_llm_response` | ✅ |
| `meta_sample_evaluator.py` | `get_score` | ✅ |
| `question_refiner.py` | `_extract_refined_question` | ✅ Round 6 |
| `text_answer_generator.py` | 存 raw，后续 strip | ✅ |

**不在当前 pipeline 内**（暂未改）：`text_question_generator.py`、`alpagasus_sample_evaluator.py`、`treeinstruct_sample_evaluator.py`、`instag_sample_evaluator.py`、`condor_generator.py`

---

## 试跑验证（dataflow 小数据集，12 条 QA）

修复 AnswerRefiner 后重跑 `text_chunks_20260113_150815_dataflow.json`：

| Step | 行数 | 说明 |
|------|------|------|
| step6 | 12 | grounding: REWRITE=3, KEEP=9 |
| step18 | 12 | rubric: **PASS=12**（修复前 6 FAIL） |
| step21 | 若干 | MetaFilter 后保留 |

REWRITE 3 条全程保留到 step18，`refined_answer` 均为真实内容（非 `'...'`）。

---

## 漏斗说明（用户常见问题）

### REWRITE 行去哪了？

REWRITE **不是被跳过**，而是走完 Grounding → Rewrite → Rubric 后，Rubric FAIL 被 `_rubric_pass` 过滤。修复前很多 REWRITE/FAIL 是因为 `refined_answer='...'` 导致 Rubric 解析失败。

### step18 多条 → step21 只剩 1 条？

典型漏斗：
1. **Rubric PASS 过滤**：FAIL 行删除（修复前大量因 `'...'` 误 FAIL）
2. **MetaFilter**：`MetaScore < 3.5` 或评分解析失败（None）删除

### MCQ 选项被 Distiller 当成问题

如 `"A. 从 8.2.0..."` 这类选项句式，Rubric 给 FAIL 是**正常行为**（非 bug）。

---

## 运行脚本变更（2026-05-24）

默认输入从小数据集切回全量 converted：

| 项 | 值 |
|----|-----|
| 输入 | `text_chunks_20260113_150815_converted.json`（584 chunks） |
| 缓存 | `qa_pipeline_cache/` |
| 前缀 | `fault_mgmt_converted_qa` |

**小数据集试跑**（改环境变量，不动默认）：

**云端 token-plan**：

```bash
export TECHDOC_INPUT=/home/wugk/finetune/finetune/data/故障管理/text_chunks_20260113_150815_dataflow.json
export QA_CACHE_DIR=/home/wugk/finetune/finetune/data/故障管理/qa_pipeline_cache_dataflow
export QA_FILE_PREFIX=fault_mgmt_dataflow_qa
/home/wugk/.cursor/plans/脚本08_后台运行_故障管理_阿里云API.sh
```

**本地 llama-server**（429 或离线时使用，见下文「本地 Pipeline 试跑」）：

```bash
export TECHDOC_INPUT=/home/wugk/finetune/finetune/data/故障管理/text_chunks_20260113_150815_dataflow.json
export QA_CACHE_DIR=/home/wugk/finetune/finetune/data/故障管理/qa_pipeline_cache_local
export QA_FILE_PREFIX=fault_mgmt_local_qa
export DF_API_URL=http://127.0.0.1:19000/v1/chat/completions
export DF_API_KEY=local
export DF_MODEL_NAME=Qwen3.6-27B-Q5_K_M.gguf
export DF_MAX_WORKERS=1
/home/wugk/.cursor/plans/脚本03_前台运行_故障管理数据集.sh
```

---

## API 配置与 429 配额耗尽

### 当前使用的协议

| 项 | 值 |
|----|-----|
| 协议 | **OpenAI 兼容**（非 Anthropic） |
| Base URL | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| Pipeline 实际 URL | `.../compatible-mode/v1/chat/completions` |
| 模型 | `qwen3.6-plus` |

Anthropic 端点 `https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic` **未使用**。

### 429 错误

```
HTTP 429
"Your token-plan quota has been exhausted."
```

**原因**：Token Plan Credits 用完，不是 URL 配错。

**查看额度**：
- [Token Plan 控制台](https://bailian.console.aliyun.com/?tab=plan#/efm/subscription/token-plan)
- [费用中心 Token Plan](https://billing-cost.console.aliyun.com/token-plan/summary)

**验证 API**：

```bash
curl -s -m 15 -o /tmp/tp_test.json -w "HTTP %{http_code}\n" \
  -H "Authorization: Bearer $DF_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"hi"}]}' \
  "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
cat /tmp/tp_test.json
```

**额度恢复后重启**（建议降并发）：

```bash
kill $(cat /home/wugk/finetune/finetune/data/故障管理/logs/qa_pipeline.pid) 2>/dev/null
export DF_MAX_WORKERS=5
/home/wugk/.cursor/plans/脚本08_后台运行_故障管理_阿里云API.sh
```

**注意**：Pipeline 不支持断点续跑；重启会从头执行并覆盖 step 缓存。

---

## 本地模型下载：Qwen3.6-27B-Q5_K_M

Hugging Face 在此服务器**网络不通**，改用 ModelScope。

| 项 | 值 |
|----|-----|
| 模型文件 | `Qwen3.6-27B-Q5_K_M.gguf` |
| 来源 | `unsloth/Qwen3.6-27B-GGUF`（ModelScope） |
| 大小 | ~18.2 GB |
| 保存路径 | `/sdd/models/Qwen3.6-27B-GGUF/` |

**启动下载**：

```bash
mkdir -p /sdd/models/Qwen3.6-27B-GGUF/logs
LOG="/sdd/models/Qwen3.6-27B-GGUF/logs/download_ms_$(date +%Y%m%d_%H%M%S).log"
nohup /sdd/conda/miniconda3/bin/modelscope download \
  --model unsloth/Qwen3.6-27B-GGUF \
  --local_dir /sdd/models/Qwen3.6-27B-GGUF \
  --include Qwen3.6-27B-Q5_K_M.gguf \
  > "$LOG" 2>&1 &
echo $! > /sdd/models/Qwen3.6-27B-GGUF/logs/download.pid
```

**监控**：

```bash
tail -f /sdd/models/Qwen3.6-27B-GGUF/logs/download_ms_*.log
ls -lh /sdd/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q5_K_M.gguf
ps -p $(cat /sdd/models/Qwen3.6-27B-GGUF/logs/download.pid) -o pid,cmd
```

**显存参考**：Q5_K_M 建议 ≥24GB VRAM（更稳 32GB）。

---

## 本地 llama.cpp 推理服务（2026-05-24 晚间）

因 token-plan **429 配额耗尽**，改为本地双卡 RTX 3090 跑 `Qwen3.6-27B-Q5_K_M.gguf`。

### 方案选型与踩坑

| 方案 | 结果 | 原因 |
|------|------|------|
| Hugging Face 下载 | ❌ | 服务器 `Network is unreachable` |
| **ModelScope 下载** | ✅ | 18.2GB，路径 `/sdd/models/Qwen3.6-27B-GGUF/` |
| Ollama 0.7.1 加载 GGUF | ❌ | `unknown model architecture: 'qwen35'` |
| **llama.cpp 源码编译** | ✅ | 支持 qwen35 架构 |

**编译环境**：

- 系统 `cmake 3.16.3` 过旧 → 使用 `/sdd/tools/cmake-3.28.3/bin/cmake`
- GitHub 不通 → 镜像克隆：`https://ghfast.top/https://github.com/ggml-org/llama.cpp.git`
- OpenSSL not found 仅为 HTTPS 警告，**不影响本地 OpenAI API**

```bash
cd /sdd/models/llama.cpp
/sdd/tools/cmake-3.28.3/bin/cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
/sdd/tools/cmake-3.28.3/bin/cmake --build build --config Release -j$(nproc)
ls -lh build/bin/llama-server   # 编译产物
```

### 端口与浏览器 404

| 现象 | 原因 | 处理 |
|------|------|------|
| `18000` bind 失败 | 被 llamafactory Docker 占用（`0.0.0.0:18000->8000`） | 改用 **19000** |
| 浏览器打开 `http://localhost:19000/` 返回 404 | llama-server 根路径无页面，**仅提供 API** | 正常；API 路径为 `/v1/chat/completions` |
| 需要聊天网页 | llama.cpp 仓库无预编译 `public/`，Open WebUI Docker 拉取失败 | 自行构建官方 UI（见下） |

**API 测试**（不要用浏览器根路径）：

```bash
curl http://127.0.0.1:19000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-q5km",
    "messages": [{"role":"user","content":"你好，一句话介绍告警管理。"}],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

**健康检查**：

```bash
curl -s http://127.0.0.1:19000/health   # {"status":"ok"}
```

### thinking 模式与 `content` 为空

开 `--reasoning auto` 时，响应分为：

```json
{
  "message": {
    "reasoning_content": "思考过程...",
    "content": "最终答案..."
  },
  "finish_reason": "stop" | "length"
}
```

| 现象 | 原因 | 处理 |
|------|------|------|
| `"content": ""`，`finish_reason: "length"` | `max_tokens` 太小，全被 thinking 吃掉 | 开 thinking 时 `max_tokens` 至少 **1024**，复杂任务 **2048+** |
| Pipeline 批处理想省 token | thinking 拖慢且占上下文 | 可 `--reasoning off`（`thinking = 0`），但用户要求保留 thinking |

**llama-server 相关参数**（`--help`）：

- `--reasoning [on|off|auto]`：控制 thinking（默认 auto）
- `--reasoning-budget N`：限制思考 token（`-1` 不限，`0` 立即结束）
- `--reasoning-format`：thinking 输出格式

### Context size exceeded / 服务断开（Pipeline 联调核心问题）

**不是 Python 逻辑错**，而是 llama-server 在处理请求时 slot 上下文不够或 KV cache 争用。

| 配置 | 每 slot 实际上下文 | 结果 |
|------|-------------------|------|
| `-c 8192`，默认 `n_parallel=4` | ~2048 tokens/slot | ❌ 大 chunk + thinking → `Context size has been exceeded`（HTTP 500） |
| `-c 16384 -np 2` | 8192 tokens/slot | ⚠️ 仍可能断；多 slot 切分总上下文 |
| **`-c 32768 -np 1`** | **32768 tokens/slot** | ✅ 稳定（thinking 模式推荐） |

**错误表现**：

- Pipeline 日志：`API request failed with status 404: {"error":{"message":"File Not Found"...}}`
  - 多数是 **llama-server 已崩溃或 slot 不可用**，客户端收到 404/断连，并非 URL 写错
- 直接 curl 大表格 chunk：`HTTP 500`，`"Context size has been exceeded."`

**其他踩坑**：

- 旧 llama-server 进程（`-c 8192`）未杀干净，与新进程**双占 GPU 显存** → `kill` 旧 PID，只保留一个实例
- `DF_API_URL` 必须是**完整路径**：`http://127.0.0.1:19000/v1/chat/completions`（不能只写到 `/v1`）

### 聊天网页 UI 构建

llama.cpp 无内置 `tools/server/public`，需编译 Svelte UI：

```bash
# Node 20.18 不满足 @sveltejs/vite-plugin-svelte@6 要求，使用 Node 22
export PATH=/sdd/tools/node-v22.15.0-linux-x64/bin:$PATH
cd /sdd/models/llama.cpp/tools/ui
npm config set registry https://registry.npmmirror.com
npm install && npm run build
# 产物：tools/ui/dist/
```

启动时加 `--path /sdd/models/llama.cpp/tools/ui/dist`，浏览器访问 `http://localhost:19000/` 即有聊天页。

Open WebUI（Docker）因 `ghcr.io` / daocloud 镜像拉取失败**未采用**。

### 推荐稳定启动命令（当前生产配置）

```bash
mkdir -p /sdd/models/Qwen3.6-27B-GGUF/logs
pkill -f 'llama-server.*Qwen3.6-27B-Q5_K_M' 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0,1 nohup /sdd/models/llama.cpp/build/bin/llama-server \
  -m /sdd/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q5_K_M.gguf \
  -ngl 99 \
  -c 32768 \
  -np 1 \
  --host 0.0.0.0 \
  --port 19000 \
  --jinja \
  --reasoning auto \
  --reasoning-budget 1024 \
  --path /sdd/models/llama.cpp/tools/ui/dist \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  > /sdd/models/Qwen3.6-27B-GGUF/logs/llama_server_19000.log 2>&1 &

echo $! > /sdd/models/Qwen3.6-27B-GGUF/logs/llama_server.pid
tail -f /sdd/models/Qwen3.6-27B-GGUF/logs/llama_server_19000.log
# 期望：chat template, thinking = 1
#       server is listening on http://0.0.0.0:19000
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `-c 32768 -np 1` | 单 slot 32K | thinking 模式**不要** `-np 2`，否则上下文被切分 |
| `--reasoning auto` | thinking=1 | 保留思考，与 cloud qwen3.6-plus 行为一致 |
| `--reasoning-budget 1024` | 限制思考 token | 单条从 ~70s 降到 ~31–47s，减少占满 context |
| 双卡 | `CUDA_VISIBLE_DEVICES=0,1` | 模型 ~20GB 分摊到两张 3090 |

**监控**：

```bash
curl -s http://127.0.0.1:19000/health
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
tail -f /sdd/models/Qwen3.6-27B-GGUF/logs/llama_server_19000.log
```

---

## 本地 Pipeline 试跑（dataflow 3 chunks）

### 环境变量

```bash
export TECHDOC_INPUT="/home/wugk/finetune/finetune/data/故障管理/text_chunks_20260113_150815_dataflow.json"
export QA_CACHE_DIR="/home/wugk/finetune/finetune/data/故障管理/qa_pipeline_cache_local"
export QA_FILE_PREFIX="fault_mgmt_local_qa"
export DF_API_URL="http://127.0.0.1:19000/v1/chat/completions"
export DF_API_KEY="local"
export DF_MODEL_NAME="Qwen3.6-27B-Q5_K_M.gguf"
export DF_MAX_WORKERS=1

cd /home/wugk/.cursor/plans
nohup bash 脚本03_前台运行_故障管理数据集.sh > /tmp/techdoc_local_test.log 2>&1 &
```

### 联调时间线（问题 → 解决）

| 阶段 | 现象 | 根因 | 解决 |
|------|------|------|------|
| 1 | `DF_API_URL=.../v1` → 404 | Pipeline 用 `APILLMServing_request` 直接 POST 完整 URL | 改为 `.../v1/chat/completions` |
| 2 | 部分 chunk 404 / parse failed | `-c 8192` + 多 slot，context 不够 | 升到 `-c 16384 -np 2`，仍不稳 |
| 3 | `Context size has been exceeded` | 大表格 chunk + thinking 超 slot | **`-c 32768 -np 1`** |
| 4 | 速度极慢（~70s/条） | thinking 无 budget | **`--reasoning-budget 1024`** |
| 5 | 双 llama-server 占显存 | 重启未杀旧进程 | `pkill -f 'llama-server.*Qwen3.6'` |
| 6 | 清缓存重跑 | 失败 step 留下脏缓存 | `rm qa_pipeline_cache_local/*.json` 后重启 |

### 当前试跑进度（2026-05-24 23:20+）

- 输入：3 chunks → 约 12 条初始 QA
- 缓存目录：`qa_pipeline_cache_local/`，已写入 **step1–step17**
- Step 15 QuestionRefiner：`fact-closure: ok=11 violation=0 missing=0`
- Step 8 Rubric 进行中（11 条，~50s/条）
- llama-server：`health ok`，`thinking = 1`

**日志**：

```bash
tail -f /tmp/techdoc_local_test.log
ls /home/wugk/finetune/finetune/data/故障管理/qa_pipeline_cache_local/fault_mgmt_local_qa_step*.json
```

本地 3-chunk 全流程预估 **30–60 分钟**（21 step × 多次 LLM 调用 × ~40–55s/次）。

---

## 相关文件索引

| 文件 / 路径 | 说明 |
|------|------|
| `文档12_故障管理数据集_QA管道运行手册.plan.md` | 运行手册（启动/监控/导出） |
| `脚本03_前台运行_故障管理数据集.sh` | 前台入口 |
| `脚本08_后台运行_故障管理_阿里云API.sh` | 后台入口（推荐） |
| `脚本07_QA管道主程序_Python入口.py` | Python 主逻辑 |
| `脚本01_下载Qwen3.6-27B-Q5KM量化模型.sh` | ModelScope 下载 Qwen3.6-27B-Q5_K_M |
| `文档10_第5轮_代码修复记录.plan.md` | Round 5 改动 |
| `文档11_第6轮_thinking标签解析与本地llama部署.plan.md` | 本文件 |
| `/sdd/models/Qwen3.6-27B-GGUF/` | 本地 GGUF 模型与日志 |
| `/sdd/models/llama.cpp/` | llama.cpp 源码与 `build/bin/llama-server` |
| `/sdd/models/llama.cpp/tools/ui/dist/` | 聊天 Web UI 静态资源 |
| `/sdd/tools/cmake-3.28.3/` | 编译用 cmake |
| `/sdd/tools/node-v22.15.0-linux-x64/` | UI 构建用 Node 22 |
| `qa_pipeline_cache_local/` | 本地 3-chunk 试跑缓存（前缀 `fault_mgmt_local_qa`） |
| `/tmp/techdoc_local_test.log` | 本地 Pipeline 运行日志 |
