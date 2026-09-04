# LLaMA-Factory 微调模型 · 1010 题答题

在微调服务器上加载权重 → 调 API 生成回答 → 回传 `answers.jsonl`。

---

## 1. 拷贝文件到微调服务器

```bash
# 评测机上
scp test/generate_model_answers.py \
    test/run_lf_eval_1010.sh \
    test/config/qwen3_lora_infer.yaml \
    test/requirements_eval.txt \
    data/combined_three_documents_qa_minimal_1010_v1.json \
    user@微调服务器:/data/eval/
```

或在微调服务器上直接新建下面两个文件（内容见文末附录）。

---

## 2. 安装 Python 依赖

`generate_model_answers.py` 只需要两个包：

| 包 | 作用 |
|----|------|
| `openai` | 调用 LLaMA-Factory 的 OpenAI 兼容 API |
| `tqdm` | 显示答题进度条 |

```bash
pip install openai tqdm
```

若已在 LLaMA-Factory 的 conda/venv 里，可先检查是否已装：

```bash
python3 -c "import openai, tqdm; print('ok')"
```

也可新建 `requirements.txt`（内容见附录 C）后安装：

```bash
pip install -r requirements.txt
```

---

## 3. 启动 LLaMA-Factory API

编辑 `qwen3_lora_infer.yaml`（路径改成你的），然后启动：

```bash
cd /path/to/LLaMA-Factory
API_PORT=8000 llamafactory-cli api /data/eval/qwen3_lora_infer.yaml
```

若已 merge 全量权重，yaml 里只保留 `model_name_or_path: /path/to/output/qwen3_lora_sft`，删掉 `adapter_name_or_path`。

确认 API 正常：

```bash
curl http://127.0.0.1:8000/v1/models
```

记下返回的模型名（后面 `MODEL_NAME` 要用）。

---

## 4. 生成回答

先测 5 题：

```bash
cd /data/eval
MODEL_NAME=你的模型名 LIMIT=5 bash run_lf_eval_1010.sh
```

`run_lf_eval_1010.sh` 默认读同目录下的 QA 文件；若目录不同，可直接：

```bash
python3 generate_model_answers.py \
  --qa combined_three_documents_qa_minimal_1010_v1.json \
  --out-dir ./eval_out \
  --base-url http://127.0.0.1:8000/v1 \
  --model 你的模型名 \
  --limit 5
```

没问题后跑全量 1010 题：

```bash
cd /data/eval
nohup env MODEL_NAME=你的模型名 OUT_DIR=./eval_out bash run_lf_eval_1010.sh \
  > ./eval_out/run.log 2>&1 &

tail -f ./eval_out/run.log
```

输出：`eval_out/answers.jsonl`（1010 行）。中断后重跑同一命令会自动续跑。

---

## 5. 回传评测机

```bash
scp /data/eval/eval_out/answers.jsonl \
    user@评测机:/home/wugk/finetune/finetune/data/eval_my_model/
```

回传后在评测机用 rubric 打分即可。

---

## 附录 A：`qwen3_lora_infer.yaml`

保存为 `/data/eval/qwen3_lora_infer.yaml`：

```yaml
model_name_or_path: /path/to/Qwen3-8B
adapter_name_or_path: /path/to/LLaMA-Factory/saves/qwen3-8b/lora/sft
template: qwen3
trust_remote_code: true
```

---

## 附录 B：`run_lf_eval_1010.sh`

保存为 `/data/eval/run_lf_eval_1010.sh`，并 `chmod +x run_lf_eval_1010.sh`：

```bash
#!/usr/bin/env bash
# 用法: MODEL_NAME=xxx bash run_lf_eval_1010.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "${DIR}/generate_model_answers.py" \
  --qa "${DIR}/combined_three_documents_qa_minimal_1010_v1.json" \
  --out-dir "${OUT_DIR:-${DIR}/eval_out}" \
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model "${MODEL_NAME:?请设置 MODEL_NAME}" \
  --workers "${WORKERS:-2}" \
  ${LIMIT:+--limit "${LIMIT}"}
```

用法示例：

```bash
# 冒烟 5 题
MODEL_NAME=你的模型名 LIMIT=5 bash run_lf_eval_1010.sh

# 全量 1010 题
MODEL_NAME=你的模型名 OUT_DIR=./eval_out bash run_lf_eval_1010.sh
```

---

## 附录 C：`requirements.txt`

保存为 `/data/eval/requirements.txt`：

```text
openai>=1.0.0
tqdm>=4.0.0
```

安装：

```bash
pip install -r requirements.txt
```
