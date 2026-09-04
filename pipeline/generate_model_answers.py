#!/usr/bin/env python3
"""Generate model answers for the 1010-item QA test set (OpenAI-compatible API).

Designed to run on a remote server where a fine-tuned model is deployed
(vLLM, llama.cpp server, SGLang, Ollama, etc.).

Output format matches data/eval_qwen3_8b_1010/answers.jsonl and can be scored
later with eval_qwen3_1010_rubric.py --phase score.

Examples
--------
# vLLM on the same machine
python3 generate_model_answers.py \\
  --qa ../data/combined_three_documents_qa_minimal_1010_v1.json \\
  --out-dir ./eval_my_model \\
  --base-url http://127.0.0.1:8000/v1 \\
  --model my-finetuned-model \\
  --workers 4

# llama.cpp server (Qwen thinking off)
python3 generate_model_answers.py \\
  --base-url http://127.0.0.1:19000/v1 \\
  --model Qwen3-8B-Instruct.gguf \\
  --disable-thinking \\
  --workers 1

# smoke test
python3 generate_model_answers.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = ROOT / "data/combined_three_documents_qa_minimal_1010_v1.json"
DEFAULT_OUT_DIR = ROOT / "data/eval_custom_model"

ANSWER_SYSTEM = """你是一名资深存储与数据库运维工程师。
请根据问题给出专业、可执行、结构清晰的技术回答。
如涉及命令、阈值、端口、路径或操作顺序，请尽量给出完整细节。
不要编造文档中未出现的具体参数；若信息不足请明确说明。"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key or "local", base_url=base_url)


def build_extra_body(disable_thinking: bool, extra_body_json: str) -> dict:
    extra: dict = {}
    if extra_body_json:
        extra.update(json.loads(extra_body_json))
    if disable_thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        extra.setdefault("enable_thinking", False)
    return extra


def call_model_answer(
    client: OpenAI,
    model: str,
    question: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning.strip():
            text = reasoning.strip()
    return text


def write_run_meta(out_dir: Path, meta: dict) -> None:
    meta_path = out_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run_answers(
    client: OpenAI,
    items: list[dict],
    model: str,
    out_writer: JsonlWriter,
    workers: int,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    extra_body: dict,
    retries: int,
) -> None:
    done = load_done_ids(out_writer.path)
    pending = [x for x in items if x["id"] not in done]
    if not pending:
        print("✅ 所有题目已有答案，跳过。")
        return

    def one(item: dict) -> dict:
        last_err = ""
        for attempt in range(retries + 1):
            try:
                answer = call_model_answer(
                    client,
                    model,
                    item["question"],
                    system_prompt,
                    max_tokens,
                    temperature,
                    extra_body,
                )
                rec = {
                    "id": item["id"],
                    "rubric_id": item["rubric_id"],
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "model_answer": answer,
                    "model": model,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                out_writer.append(rec)
                return rec
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
        rec = {
            "id": item["id"],
            "rubric_id": item["rubric_id"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "model_answer": "",
            "model": model,
            "error": last_err,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_writer.append(rec)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, item) for item in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="generate answers"):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers for 1010 QA items via OpenAI-compatible API")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA, help="QA test set JSON path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--output", type=Path, default=None, help="Override answers output path (default: OUT_DIR/answers.jsonl)")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "qwen3-8b"), help="Model name served by the API")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent request workers")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Only run first N items (0 = all)")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable Qwen thinking (llama.cpp / some vLLM)")
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not send system prompt")
    parser.add_argument(
        "--extra-body-json",
        default="",
        help='Extra JSON merged into request extra_body, e.g. \'{"top_p":0.8}\'',
    )
    args = parser.parse_args()

    if not args.qa.exists():
        raise SystemExit(f"QA 文件不存在: {args.qa}")

    items = load_json(args.qa)
    if args.limit > 0:
        items = items[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.output or (args.out_dir / "answers.jsonl")
    system_prompt = "" if args.no_system_prompt else ANSWER_SYSTEM
    extra_body = build_extra_body(args.disable_thinking, args.extra_body_json)

    write_run_meta(
        args.out_dir,
        {
            "qa": str(args.qa),
            "answers": str(answers_path),
            "model": args.model,
            "base_url": args.base_url,
            "workers": args.workers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "disable_thinking": args.disable_thinking,
            "total_items": len(items),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    client = make_client(args.api_key, args.base_url)
    print(f"📂 QA: {args.qa} ({len(items)} items)")
    print(f"🤖 Model: {args.model}")
    print(f"🔗 API: {args.base_url}")
    print(f"💾 Output: {answers_path}")

    run_answers(
        client,
        items,
        args.model,
        JsonlWriter(answers_path),
        args.workers,
        args.max_tokens,
        args.temperature,
        system_prompt,
        extra_body,
        args.retries,
    )

    done_count = len(load_done_ids(answers_path))
    print(f"\n✅ 完成，共 {done_count} 条答案写入 {answers_path}")


if __name__ == "__main__":
    main()
