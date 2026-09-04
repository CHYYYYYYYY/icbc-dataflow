#!/usr/bin/env python3
"""Generate qwen3-8b answers for 1010 QA items and score with rubric judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = ROOT / "data/combined_three_documents_qa_minimal_1010_v1.json"
DEFAULT_RUBRIC = ROOT / "data/combined_three_documents_multihop_rubric_1010_v1.json"
DEFAULT_OUT_DIR = ROOT / "data/eval_qwen3_8b_1010"
DEFAULT_FINAL = ROOT / "data/combined_three_documents_qwen3-8b_eval_1010_v1.json"

ANSWER_SYSTEM = """你是一名资深存储与数据库运维工程师。
请根据问题给出专业、可执行、结构清晰的技术回答。
如涉及命令、阈值、端口、路径或操作顺序，请尽量给出完整细节。
不要编造文档中未出现的具体参数；若信息不足请明确说明。"""

DOMAIN_POLICY = {
    "OceanStor Dorado故障处理": {"pass_threshold": 75, "hard_fail_cap": 49},
    "华为告警与Trap处理": {"pass_threshold": 70, "hard_fail_cap": 40},
    "GaussDB 24.1.30故障管理": {"pass_threshold": 70, "hard_fail_cap": 40},
}


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
    return OpenAI(api_key=api_key, base_url=base_url)


def call_qwen3_answer(client: OpenAI, model: str, question: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    return (resp.choices[0].message.content or "").strip()


def build_judge_prompt(
    rubric_meta: dict,
    rubric: dict,
    question: str,
    reference_answer: str,
    model_answer: str,
) -> str:
    criteria_text = json.dumps(rubric.get("criteria", []), ensure_ascii=False, indent=2)
    hard_fails = json.dumps(rubric.get("hard_fail_conditions", []), ensure_ascii=False, indent=2)
    global_req = json.dumps(rubric_meta.get("global_requirements", []), ensure_ascii=False, indent=2)
    schema = json.dumps(rubric_meta.get("evaluator_output_schema", {}), ensure_ascii=False, indent=2)
    eval_prompt = rubric_meta.get("evaluation_prompt", "")

    return f"""{eval_prompt}

【全局要求】
{global_req}

【题目信息】
item_id: {rubric.get("item_id")}
source_domain: {rubric.get("source_domain")}

【问题】
{question}

【参考答案（仅作评分依据，不得要求候选答案逐字匹配）】
{reference_answer}

【候选答案（待评分）】
{model_answer}

【本题Rubric criteria】
{criteria_text}

【本题hard_fail_conditions】
{hard_fails}

请按以下步骤评分：
1. 先检查是否触发任一 hard_fail_conditions；
2. 再按每个 criterion 的 anchors 独立给分；
3. total_score 为各 criterion 得分之和，最高 {rubric.get("max_score", 100)}；
4. 若触发 hard_fail，total_score 不得超过对应域 hard_fail_cap；
5. verdict 只能是：优秀、合格、不足、失败。

输出必须符合以下 JSON schema（只输出 JSON，不要 markdown 代码块）：
{schema}
"""


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def apply_hard_fail_cap(score: dict, source_domain: str) -> dict:
    policy = DOMAIN_POLICY.get(source_domain, {"pass_threshold": 70, "hard_fail_cap": 40})
    cap = policy["hard_fail_cap"]
    if score.get("hard_fail_triggered") and score.get("total_score", 0) > cap:
        score["total_score"] = cap
        score["hard_fail_cap_applied"] = cap
    score["pass_threshold"] = policy["pass_threshold"]
    score["passed"] = score.get("total_score", 0) >= policy["pass_threshold"] and not score.get(
        "hard_fail_triggered", False
    )
    return score


def call_judge(
    client: OpenAI,
    model: str,
    rubric_meta: dict,
    rubric: dict,
    question: str,
    reference_answer: str,
    model_answer: str,
    max_tokens: int,
) -> dict:
    prompt = build_judge_prompt(rubric_meta, rubric, question, reference_answer, model_answer)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    parsed = parse_judge_json(resp.choices[0].message.content or "{}")
    parsed["item_id"] = rubric.get("item_id")
    parsed["source_domain"] = rubric.get("source_domain")
    return apply_hard_fail_cap(parsed, rubric.get("source_domain", ""))


def run_answers(
    client: OpenAI,
    items: list[dict],
    model: str,
    out_writer: JsonlWriter,
    workers: int,
    max_tokens: int,
    retries: int,
) -> None:
    done = load_done_ids(out_writer.path)

    def one(item: dict) -> dict:
        if item["id"] in done:
            return {}
        last_err = ""
        for attempt in range(retries + 1):
            try:
                answer = call_qwen3_answer(client, model, item["question"], max_tokens)
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
        out_writer.append(
            {
                "id": item["id"],
                "rubric_id": item["rubric_id"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "model_answer": "",
                "model": model,
                "error": last_err,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return {}

    pending = [x for x in items if x["id"] not in done]
    if not pending:
        print("✅ 答案阶段已全部完成，跳过。")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, item) for item in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="qwen3-8b answers"):
            pass


def load_answers_map(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mapping[rec["id"]] = rec
    return mapping


def run_scores(
    client: OpenAI,
    items: list[dict],
    rubric_by_id: dict[str, dict],
    rubric_meta: dict,
    answers_map: dict[str, dict],
    model: str,
    out_writer: JsonlWriter,
    workers: int,
    max_tokens: int,
    retries: int,
) -> None:
    done = load_done_ids(out_writer.path)

    def one(item: dict) -> dict:
        if item["id"] in done:
            return {}
        ans = answers_map.get(item["id"], {})
        model_answer = ans.get("model_answer", "")
        rubric = rubric_by_id.get(item["rubric_id"])
        if not rubric:
            rec = {
                "id": item["id"],
                "rubric_id": item["rubric_id"],
                "error": f"missing rubric {item['rubric_id']}",
            }
            out_writer.append(rec)
            return rec
        if not model_answer:
            rec = {
                "id": item["id"],
                "rubric_id": item["rubric_id"],
                "error": "empty model_answer",
            }
            out_writer.append(rec)
            return rec

        last_err = ""
        for attempt in range(retries + 1):
            try:
                score = call_judge(
                    client,
                    model,
                    rubric_meta,
                    rubric,
                    item["question"],
                    item["reference_answer"],
                    model_answer,
                    max_tokens,
                )
                rec = {
                    "id": item["id"],
                    "rubric_id": item["rubric_id"],
                    "model_answer": model_answer,
                    "score": score,
                    "judge_model": model,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                out_writer.append(rec)
                return rec
            except Exception as e:
                last_err = str(e)
                time.sleep(2.0 * (attempt + 1))
        out_writer.append(
            {
                "id": item["id"],
                "rubric_id": item["rubric_id"],
                "model_answer": model_answer,
                "error": last_err,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return {}

    pending = [x for x in items if x["id"] not in done]
    if not pending:
        print("✅ 评分阶段已全部完成，跳过。")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, item) for item in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="rubric judge"):
            pass


def merge_final(
    items: list[dict],
    answers_map: dict[str, dict],
    scores_map: dict[str, dict],
    answer_model: str,
    judge_model: str,
    final_path: Path,
) -> dict:
    results = []
    total_scores = []
    passed = 0
    hard_fails = 0
    for item in items:
        ans = answers_map.get(item["id"], {})
        sc = scores_map.get(item["id"], {})
        score_obj = sc.get("score", {})
        if score_obj.get("total_score") is not None:
            total_scores.append(score_obj["total_score"])
            if score_obj.get("passed"):
                passed += 1
            if score_obj.get("hard_fail_triggered"):
                hard_fails += 1
        results.append(
            {
                "id": item["id"],
                "rubric_id": item["rubric_id"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "model_answer": ans.get("model_answer", ""),
                "answer_error": ans.get("error"),
                "score": score_obj if score_obj else None,
                "score_error": sc.get("error"),
            }
        )

    summary = {
        "total_items": len(items),
        "answered": sum(1 for r in results if r["model_answer"]),
        "scored": len(total_scores),
        "avg_score": round(sum(total_scores) / len(total_scores), 2) if total_scores else None,
        "pass_count": passed,
        "pass_rate": round(passed / len(total_scores), 4) if total_scores else None,
        "hard_fail_count": hard_fails,
        "answer_model": answer_model,
        "judge_model": judge_model,
    }

    payload = {
        "meta": {
            "testset": "combined_three_documents_qa_minimal_1010_v1",
            "rubric_set": "combined_three_documents_multihop_rubric_1010_v1",
            "answer_model": answer_model,
            "judge_model": judge_model,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "results": results,
    }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--phase", choices=["all", "answer", "score", "merge"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--answer-model", default="qwen3-8b")
    parser.add_argument("--judge-model", default="qwen-plus")
    parser.add_argument("--answer-workers", type=int, default=6)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--answer-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=2500)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPENAI_BASE_URL",
            os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ),
    )
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", "local"))
    args = parser.parse_args()

    api_key = args.api_key

    items = load_json(args.qa)
    if args.limit > 0:
        items = items[: args.limit]

    rubric_meta = load_json(args.rubric)
    rubric_by_id = {r["rubric_id"]: r for r in rubric_meta["rubrics"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.out_dir / "answers.jsonl"
    scores_path = args.out_dir / "scored.jsonl"
    client = make_client(api_key or "local", args.base_url)

    if args.phase in ("all", "answer"):
        run_answers(
            client,
            items,
            args.answer_model,
            JsonlWriter(answers_path),
            args.answer_workers,
            args.answer_max_tokens,
            args.retries,
        )

    if args.phase in ("all", "score"):
        answers_map = load_answers_map(answers_path)
        run_scores(
            client,
            items,
            rubric_by_id,
            rubric_meta,
            answers_map,
            args.judge_model,
            JsonlWriter(scores_path),
            args.judge_workers,
            args.judge_max_tokens,
            args.retries,
        )

    if args.phase in ("all", "merge"):
        answers_map = load_answers_map(answers_path)
        scores_map = load_answers_map(scores_path)
        summary = merge_final(
            items,
            answers_map,
            scores_map,
            args.answer_model,
            args.judge_model,
            args.final,
        )
        print("\n" + "=" * 60)
        print("📊 汇总")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"💾 最终文件: {args.final}")
        print("=" * 60)


if __name__ == "__main__":
    main()
