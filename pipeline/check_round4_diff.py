"""Round 4 验证脚本：对比 round 3 与 round 4 final step 输出。

跑完 V2 pipeline 后运行：
    python finetune/test/check_round4_diff.py [--round3 step16.json] [--round4 step17.json]

默认对比：
- round3 = finetune/test/cache_local_qa/qa_pipeline_v2_step_step16.json
- round4 = 最新的 qa_pipeline_v2_step_stepN.json

检查口径（plan 第四轮规定）：
1. 必须 0 条：包含 事实锚 / Target Answer / OUTOFSCOPE / 参考内容明确指出 / 原文指出 / 应检查并重新确认 等关键词的答案
2. 必须 0 条：refined_reverse_question 出现 answer/context 未提及的字段名 / 路径 / 占位值（人工抽样为主，本脚本只做关键词扫描）
3. MetaFilter 阈值维持 3.5（不变）；Rubric 维持 SC-1~SC-4（不变）
4. 预期总数从 82 略降，主要来自 表格 chunk 入口拦截 + grounding OUTOFSCOPE drop + question refine 闭包回退
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache_local_qa"
DEFAULT_PREFIX = "qa_pipeline_v2_step_step"
DEFAULT_ROUND3 = DEFAULT_CACHE_DIR / "qa_pipeline_v2_step_step16.json"


FORBIDDEN_KEYWORDS_ANSWER = [
    "事实锚",
    "Target Answer",
    "OUTOFSCOPE",
    "参考内容明确指出",
    "原文指出",
    "应检查并重新确认",
    "事实集封闭",
    "Fact-Set Closure",
    "Fact Baseline",
    "事实基准",
    "精炼答案",
    "可用信息集",
    "批评反馈",
    "Critique",
    "根据参考内容",
    "参考内容显示",
    "原文中",
    "根据原文",
    "按照原文",
    "根据文档",
    "根据流程图",
    "根据表格",
]

ANSWER_FIELDS = [
    "refined_answer",
    "clean_answer",
    "initial_answer",
]

QUESTION_FIELDS = [
    "refined_reverse_question",
    "rough_question",
]


def latest_step_file(cache_dir: Path, prefix: str) -> Path | None:
    if not cache_dir.is_dir():
        return None
    candidates = []
    pat = re.compile(rf"{re.escape(prefix)}(\d+)\.json$")
    for p in cache_dir.iterdir():
        m = pat.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []


def scan_forbidden(records: list[dict], fields: list[str], keywords: list[str]) -> dict:
    """Return mapping keyword -> [ (idx, field, snippet) ]."""
    hits: dict[str, list[tuple[int, str, str]]] = {kw: [] for kw in keywords}
    for idx, rec in enumerate(records):
        for field in fields:
            text = rec.get(field)
            if not isinstance(text, str) or not text:
                continue
            for kw in keywords:
                if kw in text:
                    pos = text.find(kw)
                    snippet = text[max(0, pos - 20): pos + len(kw) + 20]
                    hits[kw].append((idx, field, snippet))
    return hits


def report_hits(label: str, hits: dict, max_examples: int = 3) -> int:
    total = sum(len(v) for v in hits.values())
    print(f"\n=== {label} 关键词命中（共 {total} 处） ===")
    for kw, occs in hits.items():
        if not occs:
            continue
        print(f"  - {kw!r}: {len(occs)} 处")
        for idx, field, snippet in occs[:max_examples]:
            print(f"      [#{idx} {field}] ...{snippet}...")
        if len(occs) > max_examples:
            print(f"      ... 还有 {len(occs) - max_examples} 处")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round3", default=str(DEFAULT_ROUND3))
    parser.add_argument(
        "--round4",
        default=None,
        help="round 4 最终 step 文件路径；不传则自动选最新 stepN.json",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    round3_path = Path(args.round3)
    if args.round4:
        round4_path = Path(args.round4)
    else:
        latest = latest_step_file(cache_dir, DEFAULT_PREFIX)
        if latest is None:
            print(f"[error] 未在 {cache_dir} 找到 {DEFAULT_PREFIX}*.json", file=sys.stderr)
            sys.exit(1)
        round4_path = latest

    print(f"round 3 file = {round3_path}")
    print(f"round 4 file = {round4_path}")

    r3 = load_records(round3_path)
    r4 = load_records(round4_path)
    print(f"\n样本量：round3={len(r3)} round4={len(r4)} (delta={len(r4) - len(r3)})")

    hits3_ans = scan_forbidden(r3, ANSWER_FIELDS, FORBIDDEN_KEYWORDS_ANSWER)
    hits4_ans = scan_forbidden(r4, ANSWER_FIELDS, FORBIDDEN_KEYWORDS_ANSWER)
    total3 = report_hits("ROUND 3 [answer fields]", hits3_ans)
    total4 = report_hits("ROUND 4 [answer fields]", hits4_ans)

    hits3_q = scan_forbidden(r3, QUESTION_FIELDS, FORBIDDEN_KEYWORDS_ANSWER)
    hits4_q = scan_forbidden(r4, QUESTION_FIELDS, FORBIDDEN_KEYWORDS_ANSWER)
    total3_q = report_hits("ROUND 3 [question fields]", hits3_q)
    total4_q = report_hits("ROUND 4 [question fields]", hits4_q)

    print("\n=== 对比小结 ===")
    print(
        f"  answer  hits  round3 -> round4: {total3} -> {total4} "
        f"(降幅 {total3 - total4})"
    )
    print(
        f"  question hits round3 -> round4: {total3_q} -> {total4_q} "
        f"(降幅 {total3_q - total4_q})"
    )
    if total4 == 0 and total4_q == 0:
        print("  [PASS] 关键词全部清零，符合 round 4 验证口径。")
        sys.exit(0)
    else:
        print("  [FAIL] 仍有关键词残留，请人工抽样查看上方命中行。")
        sys.exit(2)


if __name__ == "__main__":
    main()
