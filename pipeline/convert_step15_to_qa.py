"""从 qa_pipeline_v2_step_step{N}.json 导出 QA 对，并可选审计 AnswerRefiner 尾部启发式。

用法：
    python convert_step15_to_qa.py           # 默认 step15（兼容旧用法）
    python convert_step15_to_qa.py 17        # 指定 step 编号
    python convert_step15_to_qa.py --step 17
    python convert_step15_to_qa.py --latest  # 自动取 cache 里最新 step
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

CACHE_DIR = Path(os.environ.get("QA_CACHE_DIR", str(Path(__file__).resolve().parent / "cache_local_qa")))
FILE_PREFIX = os.environ.get("QA_FILE_PREFIX", "qa_pipeline_v2_step")
DEFAULT_STEP = 15


def _paths_for_step(step: int) -> tuple[Path, Path, Path, Path]:
    source = CACHE_DIR / f"{FILE_PREFIX}_step{step}.json"
    target = CACHE_DIR / f"{FILE_PREFIX}_step{step}_qa.json"
    pairs_json = CACHE_DIR / f"{FILE_PREFIX}_step{step}_qa_pairs.json"
    pairs_jsonl = CACHE_DIR / f"{FILE_PREFIX}_step{step}_qa_pairs.jsonl"
    return source, target, pairs_json, pairs_jsonl


def _resolve_step_from_argv(argv: list[str]) -> int:
    if "--latest" in argv:
        import re
        pat = re.compile(rf"^{re.escape(FILE_PREFIX)}_step(\d+)\.json$")
        candidates = []
        for p in CACHE_DIR.glob(f"{FILE_PREFIX}_step*.json"):
            if pat.match(p.name):
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(f"未在 {CACHE_DIR} 找到 {FILE_PREFIX}_step*.json")
        latest = max(candidates, key=lambda p: int(pat.match(p.name).group(1)))
        return int(pat.match(latest.name).group(1))

    for i, arg in enumerate(argv):
        if arg in ("--step", "-s") and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.isdigit():
            return int(arg)
    return DEFAULT_STEP

_ROOT = Path(__file__).resolve().parents[2]
_MLF = _ROOT / "DataFlow" / "dataflow" / "utils" / "meta_label_filters.py"


def _load_mlf():
    spec = importlib.util.spec_from_file_location("_mlf_standalone", _MLF)
    m = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(m)
    return m


def pick_first_non_empty(record, keys):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_export_question(record) -> str:
    """优先取通过校验的问题；refined 无效时回退 rough_question。"""
    try:
        from dataflow.utils.meta_label_filters import is_valid_export_question
    except ImportError:
        is_valid_export_question = None

    for key in (
        "refined_reverse_question",
        "reverse_question",
        "rough_question",
        "instruction",
    ):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        q = value.strip()
        if is_valid_export_question is None or is_valid_export_question(q):
            return q
    return ""


def _parse_options(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except json.JSONDecodeError:
            return [s]
    return []


def _format_mc_fill_instruction_output(row: dict, question: str, answer: str) -> tuple[str, str, dict]:
    """Return (instruction, output, objective_extra) for objective types."""
    extra: dict = {}
    qtype = (row.get("distill_question_type") or "").strip()

    if qtype == "MultipleChoice":
        opts = _parse_options(row.get("distill_options"))
        letter = (row.get("distill_correct_answer") or "").strip()
        extra["distill_options"] = opts
        extra["distill_correct_answer"] = letter or None
        if opts:
            opt_block = "\n".join(opts)
            instruction = (
                f"{question}\n\n请从下列选项中选择正确的一项：\n{opt_block}"
            )
        else:
            instruction = question
        if letter:
            output = f"{letter}\n\n{answer}".strip()
        else:
            output = answer
        return instruction, output, extra

    if qtype == "FillBlank":
        blank = row.get("distill_blank_answer")
        bstr = blank.strip() if isinstance(blank, str) else ""
        if bstr:
            extra["distill_blank_answer"] = bstr
        # 训练目标用精炼后的整句（已把空填入上下文）；不再在 output 顶部重复贴「多空格答」，
        # 避免与正文重复或「一条列表 + 一段复述」的不自然版式；标答见 blank_answer 字段。
        output = answer.strip()
        return question, output, extra

    return question, answer, extra


def export_qa(data: list, include_context: bool = False) -> tuple[list, list]:
    """Returns (alpaca_rows, rich_pair_rows)."""
    alpaca: list[dict] = []
    rich: list[dict] = []

    for idx, row in enumerate(data):
        question = pick_export_question(row)
        answer = pick_first_non_empty(
            row,
            ["refined_answer", "clean_answer", "initial_answer"],
        )
        context = (row.get("raw_content") or "").strip()
        if not question or not answer:
            continue
        if include_context and not context:
            continue

        instruction, output, obj_extra = _format_mc_fill_instruction_output(row, question, answer)

        try:
            from dataflow.utils.meta_label_filters import (
                is_valid_export_question,
                is_valid_export_output,
                normalize_export_question,
            )

            instruction = normalize_export_question(instruction) or ""
            if not instruction or not is_valid_export_output(output):
                continue
        except ImportError:
            try:
                from clean_sft_train import clean_record as _clean_sft_record

                cleaned = _clean_sft_record({"instruction": instruction, "output": output})
                if cleaned is None:
                    continue
                instruction, output = cleaned["instruction"], cleaned["output"]
            except ImportError:
                pass

        alpaca_row = {
            "instruction": instruction,
            "input": context if include_context else "",
            "output": output,
            "question_type": row.get("distill_question_type") or row.get("question_type"),
        }
        if obj_extra:
            if "distill_options" in obj_extra:
                alpaca_row["options"] = obj_extra["distill_options"]
            if obj_extra.get("distill_correct_answer"):
                alpaca_row["correct_answer"] = obj_extra["distill_correct_answer"]
            if obj_extra.get("distill_blank_answer"):
                alpaca_row["blank_answer"] = obj_extra["distill_blank_answer"]
        alpaca.append(alpaca_row)

        meta: dict = {
            "id": idx,
            "question": instruction,
            "answer": output,
            "raw_content": context if include_context else row.get("raw_content"),
            "distill_question_type": row.get("distill_question_type"),
            "question_type": row.get("question_type"),
            "distill_scenario": row.get("distill_scenario"),
            "rubric_verdict": row.get("rubric_verdict"),
            "MetaScore": row.get("MetaScore"),
        }
        if obj_extra.get("distill_options") is not None:
            meta["options"] = obj_extra.get("distill_options") or []
        if obj_extra.get("distill_correct_answer"):
            meta["correct_answer"] = obj_extra["distill_correct_answer"]
        if obj_extra.get("distill_blank_answer"):
            meta["blank_answer"] = obj_extra["distill_blank_answer"]
        rich.append(meta)

    return alpaca, rich


def audit_refined(data: list, m) -> dict:
    """镜像 AnswerRefiner 尾部检测：quote dump / 评语腔 / 极性（在 strip_evaluator_tone 之后评极性）。"""
    stats = {
        "rows": 0,
        "quote_warn": 0,
        "evaluator_tone_warn": 0,
        "tone_stripped_rows": 0,
        "evaluator_tone_leak_after_strip": 0,
        "polarity_warn": 0,
        "polarity_codes": {},
        "examples": {"quote": [], "polarity": [], "tone_leak": []},
    }

    looks_like_answer_as_quote = m.looks_like_answer_as_quote
    detect_evaluator_tone = m.detect_evaluator_tone
    strip_evaluator_tone = m.strip_evaluator_tone
    detect_question_answer_polarity_issue = m.detect_question_answer_polarity_issue

    for row in data:
        q = pick_export_question(row)
        ans = pick_first_non_empty(row, ["refined_answer", "clean_answer", "initial_answer"])
        ctx = (row.get("raw_content") or "") if isinstance(row.get("raw_content"), str) else ""
        if not q or not ans:
            continue
        stats["rows"] += 1

        if looks_like_answer_as_quote(ans, ctx):
            stats["quote_warn"] += 1
            if len(stats["examples"]["quote"]) < 3:
                stats["examples"]["quote"].append(
                    {"question": q[:100], "answer_prefix": ans[:120]}
                )

        tone_hits_before = detect_evaluator_tone(ans)
        if tone_hits_before:
            stats["evaluator_tone_warn"] += 1
        ans_toned, strip_hits = strip_evaluator_tone(ans)
        if strip_hits and ans_toned != ans:
            stats["tone_stripped_rows"] += 1
        leaks = detect_evaluator_tone(ans_toned)
        if leaks:
            stats["evaluator_tone_leak_after_strip"] += 1
            if len(stats["examples"]["tone_leak"]) < 3:
                stats["examples"]["tone_leak"].append(
                    {"hits": leaks[:5], "answer_prefix": ans_toned[:120]}
                )

        pol = detect_question_answer_polarity_issue(q, ans_toned)
        if pol:
            stats["polarity_warn"] += 1
            stats["polarity_codes"][pol] = stats["polarity_codes"].get(pol, 0) + 1
            if len(stats["examples"]["polarity"]) < 3:
                stats["examples"]["polarity"].append(
                    {"code": pol, "question": q[:100], "answer_prefix": ans_toned[:120]}
                )

    return stats


def main():
    audit = "--no-audit" not in sys.argv
    include_context = "--with-context" in sys.argv
    argv = [a for a in sys.argv[1:] if a not in ("--no-audit", "--with-context")]
    step = _resolve_step_from_argv(argv)
    source, target, pairs_json, pairs_jsonl = _paths_for_step(step)
    if not source.is_file():
        raise FileNotFoundError(f"源文件不存在: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    alpaca, rich = export_qa(data, include_context=include_context)

    suffix = "_ctx" if include_context else ""
    target = target.with_name(target.stem + suffix + target.suffix)
    pairs_json = pairs_json.with_name(pairs_json.stem + suffix + pairs_json.suffix)
    pairs_jsonl = pairs_jsonl.with_name(pairs_jsonl.stem + suffix + pairs_jsonl.suffix)

    target.write_text(json.dumps(alpaca, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs_json.write_text(json.dumps(rich, ensure_ascii=False, indent=2), encoding="utf-8")
    with pairs_jsonl.open("w", encoding="utf-8") as f:
        for item in rich:
            if include_context:
                line = {
                    "instruction": item["question"],
                    "input": item.get("raw_content") or "",
                    "output": item["answer"],
                    "question_type": item.get("question_type") or item.get("distill_question_type"),
                }
            else:
                line = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_type": item.get("distill_question_type"),
                }
            if item.get("options"):
                line["options"] = item["options"]
            if item.get("correct_answer"):
                line["correct_answer"] = item["correct_answer"]
            if item.get("blank_answer"):
                line["blank_answer"] = item["blank_answer"]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"step={step} source={source.name} with_context={include_context}")
    print(f"wrote {len(alpaca)} rows -> {target.name}, {pairs_json.name}, {pairs_jsonl.name}")
    from collections import Counter

    tc = Counter((a.get("question_type") or "unknown") for a in alpaca)
    print("question_type breakdown:", dict(sorted(tc.items(), key=lambda x: (-x[1], x[0]))))

    if audit:
        m = _load_mlf()
        stats = audit_refined(data, m)
        print("\n[Refine heuristics audit on refined_answer + raw_content]")
        for k in (
            "rows",
            "quote_warn",
            "evaluator_tone_warn",
            "tone_stripped_rows",
            "evaluator_tone_leak_after_strip",
            "polarity_warn",
        ):
            print(f"  {k}: {stats[k]}")
        if stats["polarity_codes"]:
            print(f"  polarity_codes: {stats['polarity_codes']}")
        if stats["examples"]["quote"]:
            print("  sample quote_warn:", stats["examples"]["quote"][0])
        if stats["examples"]["polarity"]:
            print("  sample polarity:", stats["examples"]["polarity"][0])
        if stats["examples"]["tone_leak"]:
            print("  sample tone_leak_after_strip:", stats["examples"]["tone_leak"][0])


if __name__ == "__main__":
    main()
