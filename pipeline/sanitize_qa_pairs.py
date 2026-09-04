#!/usr/bin/env python3
"""Offline cleanup for exported QA pairs: strip ✅/⚠️ decorators, scrub meta labels.

Reads JSONL (one object per line) or a JSON array; writes the same shape.
Default answer fields tried: output, answer, refined_answer.

Usage:
  python sanitize_qa_pairs.py -i qa_pairs.jsonl -o qa_pairs.clean.jsonl
  python sanitize_qa_pairs.py -i qa_pairs_min.json -o qa_pairs_min.clean.json --is-array
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import importlib.util

# Resolve .../finetune (repo root that contains DataFlow/)
_ROOT = Path(__file__).resolve().parents[2]
_MLF = _ROOT / "DataFlow" / "dataflow" / "utils" / "meta_label_filters.py"
_spec = importlib.util.spec_from_file_location("_mlf_standalone", _MLF)
_mlf = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mlf)

scrub_meta_labels = _mlf.scrub_meta_labels
strip_training_decorators = _mlf.strip_training_decorators
looks_like_answer_as_quote = _mlf.looks_like_answer_as_quote
detect_question_answer_polarity_issue = _mlf.detect_question_answer_polarity_issue
detect_evaluator_tone = _mlf.detect_evaluator_tone


def _clean_text(s: str) -> str:
    out, _ = scrub_meta_labels(s or "")
    out, _ = strip_training_decorators(out)
    return out.strip()


def _quality_issues(obj: dict, answer_key: str, question_key: str, context_key: str) -> list[str]:
    ans = obj.get(answer_key, "")
    if not isinstance(ans, str) or not ans.strip():
        return []
    issues: list[str] = []
    ctx = obj.get(context_key, "")
    if isinstance(ctx, str) and looks_like_answer_as_quote(ans, ctx):
        issues.append("quote_dump")
    q = obj.get(question_key, "")
    if isinstance(q, str):
        issue = detect_question_answer_polarity_issue(q, ans)
        if issue:
            issues.append(issue)
    tone = detect_evaluator_tone(ans)
    if tone:
        issues.append("evaluator_tone:" + "|".join(tone[:3]))
    return issues


def _clean_obj(
    obj: dict,
    keys: list[str],
    question_key: str,
    context_key: str,
    drop_suspect: bool,
    stats: dict[str, int],
) -> dict | None:
    o = dict(obj)
    for k in keys:
        if k in o and isinstance(o[k], str):
            o[k] = _clean_text(o[k])
            issues = _quality_issues(o, k, question_key, context_key)
            for issue in issues:
                stats[issue] = stats.get(issue, 0) + 1
            if issues:
                o.setdefault("sanitize_warnings", {})[k] = issues
                if drop_suspect:
                    stats["dropped"] = stats.get("dropped", 0) + 1
                    return None
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanitize QA export for SFT (decorators + meta scrub).")
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    ap.add_argument(
        "--keys",
        default="output,answer,refined_answer",
        help="Comma-separated field names to clean",
    )
    ap.add_argument(
        "--is-array",
        action="store_true",
        help="Input is a JSON array of objects (not JSONL)",
    )
    ap.add_argument("--question-key", default="instruction", help="Question field for polarity checks")
    ap.add_argument("--context-key", default="input", help="Context field for quote-dump checks")
    ap.add_argument(
        "--drop-suspect",
        action="store_true",
        help="Drop records that still trigger quote/polarity/evaluator-tone warnings",
    )
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    stats: dict[str, int] = {"read": 0, "written": 0, "dropped": 0}

    raw = args.input.read_text(encoding="utf-8")
    if args.is_array:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise SystemExit("Expected JSON array with --is-array")
        out = []
        for x in data:
            if not isinstance(x, dict):
                continue
            stats["read"] += 1
            cleaned = _clean_obj(
                x, keys, args.question_key, args.context_key, args.drop_suspect, stats
            )
            if cleaned is not None:
                out.append(cleaned)
                stats["written"] += 1
        args.output.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            stats["read"] += 1
            cleaned = _clean_obj(
                obj, keys, args.question_key, args.context_key, args.drop_suspect, stats
            )
            if cleaned is not None:
                lines.append(json.dumps(cleaned, ensure_ascii=False))
                stats["written"] += 1
    args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
