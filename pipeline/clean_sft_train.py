#!/usr/bin/env python3
"""清洗 SFT 语料：过滤 Pipeline 产出的无效 instruction/output。

典型脏数据：
- instruction 为 ``, `...`, ` 占位符（逆向出题失败）
- instruction 为「改进后的问题」等 Refiner 泄漏
- QuestionRefiner 整段英文推理 / [Analysis Start] / [Improved Question Start] 泄漏
- output 为 `...` 或过短无效答案

用法：
    python clean_sft_train.py /path/to/sft_train.jsonl
    python clean_sft_train.py --all   # 清洗全部数据集并重建 sft_train_all.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MLF = _ROOT / "DataFlow" / "dataflow" / "utils" / "meta_label_filters.py"


def _load_mlf():
    spec = importlib.util.spec_from_file_location("_mlf_clean", _MLF)
    m = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(m)
    return m


_mlf = _load_mlf()
normalize_export_question = _mlf.normalize_export_question
is_valid_export_output = _mlf.is_valid_export_output


def clean_record(record: dict) -> dict | None:
    raw_inst = record.get("instruction") or record.get("question") or ""
    inst = normalize_export_question(raw_inst)
    if not inst:
        return None
    out = (record.get("output") or record.get("answer") or "").strip()
    if not is_valid_export_output(out):
        return None
    return {"instruction": inst, "output": out}


def clean_file(path: Path, in_place: bool = True) -> tuple[int, int]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept: list[dict] = []
    for ln in lines:
        row = clean_record(json.loads(ln))
        if row:
            kept.append(row)
    if in_place:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""),
            encoding="utf-8",
        )
    return len(lines), len(kept)


ALL_DATASETS = [
    Path("/home/wugk/finetune/finetune/data/故障管理/sft_train.jsonl"),
    Path("/home/wugk/finetune/finetune/data/用户指南/sft_train.jsonl"),
    Path(
        "/home/wugk/finetune/finetune/data/分布式数据库中间件(DDM) 24.1.35.20 开发指南(for 华为云Stack 8.3.1) 01/sft_train.jsonl"
    ),
    Path(
        "/home/wugk/finetune/finetune/data/分布式数据库中间件(DDM) 24.1.35.20 用户指南(for 华为云Stack 8.3.1) 01/sft_train.jsonl"
    ),
    Path(
        "/home/wugk/finetune/finetune/data/分布式数据库中间件(DDM) 24.1.35.20 API参考(for 华为云Stack 8.3.1) 01/sft_train.jsonl"
    ),
    Path("/home/wugk/finetune/finetune/data/01-01 告警处理/sft_train.jsonl"),
    Path(
        "/home/wugk/finetune/finetune/data/OceanStor Dorado V6系列 6.1.x & V700R001 性能监控指南/sft_train.jsonl"
    ),
]

MERGE_OUT = Path("/home/wugk/finetune/finetune/data/sft_train_all.jsonl")
MERGE_JSON = Path("/home/wugk/finetune/finetune/data/sft_train_all.json")


def merge_all() -> int:
    all_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for p in ALL_DATASETS:
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            key = (row["instruction"], row["output"])
            if key not in seen:
                seen.add(key)
                all_rows.append(row)
    MERGE_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + ("\n" if all_rows else ""),
        encoding="utf-8",
    )
    MERGE_JSON.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 sft_train.jsonl 无效 QA")
    parser.add_argument("files", nargs="*", help="待清洗 jsonl 文件")
    parser.add_argument("--all", action="store_true", help="清洗全部数据集并重建合并文件")
    args = parser.parse_args()

    targets = ALL_DATASETS if args.all else [Path(f) for f in args.files]
    if not targets:
        parser.error("请指定文件或使用 --all")

    print("=" * 60)
    total_before = total_after = 0
    for path in targets:
        if not path.is_file():
            print(f"[SKIP] 不存在: {path}")
            continue
        before, after = clean_file(path)
        total_before += before
        total_after += after
        print(f"[OK] {path.parent.name}: {before} -> {after} (-{before - after})")

    if args.all:
        merged = merge_all()
        print("=" * 60)
        print(f"合并: {MERGE_OUT} ({merged} 条)")
        print(f"JSON: {MERGE_JSON}")

    print("=" * 60)
    print(f"合计: {total_before} -> {total_after} (-{total_before - total_after})")


if __name__ == "__main__":
    main()
