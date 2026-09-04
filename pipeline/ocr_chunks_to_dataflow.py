"""
将 DeepSeek-OCR 输出的 text_chunks_*.json 转换为 DataFlow QA 管道所需的
dataflow.json 格式。

DeepSeek-OCR text_chunks 格式（单文件 / 多文件目录均支持）:
  {
    "fileName": "xxx.pdf",
    "result": [
      {"id": 0, "title": [...], "content": "## 章节标题\n\n正文内容...", "external_filed": ...},
      ...
    ]
  }

输出格式（与 test_filter.py 的 TechDocQAPipelineV2 直接兼容）:
  [{"text": "..."}, ...]

用法示例:
  # 单个 text_chunks_*.json → dataflow.json
  python ocr_chunks_to_dataflow.py -i text_chunks_20251104_104727.json -o dataflow.json

  # 目录下所有 json 文件合并 → dataflow.json（多文档批量）
  python ocr_chunks_to_dataflow.py -i ./ocr_results/ -o dataflow.json

  # 控制最小/最大 chunk 字符数（过滤过短/过长的碎片）
  python ocr_chunks_to_dataflow.py -i text_chunks.json -o out.json --min-chars 80 --max-chars 4000
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# 内容质量过滤（与 test_filter.py _is_table_only_chunk 逻辑保持一致）
# ──────────────────────────────────────────────

_TABLE_RE = re.compile(r"^\s*\|", re.MULTILINE)


def _is_table_only_chunk(text: str) -> bool:
    """判断 chunk 是否以表格/路径列为主（无需在 QA pipeline 里重复过滤）。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    table_lines = sum(1 for ln in lines if _TABLE_RE.match(ln))
    return table_lines / len(lines) >= 0.6


def _clean_content(content: str) -> str:
    """去除首尾空白；多余空行压缩为两行。"""
    content = content.strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content


# ──────────────────────────────────────────────
# 读取单个 text_chunks_*.json 文件
# ──────────────────────────────────────────────

def load_ocr_chunks_file(path: Path, min_chars: int, max_chars: int) -> list[dict]:
    """从一个 DeepSeek-OCR text_chunks JSON 文件中提取合格 chunk。"""
    raw = json.loads(path.read_text(encoding="utf-8"))

    # 兼容两种结构：
    #   {"fileName": ..., "result": [...]}   ← DeepSeek-OCR 标准输出
    #   [{"content": ...}, ...]              ← 直接的 chunks 数组
    if isinstance(raw, dict):
        chunks = raw.get("result", raw.get("chunks", []))
        file_label = raw.get("fileName", path.name)
    elif isinstance(raw, list):
        chunks = raw
        file_label = path.name
    else:
        print(f"[WARN] {path}: 无法识别的 JSON 结构，跳过", file=sys.stderr)
        return []

    records = []
    for item in chunks:
        # 兼容字段名 content / text
        content = item.get("content") or item.get("text") or ""
        content = _clean_content(content)

        if len(content) < min_chars:
            continue
        if len(content) > max_chars:
            # 超长 chunk 直接截断（保留头部；LLM context 窗口有限）
            content = content[:max_chars]

        if _is_table_only_chunk(content):
            continue

        records.append({"text": content})

    print(f"  [{file_label}] chunks: {len(chunks)} 原始 → {len(records)} 保留")
    return records


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将 DeepSeek-OCR text_chunks.json 转换为 DataFlow dataflow.json"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入文件（text_chunks_*.json）或包含多个 json 文件的目录",
    )
    parser.add_argument(
        "-o", "--output", default="dataflow.json",
        help="输出文件路径（默认 dataflow.json）",
    )
    parser.add_argument(
        "--min-chars", type=int, default=80,
        help="保留 chunk 的最小字符数（默认 80）",
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000,
        help="保留 chunk 的最大字符数，超长截断（默认 4000）",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="关闭完全重复 chunk 去重（默认开启）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 收集所有待处理 json 文件
    if input_path.is_dir():
        json_files = sorted(input_path.rglob("text_chunks_*.json"))
        if not json_files:
            # 兜底：目录下所有 .json
            json_files = sorted(input_path.rglob("*.json"))
        print(f"目录模式：共发现 {len(json_files)} 个 JSON 文件")
    elif input_path.is_file():
        json_files = [input_path]
    else:
        print(f"[ERROR] 输入路径不存在：{input_path}", file=sys.stderr)
        sys.exit(1)

    # 逐文件提取 chunk
    all_records: list[dict] = []
    for fp in json_files:
        records = load_ocr_chunks_file(fp, args.min_chars, args.max_chars)
        all_records.extend(records)

    # 去重（保序）
    if not args.no_dedup:
        seen: set[str] = set()
        deduped: list[dict] = []
        for rec in all_records:
            key = rec["text"]
            if key not in seen:
                seen.add(key)
                deduped.append(rec)
        n_dup = len(all_records) - len(deduped)
        if n_dup:
            print(f"去重：移除 {n_dup} 条重复 chunk")
        all_records = deduped

    if not all_records:
        print("[WARN] 转换结果为空，请检查输入文件格式或 --min-chars 设置", file=sys.stderr)
        sys.exit(1)

    # 写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成：共 {len(all_records)} 个 chunk → {output_path}")
    print(f"下一步：")
    print(f"  TECHDOC_INPUT={output_path.resolve()} python test_filter.py v2")
    print(f"  # 或容器内：")
    print(f"  docker run --rm -it \\")
    print(f"    -e TECHDOC_INPUT=/data/{output_path.name} \\")
    print(f"    -v {output_path.parent.resolve()}:/data \\")
    print(f"    dataflow-qa:1.0.8-v2")


if __name__ == "__main__":
    main()
