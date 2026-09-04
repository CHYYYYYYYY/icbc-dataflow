#!/usr/bin/env python3
"""运行 TechDocQAPipelineV2：技术文档 QA 管道主程序。

用法：
    /home/wugk/.cursor/plans/脚本03_前台运行_故障管理数据集.sh
    /home/wugk/.cursor/plans/脚本04_前台运行_用户指南数据集.sh
    /home/wugk/.cursor/plans/脚本05_后台运行_故障管理_本地llama-server.sh   # 本地 llama-server 后台
    /home/wugk/.cursor/plans/脚本06_后台运行_用户指南_本地llama-server.sh
    /home/wugk/.cursor/plans/脚本08_后台运行_故障管理_阿里云API.sh   # 阿里云 API 后台

环境变量：
    DF_API_KEY / DF_API_URL / DF_MODEL_NAME
    TECHDOC_INPUT / QA_OUTPUT_DIR / QA_CACHE_DIR
    DF_MAX_WORKERS / NUM_QUESTIONS
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATAFLOW_ROOT = Path(os.environ.get("DATAFLOW_ROOT", REPO_ROOT / "DataFlow"))
TEST_DIR = Path(os.environ.get("TECHDOC_PIPELINE_DIR", REPO_ROOT / "pipeline"))
OUTPUT_DIR = Path(
    os.environ.get(
        "QA_OUTPUT_DIR",
        str(REPO_ROOT / "outputs"),
    )
)
DEFAULT_INPUT = REPO_ROOT / "data" / "text_chunks.json"
CACHE_DIR = Path(
    os.environ.get(
        "QA_CACHE_DIR",
        str(OUTPUT_DIR / "qa_pipeline_cache"),
    )
)
FILE_PREFIX = os.environ.get("QA_FILE_PREFIX", "fault_mgmt_converted_qa")

# 默认使用阿里云 OpenAI 兼容接口；密钥必须由环境变量传入。
os.environ.setdefault(
    "DF_API_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
)
os.environ.setdefault("DF_MODEL_NAME", "qwen3.6-plus")

if str(DATAFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_filter import TechDocQAPipelineV2  # noqa: E402


def _latest_step_file(cache_dir: Path, prefix: str) -> Path | None:
    if not cache_dir.is_dir():
        return None
    pat = re.compile(rf"^{re.escape(prefix)}_step(\d+)\.json$")
    candidates = []
    for p in cache_dir.glob(f"{prefix}_step*.json"):
        if pat.match(p.name):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(pat.match(p.name).group(1)))


def _export_latest(cache_dir: Path, prefix: str, output_dir: Path) -> tuple[Path | None, Path | None]:
    latest = _latest_step_file(cache_dir, prefix)
    if not latest:
        print("[WARN] 未找到 step 缓存，跳过导出。", file=sys.stderr)
        return None, None

    # 从文件名提取 step 编号：fault_mgmt_converted_qa_step17.json
    stem = latest.stem  # fault_mgmt_converted_qa_step17
    step_part = stem.rsplit("_step", 1)[-1]
    if not step_part.isdigit():
        print(f"[WARN] 无法解析 step 编号: {latest.name}", file=sys.stderr)
        return None, None

    env = os.environ.copy()
    env["QA_CACHE_DIR"] = str(cache_dir)
    env["QA_FILE_PREFIX"] = prefix

    cmd = [
        sys.executable,
        str(TEST_DIR / "convert_step15_to_qa.py"),
        step_part,
        "--with-context",
        "--no-audit",
    ]
    print(f"\n[export] 从 {latest.name} 导出 QA 对（with-context）...")
    subprocess.run(cmd, cwd=str(TEST_DIR), env=env, check=True)

    out_jsonl = cache_dir / f"{prefix}_step{step_part}_qa_pairs_ctx.jsonl"
    out_alpaca = cache_dir / f"{prefix}_step{step_part}_qa_ctx.json"
    sft_jsonl = output_dir / f"{prefix}_sft.jsonl"
    sft_json = output_dir / f"{prefix}_sft.json"
    if out_jsonl.is_file():
        sft_jsonl.write_bytes(out_jsonl.read_bytes())
    if out_alpaca.is_file():
        sft_json.write_bytes(out_alpaca.read_bytes())
    return out_jsonl if out_jsonl.is_file() else None, out_alpaca if out_alpaca.is_file() else None


def main() -> None:
    input_file = Path(os.environ.get("TECHDOC_INPUT", str(DEFAULT_INPUT)))
    if not input_file.is_file():
        print(f"[ERROR] 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)

    num_questions = int(os.environ.get("NUM_QUESTIONS", "4"))
    max_workers = int(os.environ.get("DF_MAX_WORKERS", "20"))
    resume = os.environ.get("QA_RESUME", "1").lower() not in ("0", "false", "no")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import json
    chunk_count = len(json.loads(input_file.read_text(encoding="utf-8")))
    latest_cached = _latest_step_file(CACHE_DIR, FILE_PREFIX)
    latest_step = None
    if latest_cached:
        m = re.match(rf".*_step(\d+)\.json$", latest_cached.name)
        latest_step = int(m.group(1)) if m else None

    dataset_label = os.environ.get("DATASET_LABEL", OUTPUT_DIR.name)
    distill_tag = os.environ.get("DISTILL_TAG", "运维")

    print("=" * 60)
    print(f"TechDoc QA Pipeline V2 — {dataset_label} converted 数据集")
    print("=" * 60)
    print(f"  输入文件   : {input_file}  ({chunk_count} chunks)")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  缓存目录   : {CACHE_DIR}")
    print(f"  文件前缀   : {FILE_PREFIX}")
    print(f"  API URL    : {os.environ['DF_API_URL']}")
    print(f"  模型       : {os.environ.get('DF_MODEL_NAME', 'qwen3.6-plus')}")
    print(f"  每 chunk 题数: {num_questions}  (预估初始 QA ≈ {chunk_count * num_questions})")
    print(f"  并发数     : {max_workers}")
    print(f"  断点续跑   : {'开启' if resume else '关闭'}", end="")
    if resume and latest_step:
        print(f"（已有 step{latest_step}.json）")
    else:
        print()
    print("=" * 60)

    os.chdir(TEST_DIR)

    pipeline = TechDocQAPipelineV2(
        input_file=str(input_file),
        num_questions=num_questions,
        distill_tag=distill_tag,
        min_score=3.5,
        max_score=5.0,
        min_soft_score=12.0,
        api_url=os.environ["DF_API_URL"],
        model_name=os.environ.get("DF_MODEL_NAME", "qwen3.6-plus"),
        max_workers=max_workers,
        cache_path=str(CACHE_DIR),
        file_name_prefix=FILE_PREFIX,
        resume=resume,
    )
    pipeline.forward()

    exported_jsonl, exported_alpaca = _export_latest(CACHE_DIR, FILE_PREFIX, OUTPUT_DIR)

    latest = _latest_step_file(CACHE_DIR, FILE_PREFIX)
    print("=" * 60)
    print("Pipeline 运行完成。")
    if latest:
        print(f"  最新 step 缓存: {latest}")
    if exported_jsonl:
        print(f"  导出 JSONL    : {exported_jsonl}")
    if exported_alpaca:
        print(f"  导出 Alpaca   : {exported_alpaca}")
    print("=" * 60)


if __name__ == "__main__":
    main()
