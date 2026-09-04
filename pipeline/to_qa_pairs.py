"""Convert qa_pipeline_v2_step_step16.json to QA pair datasets.

Outputs (written alongside the source file):
- qa_pairs.json        : list of {"question", "answer"} (pretty JSON)
- qa_pairs.jsonl       : one {"question", "answer"} per line
- qa_pairs_alpaca.jsonl: Alpaca-style {"instruction", "input", "output"}
- qa_pairs_sharegpt.jsonl: ShareGPT-style {"conversations": [user, assistant]}
"""

from __future__ import annotations

import json
from pathlib import Path


SRC = Path(__file__).parent / "cache_local_qa" / "qa_pipeline_v2_step_step16.json"
OUT_DIR = SRC.parent


def pick_question(row: dict) -> str:
    # Prefer the final refined reverse-question, then fall back gracefully.
    for key in ("refined_reverse_question", "reverse_question", "rough_question", "instruction"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def pick_answer(row: dict) -> str:
    for key in ("refined_answer", "clean_answer", "initial_answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"Unexpected top-level type: {type(data)}"

    pairs: list[dict] = []
    for row in data:
        q = pick_question(row)
        a = pick_answer(row)
        if not q or not a:
            continue
        pairs.append(
            {
                "question": q,
                "answer": a,
                "question_type": row.get("question_type"),
                "difficulty": row.get("difficulty"),
                "rubric_score": row.get("rubric_score"),
                "rubric_verdict": row.get("rubric_verdict"),
                "meta_score": row.get("MetaScore"),
            }
        )

    qa_json = OUT_DIR / "qa_pairs.json"
    qa_jsonl = OUT_DIR / "qa_pairs.jsonl"
    alpaca_jsonl = OUT_DIR / "qa_pairs_alpaca.jsonl"
    sharegpt_jsonl = OUT_DIR / "qa_pairs_sharegpt.jsonl"

    qa_json.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with qa_jsonl.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps({"question": p["question"], "answer": p["answer"]}, ensure_ascii=False)
                + "\n"
            )

    with alpaca_jsonl.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {"instruction": p["question"], "input": "", "output": p["answer"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with sharegpt_jsonl.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": p["question"]},
                            {"from": "gpt", "value": p["answer"]},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Source records       : {len(data)}")
    print(f"Valid QA pairs       : {len(pairs)}")
    print(f"Written              :")
    print(f"  - {qa_json}")
    print(f"  - {qa_jsonl}")
    print(f"  - {alpaca_jsonl}")
    print(f"  - {sharegpt_jsonl}")


if __name__ == "__main__":
    main()
