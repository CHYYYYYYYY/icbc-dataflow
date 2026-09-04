#!/usr/bin/env python3
"""单文档跨 chunk 多跳 QA 旁路生成（与 V2 单跳管道并行，不修改 test_filter.py）。

用法：
    OCR_SOURCE=/path/to/text_chunks_*.json \\
    QA_OUTPUT_DIR=/path/to/doc_dir \\
    MH_FILE_PREFIX=dorado_perf_multihop \\
    MH_DRY_RUN=1 \\
    python multihop_cross_chunk.py

环境变量见 plan：单文档跨chunk多跳qa_4415a191.plan.md
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATAFLOW_ROOT = Path(os.environ.get("DATAFLOW_ROOT", REPO_ROOT / "DataFlow"))
if str(DATAFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_ROOT))

_MLF = DATAFLOW_ROOT / "dataflow" / "utils" / "meta_label_filters.py"
_spec = importlib.util.spec_from_file_location("_mlf_mh", _MLF)
_mlf = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mlf)
extract_llm_answer_text = _mlf.extract_llm_answer_text
is_valid_export_question = _mlf.is_valid_export_question

# ── 配置（环境变量） ──────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


@dataclass
class MHConfig:
    ocr_source: Path
    output_dir: Path
    file_prefix: str
    dataset_label: str
    max_clusters: int = 80
    num_q_per_cluster: int = 1
    min_chunk_chars: int = 150
    max_context_chars: int = 10000
    max_fragment_chars: int = 3500
    min_complexity: float = 0.55
    enable_degeneracy: bool = True
    dry_run: bool = False
    resume: bool = True
    api_url: str = "http://127.0.0.1:19000/v1/chat/completions"
    api_key: str = "local"
    model_name: str = "Qwen3.6-27B-Q5_K_M.gguf"
    max_workers: int = 2
    max_llm_retries: int = 1

    @classmethod
    def from_env(cls) -> MHConfig:
        ocr = os.environ.get("OCR_SOURCE", "").strip()
        if not ocr:
            raise ValueError("请设置环境变量 OCR_SOURCE（原始 text_chunks_*.json 路径）")
        ocr_path = Path(ocr)
        out = Path(
            os.environ.get("QA_OUTPUT_DIR", str(ocr_path.parent))
        )
        return cls(
            ocr_source=ocr_path,
            output_dir=out,
            file_prefix=os.environ.get("MH_FILE_PREFIX", "multihop_qa"),
            dataset_label=os.environ.get("DATASET_LABEL", out.name),
            max_clusters=_env_int("MH_MAX_CLUSTERS", 80),
            num_q_per_cluster=_env_int("MH_NUM_Q", 1),
            min_chunk_chars=_env_int("MH_MIN_CHUNK_CHARS", 150),
            max_context_chars=_env_int("MH_MAX_CONTEXT", 10000),
            max_fragment_chars=_env_int("MH_MAX_FRAGMENT", 3500),
            min_complexity=_env_float("MH_MIN_COMPLEXITY", 0.55),
            enable_degeneracy=_env_bool("MH_ENABLE_DEGENERACY", True),
            dry_run=_env_bool("MH_DRY_RUN", False),
            resume=_env_bool("MH_RESUME", True),
            api_url=os.environ.get(
                "DF_API_URL", "http://127.0.0.1:19000/v1/chat/completions"
            ),
            api_key=os.environ.get("DF_API_KEY", "local"),
            model_name=os.environ.get("DF_MODEL_NAME", "Qwen3.6-27B-Q5_K_M.gguf"),
            max_workers=_env_int("DF_MAX_WORKERS", 2),
            max_llm_retries=_env_int("MH_MAX_LLM_RETRIES", 1),
        )


# ── Chunk 加载与过滤 ──────────────────────────────────────────────

_TABLE_LINE_RE = re.compile(r"^\s*\|", re.MULTILINE)
_SECTION_PATTERNS = [
    re.compile(r"^#{1,4}\s*(\d+(?:\.\d+)*)\b"),
    re.compile(r"^(\d+(?:\.\d+)+)\s+"),
    re.compile(r"^表\s*(\d+)[-－](\d+)"),
    re.compile(r"^表\s*(\d+(?:\.\d+)+)"),
]
_TOC_SECTION_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_ENTITY_RE = re.compile(
    r"[`A-Za-z][\w./-]{2,}|[\u4e00-\u9fff]{2,}(?:命令|告警|指标|端口|链路|存储池|LUN|协议)"
)
_CAUSAL_RE = re.compile(r"因此|从而|然后|需|导致|才能|之后|方可|进而|以便")


@dataclass
class ChunkRecord:
    chunk_id: str
    content: str
    section: str | None = None
    title: str = ""
    order: int = 0


@dataclass
class Cluster:
    cluster_id: str
    chunks: list[ChunkRecord]
    parent_section: str
    sections: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.chunks)


def _clean_text(text: str) -> str:
    text = text.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _is_table_heavy(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    table_lines = sum(1 for ln in lines if _TABLE_LINE_RE.match(ln) or "<table>" in ln.lower())
    return table_lines / len(lines) >= 0.6


def _is_toc_chunk(text: str) -> bool:
    sections = _TOC_SECTION_RE.findall(text)
    prose = re.sub(r"\s+", "", text)
    prose = re.sub(r"\d+(?:\.\d+)+", "", prose)
    prose = re.sub(r"[|<>/\-\.\s]", "", prose)
    if len(sections) >= 4 and len(prose) < 80:
        return True
    # 单行/短文本内密集节号列表（如「3.1 xxx3.2 xxx3.3 xxx」）
    head = text.strip().split("\n", 1)[0]
    head_sections = _TOC_SECTION_RE.findall(head)
    if len(head_sections) >= 3 and len(head) < 500:
        head_prose = re.sub(r"\d+(?:\.\d+)+", "", re.sub(r"\s+", "", head))
        if len(head_prose) < max(len(head) * 0.55, 120):
            return True
    if len(sections) >= 3 and len(text) < 500 and len(prose) < 200:
        return True
    return False


def extract_section(content: str) -> str | None:
    head = content.strip().split("\n", 1)[0].strip()
    for pat in _SECTION_PATTERNS:
        m = pat.match(head)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2 and m.group(2):
            return f"{m.group(1)}.{m.group(2)}"
        return m.group(1)
    return None


def parent_section(section: str) -> str:
    parts = section.split(".")
    if len(parts) <= 1:
        return section
    return ".".join(parts[:-1])


def _section_sort_key(section: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in section.split("."))


def load_ocr_chunks(path: Path, min_chars: int) -> list[ChunkRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("result", raw.get("chunks", []))
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"无法识别的 OCR JSON 结构: {path}")

    records: list[ChunkRecord] = []
    for order, item in enumerate(items):
        content = _clean_text(item.get("content") or item.get("text") or "")
        if len(content) < min_chars:
            continue
        if _is_table_heavy(content) or _is_toc_chunk(content):
            continue
        chunk_id = str(item.get("id", order))
        section = extract_section(content)
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                content=content,
                section=section,
                title=str(item.get("title") or ""),
                order=order,
            )
        )
    return records


def build_clusters(chunks: list[ChunkRecord], max_clusters: int) -> list[Cluster]:
    by_parent: dict[str, list[ChunkRecord]] = defaultdict(list)
    for ch in chunks:
        if not ch.section:
            continue
        by_parent[parent_section(ch.section)].append(ch)

    clusters: list[Cluster] = []
    seen_keys: set[tuple[str, ...]] = set()

    for parent, group in sorted(by_parent.items(), key=lambda x: x[0]):
        group = sorted(group, key=lambda c: _section_sort_key(c.section or "0"))
        for win_size in (2, 3):
            for i in range(len(group) - win_size + 1):
                window = group[i : i + win_size]
                sections = [c.section for c in window if c.section]
                if len(set(sections)) < 2:
                    continue
                key = tuple(c.chunk_id for c in window)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cid = f"{parent}_f{i}_s{win_size}"
                clusters.append(
                    Cluster(
                        cluster_id=cid,
                        chunks=window,
                        parent_section=parent,
                        sections=sections,
                    )
                )

    clusters.sort(key=lambda c: (c.parent_section, c.cluster_id))
    return clusters[:max_clusters]


def merge_cluster_context(cluster: Cluster, max_fragment: int, max_total: int) -> str:
    parts: list[str] = []
    total = 0
    for idx, ch in enumerate(cluster.chunks, 1):
        body = ch.content
        if len(body) > max_fragment:
            body = body[:max_fragment] + "\n…（片段截断）"
        sec = ch.section or "?"
        block = f"【片段{idx} | 节号 {sec}】\n{body}"
        if total + len(block) > max_total:
            remain = max_total - total
            if remain > 200:
                parts.append(block[:remain] + "\n…（上下文截断）")
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


# ── LLM Prompts ──────────────────────────────────────────────────

STEP_A_SYSTEM = """你是技术文档跨片段依赖分析专家。根据多个文档片段，抽取跨片段推理依赖图。
硬约束：
1. 2 个片段时 reasoning_path 长度=2；3 个片段时长度=3（每片段至少 1 个节点）
2. reasoning_path 中相邻节点必须来自不同 fragment（禁止 E2→E3 同在片段2）
3. 若无法找到满足条件的跨片段依赖路径，输出 {"skip": true, "reason": "..."}
4. 只输出 JSON，放在 <answer> 标签内；禁止在 thinking 中写 <answer> 标签

输出 JSON 格式：
{
  "entities": [{"id":"E1","fragment":1,"name":"...","fact":"..."}],
  "edges": [{"from":"E1","to":"E2","relation":"...","fragment_bridge":[1,2]}],
  "reasoning_path": ["E1","E2"],
  "path_type": "配置链|流程链|因果链|对比链|排障链"
}"""

STEP_A_RETRY_SUFFIX = """
上次输出未通过校验：{reason}
请重新生成。2 片段时 reasoning_path 必须恰好 2 个节点且 fragment 交替（如 1→2）。"""

STEP_B_SYSTEM = """你是技术文档多跳 QA 出题专家。根据已给出的跨片段依赖链出题。
硬约束：
1. 问题不得泄露中间桥接实体（不能把依赖链全写在题干里）
2. 禁止并列拼接（不允许「A是什么，B是什么」型双问句）
3. reasoning_steps 步数 >= 片段数；每步必须引用对应 fragment 的新事实
4. 问题必须依赖全部片段才能完整作答；任一片段单独无法准确回答
5. 题干不要复述片段1的负载/流程描述（避免让片段2单独可答）；用抽象场景引导
6. 只输出 JSON，放在 <answer> 标签内；禁止在 thinking 中写 <answer> 标签

输出 JSON 格式：
{
  "question": "...",
  "reasoning_steps": [
    {"step":1,"fragment":1,"uses_facts":["E1"],"text":"..."},
    {"step":2,"fragment":2,"uses_facts":["E2"],"text":"..."}
  ],
  "answer": "...",
  "supporting_facts": ["...", "..."],
  "source_fragments": [1,2],
  "type": "配置链",
  "reasoning_path": ["E1","E2"]
}"""

STEP_B_RETRY_SUFFIX = """
上次输出 JSON 解析失败或格式不合规。请严格只输出一个 <answer>...</answer>，其中仅含合法 JSON，不要 thinking 内容。"""

DEGENERACY_SYSTEM = """你是 QA 可答性审计员。给定一个问题和单个文档片段，判断能否仅凭该片段完整准确地回答问题。
只输出 YES 或 NO（放在 <answer> 标签内）。"""


_THINKING_ANSWER_RE = re.compile(
    r".*?\s*<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)


class LocalLLMClient:
    """轻量 OpenAI 兼容客户端（不依赖 dataflow/torch）。"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        max_workers: int = 2,
        temperature: float = 0.2,
        max_retries: int = 5,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_retries = max_retries
        self._log = logging.getLogger("multihop_cross_chunk")

    def _format_response(self, response: dict) -> str:
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if _THINKING_ANSWER_RE.search(content):
            return content
        reasoning = message.get("reasoning_content")
        if reasoning:
            return f"{reasoning}\n<answer>{content}</answer>"
        return content

    def _call_one(self, idx: int, system_prompt: str, user_input: str) -> tuple[int, str | None]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=1800
                )
                if resp.status_code == 200:
                    return idx, self._format_response(resp.json())
                self._log.warning(
                    "API %s attempt %d: HTTP %s",
                    self.api_url,
                    attempt + 1,
                    resp.status_code,
                )
            except Exception as exc:
                self._log.warning("API attempt %d error: %s", attempt + 1, exc)
            time.sleep(2**attempt)
        return idx, None

    def generate_from_input(
        self, user_inputs: list[str], system_prompt: str = "You are a helpful assistant"
    ) -> list[str | None]:
        if len(user_inputs) == 1:
            _, out = self._call_one(0, system_prompt, user_inputs[0])
            return [out]
        results: list[str | None] = [None] * len(user_inputs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [
                pool.submit(self._call_one, i, system_prompt, u)
                for i, u in enumerate(user_inputs)
            ]
            for fut in as_completed(futs):
                idx, out = fut.result()
                results[idx] = out
        return results


def _extract_answer_blocks(text: str) -> list[str]:
    """取所有 <answer> 块，优先使用最后一个（thinking 中常有伪 answer）。"""
    blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    body = extract_llm_answer_text(text) or str(text).strip()
    return [body] if body else []


def _json_loads_relaxed(body: str) -> dict | None:
    body = body.strip()
    if not body:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    candidates = [body]
    m = re.search(r"\{.*\}", body, re.DOTALL)
    if m and m.group(0) != body:
        candidates.append(m.group(0))
    # 括号匹配提取最外层 JSON
    start = body.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(body)):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(body[start : i + 1])
                    break
        start = body.find("{", start + 1)
    seen: set[str] = set()
    for cand in candidates:
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        if cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_json_from_llm(text: str) -> dict | None:
    if not text:
        return None
    for body in reversed(_extract_answer_blocks(text)):
        obj = _json_loads_relaxed(body)
        if obj:
            return obj
    return _json_loads_relaxed(str(text))


def _token_set(text: str) -> set[str]:
    """中英文 token：英文词 + 中文 2-gram（避免整句连成单一 token 导致 grounding 失败）。"""
    text = text.lower()
    tokens: set[str] = set()
    tokens.update(re.findall(r"[a-z0-9]{2,}", text))
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if 2 <= len(seg) <= 6:
            tokens.add(seg)
        for i in range(len(seg) - 1):
            tokens.add(seg[i : i + 2])
    return tokens


def _strip_step_meta(text: str) -> str:
    return re.sub(r"^(?:根据|查阅|结合|参考)?片段\s*\d+[，,：:\s]*", "", text.strip())


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _normalize_fact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


# ── 四层质检 ──────────────────────────────────────────────────────

@dataclass
class QARecord:
    cluster_id: str
    question: str
    answer: str
    input_context: str
    reasoning_steps: list[dict]
    supporting_facts: list[str]
    reasoning_path: list[str]
    multihop_type: str
    complexity_score: float
    source_sections: list[str]
    source_chunk_ids: list[str]
    dataset: str
    dependency_graph: dict = field(default_factory=dict)
    raw_step_a: str = ""
    raw_step_b: str = ""
    raw_degeneracy: list[str] = field(default_factory=list)


def validate_structure(qa: dict, cluster: Cluster) -> tuple[bool, str]:
    n = cluster.size
    steps = qa.get("reasoning_steps") or []
    if not isinstance(steps, list) or len(steps) < n:
        return False, f"S1: steps={len(steps)} < chunk_count={n}"

    frags_in_steps: list[int] = []
    for st in steps:
        if not isinstance(st, dict):
            return False, "S2: invalid step"
        frag = st.get("fragment")
        if not isinstance(frag, int) or frag < 1 or frag > n:
            return False, f"S2: bad fragment={frag}"
        frags_in_steps.append(frag)
        text = str(st.get("text") or "").strip()
        if len(text) < 15:
            return False, "S5: step too short"

    src_frags = qa.get("source_fragments") or []
    if len(set(src_frags)) < 2:
        return False, "S3: source_fragments < 2"

    covered = set(frags_in_steps)
    if len(covered) < n:
        return False, f"S4: fragments not fully covered {covered}"

    mh_type = str(qa.get("type") or "")
    sim_threshold = 0.85 if "对比" in mh_type else 0.65
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            if _jaccard(str(steps[i].get("text", "")), str(steps[j].get("text", ""))) >= sim_threshold:
                return False, "S6: steps too similar"

    question = str(qa.get("question") or "").strip()
    facts = qa.get("supporting_facts") or []
    if facts:
        hit_all = all(_normalize_fact(f) in _normalize_fact(question) for f in facts if f)
        if hit_all and len(facts) >= 2:
            return False, "S7: question leaks all facts"

    if not is_valid_export_question(question):
        return False, "S7: invalid question"

    answer = str(qa.get("answer") or "").strip()
    ans_chars = sum(1 for c in answer if c.isalnum() or "\u4e00" <= c <= "\u9fff")
    # 允许「DeviceManager」「30秒」类短答案，但拒绝空泛复述
    has_technical_short = bool(re.search(r"\d", answer)) and ans_chars >= 2
    if has_technical_short:
        pass
    elif ans_chars < 8 or (len(answer) < 12 and ans_chars < 10):
        return False, "answer too short"

    return True, ""


def _fact_chunk_score(fact: str, content: str) -> float:
    """事实/步骤文本与 chunk 的匹配分：优先子串命中，否则 token 重叠。"""
    norm_f = _normalize_fact(fact)
    norm_c = _normalize_fact(content)
    if norm_f and len(norm_f) >= 6 and norm_f in norm_c:
        return 1.0
    tokens = _token_set(fact)
    if tokens:
        content_tokens = _token_set(content)
        overlap = len(tokens & content_tokens) / len(tokens)
        if overlap >= 0.35:
            return 0.5 + overlap * 0.5
    return _jaccard(fact, content)


def _step_chunk_score(step_text: str, content: str) -> float:
    return _fact_chunk_score(step_text, content)


def _entity_map(qa: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ent in qa.get("_entities") or []:
        if isinstance(ent, dict) and ent.get("id") is not None:
            out[str(ent["id"])] = ent
    return out


def _step_grounded(
    step_text: str,
    content: str,
    uses_facts: list[Any] | None = None,
    entities: dict[str, dict] | None = None,
) -> bool:
    """步骤是否锚定到对应 chunk：token 重叠、子串命中或 StepA 实体 fact。"""
    text = _strip_step_meta(step_text)
    shared = _token_set(text) & _token_set(content)
    if shared:
        return True
    if _fact_chunk_score(text, content) >= 0.08:
        return True
    if uses_facts and entities:
        for ref in uses_facts:
            ref_s = str(ref).strip()
            ent = entities.get(ref_s)
            if ent:
                fact = str(ent.get("fact") or ent.get("name") or "").strip()
                if fact and _fact_chunk_score(fact, content) >= 0.08:
                    return True
            elif len(ref_s) >= 6 and _fact_chunk_score(ref_s, content) >= 0.08:
                return True
    return False


def validate_cross_source(qa: dict, cluster: Cluster) -> tuple[bool, str]:
    steps = [st for st in (qa.get("reasoning_steps") or []) if isinstance(st, dict)]
    if not steps:
        return False, "C0: no reasoning_steps"

    entities = _entity_map(qa)
    frag_hits: set[int] = set()
    for st in steps:
        frag = st.get("fragment")
        text = str(st.get("text") or "").strip()
        if not isinstance(frag, int) or frag < 1 or frag > len(cluster.chunks):
            return False, f"C0: bad step fragment={frag}"
        if len(text) < 10:
            return False, "C0: empty step text"
        ch = cluster.chunks[frag - 1]
        uses = st.get("uses_facts") or []
        if not _step_grounded(text, ch.content, uses_facts=uses, entities=entities):
            return False, f"C1: step{st.get('step')} not grounded frag={frag}"
        frag_hits.add(frag)

    if len(frag_hits) < 2:
        return False, "C2: steps cover < 2 fragments"

    entity_frags = {}
    for ent in (qa.get("_entities") or []):
        if isinstance(ent, dict) and ent.get("id") is not None:
            entity_frags[str(ent["id"])] = ent.get("fragment")

    for st in qa.get("reasoning_steps") or []:
        if not isinstance(st, dict):
            continue
        frag = st.get("fragment")
        for eid in st.get("uses_facts") or []:
            ef = entity_frags.get(str(eid))
            if ef is not None and ef != frag:
                return False, f"C3: step fragment {frag} != entity {eid} fragment {ef}"

    return True, ""


def validate_degeneracy(qa: dict, cluster: Cluster) -> tuple[bool, str]:
    """程序化 L3：步骤已跨片段时放宽；仅拒绝单片段自洽。"""
    answer = str(qa.get("answer") or "").strip()
    question = str(qa.get("question") or "").strip()
    if not answer or not question:
        return False, "L3: empty qa"

    steps = [st for st in (qa.get("reasoning_steps") or []) if isinstance(st, dict)]
    step_frags = {st.get("fragment") for st in steps if isinstance(st.get("fragment"), int)}
    if len(step_frags) >= 2 and len(step_frags) >= min(cluster.size, 2):
        return True, ""

    per_frag: list[tuple[float, float]] = []
    for ch in cluster.chunks:
        per_frag.append(
            (_fact_chunk_score(answer, ch.content), _fact_chunk_score(question, ch.content))
        )

    for fi, (ans_s, q_s) in enumerate(per_frag, 1):
        if ans_s >= 0.45 and q_s >= 0.18:
            others_ans = max(
                (s[0] for j, s in enumerate(per_frag) if j != fi - 1),
                default=0.0,
            )
            if others_ans < 0.15:
                return False, f"L3:fragment{fi}=self_sufficient"

    return True, ""


def repair_dependency_graph(graph: dict, cluster: Cluster) -> dict:
    """修正相邻同 fragment 的 path；2 片段时压缩为 1→2 交替链。"""
    entities = {
        str(e.get("id")): e
        for e in (graph.get("entities") or [])
        if isinstance(e, dict) and e.get("id") is not None
    }
    path = [str(x) for x in (graph.get("reasoning_path") or []) if str(x) in entities]
    n = cluster.size

    if n == 2:
        by_frag: dict[int, list[str]] = defaultdict(list)
        for eid, ent in entities.items():
            frag = ent.get("fragment")
            if isinstance(frag, int):
                by_frag[frag].append(eid)
        if by_frag.get(1) and by_frag.get(2):
            e1 = next((e for e in path if entities[e].get("fragment") == 1), by_frag[1][0])
            e2 = next((e for e in path if entities[e].get("fragment") == 2), by_frag[2][0])
            path = [e1, e2]
    else:
        fixed: list[str] = []
        for eid in path:
            frag = entities[eid].get("fragment")
            if fixed:
                prev_frag = entities[fixed[-1]].get("fragment")
                if prev_frag == frag:
                    continue
            fixed.append(eid)
        path = fixed
        covered = {entities[e].get("fragment") for e in path}
        for fi in range(1, n + 1):
            if fi in covered:
                continue
            for eid, ent in entities.items():
                if ent.get("fragment") == fi and eid not in path:
                    path.append(eid)
                    covered.add(fi)
                    break

    out = dict(graph)
    out["reasoning_path"] = path[: max(n, 2)]
    return out


def compute_complexity(qa: dict, cluster: Cluster) -> float:
    n = cluster.size
    steps = qa.get("reasoning_steps") or []
    frags = {st.get("fragment") for st in steps if isinstance(st, dict)}
    step_score = min(len(steps) / max(n, 1), 1.0) * 0.25
    cross_score = (len(frags) / max(n, 1)) * 0.25

    entity_hits = 0
    for st in steps:
        if isinstance(st, dict) and _ENTITY_RE.search(str(st.get("text", ""))):
            entity_hits += 1
    entity_score = (entity_hits / max(len(steps), 1)) * 0.20

    causal = 0
    texts = [str(st.get("text", "")) for st in steps if isinstance(st, dict)]
    for i in range(1, len(texts)):
        if _CAUSAL_RE.search(texts[i]) or _token_set(texts[i - 1]) & _token_set(texts[i]):
            causal += 1
    causal_score = (causal / max(len(texts) - 1, 1)) * 0.15

    answer = str(qa.get("answer") or "")
    max_step = max((len(t) for t in texts), default=0)
    incr = 1.0 if len(answer) > max_step * 1.2 else 0.3
    incr_score = incr * 0.15

    return round(step_score + cross_score + entity_score + causal_score + incr_score, 3)


def validate_dependency_graph(graph: dict, cluster: Cluster) -> tuple[bool, str]:
    if graph.get("skip"):
        return False, f"stepA skip: {graph.get('reason', '')}"
    path = graph.get("reasoning_path") or []
    n = cluster.size
    if len(path) < n:
        return False, f"stepA path len {len(path)} < {n}"

    entities = {str(e.get("id")): e for e in (graph.get("entities") or []) if isinstance(e, dict)}
    prev_frag = None
    for eid in path:
        ent = entities.get(str(eid))
        if not ent:
            return False, f"stepA missing entity {eid}"
        frag = ent.get("fragment")
        if not isinstance(frag, int):
            return False, f"stepA bad fragment on {eid}"
        if prev_frag is not None and frag == prev_frag:
            return False, f"stepA adjacent same fragment {frag}"
        prev_frag = frag

    return True, ""


# ── 主流程 ────────────────────────────────────────────────────────

class MultiHopPipeline:
    def __init__(self, cfg: MHConfig):
        self.cfg = cfg
        self.cache_dir = cfg.output_dir / "multihop_cache"
        self.log_dir = cfg.output_dir / "logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.clusters_path = self.cache_dir / f"{cfg.file_prefix}_clusters.json"
        self.raw_path = self.cache_dir / f"{cfg.file_prefix}_raw.json"
        self.sft_path = cfg.output_dir / f"{cfg.file_prefix}_multihop_sft.jsonl"

        self.reject_stats: Counter = Counter()
        self.llm = None

    def _setup_logger(self) -> logging.Logger:
        log = logging.getLogger("multihop_cross_chunk")
        log.setLevel(logging.INFO)
        if not log.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh = logging.FileHandler(
                self.log_dir / "multihop_latest.log", encoding="utf-8"
            )
            fh.setFormatter(fmt)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            log.addHandler(fh)
            log.addHandler(sh)
        return log

    def _get_llm(self) -> LocalLLMClient:
        if self.llm is None:
            self.llm = LocalLLMClient(
                api_url=self.cfg.api_url,
                api_key=self.cfg.api_key,
                model_name=self.cfg.model_name,
                max_workers=self.cfg.max_workers,
                temperature=0.2,
            )
        return self.llm

    def run(self) -> dict[str, Any]:
        log = self._setup_logger()
        log.info("=" * 60)
        log.info("MultiHop Cross-Chunk QA — %s", self.cfg.dataset_label)
        log.info("  OCR_SOURCE : %s", self.cfg.ocr_source)
        log.info("  OUTPUT_DIR : %s", self.cfg.output_dir)
        log.info("  PREFIX     : %s", self.cfg.file_prefix)
        log.info("  DRY_RUN    : %s", self.cfg.dry_run)
        log.info("=" * 60)

        chunks = load_ocr_chunks(self.cfg.ocr_source, self.cfg.min_chunk_chars)
        with_section = sum(1 for c in chunks if c.section)
        log.info("Chunks: raw usable=%d with_section=%d", len(chunks), with_section)

        clusters = build_clusters(chunks, self.cfg.max_clusters)
        cluster_payload = [
            {
                "cluster_id": c.cluster_id,
                "parent_section": c.parent_section,
                "sections": c.sections,
                "chunk_ids": [ch.chunk_id for ch in c.chunks],
                "preview": [ch.content[:120] for ch in c.chunks],
            }
            for c in clusters
        ]
        self.clusters_path.write_text(
            json.dumps(cluster_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Wrote %d clusters -> %s", len(clusters), self.clusters_path)

        if self.cfg.dry_run:
            log.info("MH_DRY_RUN=1，跳过 LLM 生成。")
            return {"clusters": len(clusters), "passed": 0, "dry_run": True}

        done_ids: set[str] = set()
        passed_rows: list[dict] = []
        raw_rows: list[dict] = []

        if self.cfg.resume and self.raw_path.is_file():
            raw_rows = json.loads(self.raw_path.read_text(encoding="utf-8"))
            done_ids = {r["cluster_id"] for r in raw_rows if r.get("cluster_id")}
            log.info("Resume: 已有 %d 条 raw 记录", len(done_ids))

        if self.cfg.resume and self.sft_path.is_file():
            for ln in self.sft_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    passed_rows.append(json.loads(ln))

        llm = self._get_llm()

        for ci, cluster in enumerate(clusters):
            if cluster.cluster_id in done_ids:
                continue
            log.info("[%d/%d] cluster=%s sections=%s", ci + 1, len(clusters), cluster.cluster_id, cluster.sections)
            context = merge_cluster_context(
                cluster, self.cfg.max_fragment_chars, self.cfg.max_context_chars
            )
            t0 = time.time()

            # Step A（含 repair + 可选重试）
            step_a_user = f"文档片段：\n\n{context}\n\n请抽取跨片段依赖图 JSON。"
            raw_a = llm.generate_from_input([step_a_user], system_prompt=STEP_A_SYSTEM)[0]
            graph = _parse_json_from_llm(raw_a or "")
            if graph and not graph.get("skip"):
                graph = repair_dependency_graph(graph, cluster)
            ok, reason = validate_dependency_graph(graph or {}, cluster) if graph else (False, "L0_stepA_json")

            for attempt in range(self.cfg.max_llm_retries):
                if ok:
                    break
                log.info("  StepA retry %d: %s", attempt + 1, reason)
                retry_user = step_a_user + STEP_A_RETRY_SUFFIX.format(reason=reason)
                raw_a = llm.generate_from_input([retry_user], system_prompt=STEP_A_SYSTEM)[0]
                graph = _parse_json_from_llm(raw_a or "")
                if graph and not graph.get("skip"):
                    graph = repair_dependency_graph(graph, cluster)
                ok, reason = (
                    validate_dependency_graph(graph, cluster)
                    if graph
                    else (False, "L0_stepA_json")
                )

            if not ok:
                key = reason.split(":")[0] if reason else "L0_stepA_json"
                self.reject_stats[key] += 1
                raw_rows.append(self._raw_fail(cluster, context, raw_a, "", reason))
                self._flush_raw(raw_rows)
                continue

            # Step B（含可选重试）
            ent_lines = []
            entities = {str(e.get("id")): e for e in (graph.get("entities") or []) if isinstance(e, dict)}
            for eid in graph.get("reasoning_path") or []:
                ent = entities.get(str(eid), {})
                ent_lines.append(f"- {eid} (片段{ent.get('fragment')}): {ent.get('fact', '')}")
            path_type = graph.get("path_type") or "流程链"
            step_b_user = (
                f"片段数：{cluster.size}\n路径类型：{path_type}\n"
                f"依赖链：\n" + "\n".join(ent_lines) + "\n\n"
                f"参考上下文：\n{context}\n\n请生成多跳 QA JSON。"
            )
            raw_b = llm.generate_from_input([step_b_user], system_prompt=STEP_B_SYSTEM)[0]
            qa = _parse_json_from_llm(raw_b or "")

            for attempt in range(self.cfg.max_llm_retries):
                if qa:
                    break
                log.info("  StepB retry %d: JSON parse failed", attempt + 1)
                retry_user = step_b_user + STEP_B_RETRY_SUFFIX
                raw_b = llm.generate_from_input([retry_user], system_prompt=STEP_B_SYSTEM)[0]
                qa = _parse_json_from_llm(raw_b or "")

            if not qa:
                self.reject_stats["L0_stepB_json"] += 1
                raw_rows.append(self._raw_fail(cluster, context, raw_a, raw_b, "L0_stepB_json"))
                self._flush_raw(raw_rows)
                continue

            qa["_entities"] = graph.get("entities") or []
            qa["type"] = qa.get("type") or path_type
            qa["reasoning_path"] = qa.get("reasoning_path") or graph.get("reasoning_path")

            ok, reason = validate_structure(qa, cluster)
            if not ok:
                self.reject_stats[reason.split(":")[0]] += 1
                raw_rows.append(self._raw_fail(cluster, context, raw_a, raw_b, reason))
                self._flush_raw(raw_rows)
                continue

            ok, reason = validate_cross_source(qa, cluster)
            if not ok:
                self.reject_stats[reason.split(":")[0]] += 1
                raw_rows.append(self._raw_fail(cluster, context, raw_a, raw_b, reason))
                self._flush_raw(raw_rows)
                continue

            if self.cfg.enable_degeneracy:
                ok, reason = validate_degeneracy(qa, cluster)
                if not ok:
                    self.reject_stats[reason.split(":")[0]] += 1
                    raw_rows.append(
                        self._raw_fail(cluster, context, raw_a, raw_b, reason)
                    )
                    self._flush_raw(raw_rows)
                    continue

            score = compute_complexity(qa, cluster)
            if score < self.cfg.min_complexity:
                self.reject_stats["L4_low_complexity"] += 1
                raw_rows.append(self._raw_fail(cluster, context, raw_a, raw_b, f"L4:{score}", qa))
                self._flush_raw(raw_rows)
                continue

            record = QARecord(
                cluster_id=cluster.cluster_id,
                question=str(qa["question"]).strip(),
                answer=str(qa["answer"]).strip(),
                input_context=context,
                reasoning_steps=qa.get("reasoning_steps") or [],
                supporting_facts=qa.get("supporting_facts") or [],
                reasoning_path=[str(x) for x in (qa.get("reasoning_path") or [])],
                multihop_type=str(qa.get("type") or path_type),
                complexity_score=score,
                source_sections=cluster.sections,
                source_chunk_ids=[ch.chunk_id for ch in cluster.chunks],
                dataset=self.cfg.dataset_label,
                dependency_graph=graph,
                raw_step_a=raw_a or "",
                raw_step_b=raw_b or "",
                raw_degeneracy=qa.get("_deg_raws") or [],
            )
            sft_row = {
                "instruction": record.question,
                "input": record.input_context,
                "output": record.answer,
                "question_type": "多跳",
                "multihop_type": record.multihop_type,
                "reasoning_steps": record.reasoning_steps,
                "supporting_facts": record.supporting_facts,
                "reasoning_path": record.reasoning_path,
                "complexity_score": record.complexity_score,
                "source_sections": record.source_sections,
                "source_chunk_ids": record.source_chunk_ids,
                "dataset": record.dataset,
            }
            passed_rows.append(sft_row)
            raw_rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "status": "passed",
                    "complexity_score": score,
                    "sft": sft_row,
                    "dependency_graph": graph,
                    "raw_step_a": raw_a,
                    "raw_step_b": raw_b,
                    "raw_degeneracy": record.raw_degeneracy,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
            )
            self._flush_raw(raw_rows)
            self._flush_sft(passed_rows)
            log.info("  PASS complexity=%.3f question=%s...", score, record.question[:60])

        log.info("=" * 60)
        log.info("完成: passed=%d / clusters=%d", len(passed_rows), len(clusters))
        log.info("拒绝统计: %s", dict(self.reject_stats))
        log.info("SFT -> %s", self.sft_path)
        log.info("=" * 60)
        return {
            "clusters": len(clusters),
            "passed": len(passed_rows),
            "rejects": dict(self.reject_stats),
        }

    def _raw_fail(
        self,
        cluster: Cluster,
        context: str,
        raw_a: str,
        raw_b: str,
        reason: str,
        deg: Any = None,
    ) -> dict:
        return {
            "cluster_id": cluster.cluster_id,
            "status": "rejected",
            "reason": reason,
            "sections": cluster.sections,
            "chunk_ids": [c.chunk_id for c in cluster.chunks],
            "context_preview": context[:500],
            "raw_step_a": raw_a,
            "raw_step_b": raw_b,
            "raw_degeneracy": deg if isinstance(deg, list) else [],
        }

    def _flush_raw(self, rows: list[dict]) -> None:
        self.raw_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _flush_sft(self, rows: list[dict]) -> None:
        self.sft_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )


def main() -> None:
    cfg = MHConfig.from_env()
    if not cfg.ocr_source.is_file():
        print(f"[ERROR] OCR_SOURCE 不存在: {cfg.ocr_source}", file=sys.stderr)
        sys.exit(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result = MultiHopPipeline(cfg).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
