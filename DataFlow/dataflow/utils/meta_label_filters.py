"""共享的元信息 / OUTOFSCOPE 过滤常量与工具函数。

被 `answer_rewriter.py` / `answer_refiner.py` / 上层 pipeline (test_filter.py) 共用，
避免同样的禁词清单 / 正则散落在多处。

- `is_outofscope_or_meta_tox(text)`：行级硬 drop 判定（OUTOFSCOPE token 或元信息托词短开头）
- `scrub_meta_labels(text)`：句级清洗，删除内部标签泄漏 / 元信息引用整句
- `has_reverse_phrasing(text)`：检测"应检查并重新确认 ... 是否正确"反向表述（仅告警）
"""
from __future__ import annotations

import re

__all__ = [
    "OUTOFSCOPE_TOKEN",
    "META_TOX_PREFIX_REGEX",
    "SCRUB_FORBIDDEN_REGEX",
    "REVERSE_PHRASING_REGEX",
    "is_outofscope_or_meta_tox",
    "extract_llm_answer_text",
    "extract_improved_question_from_refiner",
    "is_valid_export_question",
    "normalize_export_question",
    "has_refiner_leak",
    "is_valid_export_output",
    "scrub_meta_labels",
    "has_reverse_phrasing",
    "strip_training_decorators",
    "looks_like_fabricated_shell_block",
    "looks_like_answer_as_quote",
    "detect_question_answer_polarity_issue",
    "detect_evaluator_tone",
    "strip_evaluator_tone",
]


OUTOFSCOPE_TOKEN = "OUTOFSCOPE"

# 元信息托词开头：以这些前缀开头且整段较短的答案视为无效
META_TOX_PREFIX_REGEX = re.compile(
    r"^\s*(根据(?:参考)?(?:内容|文档|资料|信息|原文|上述|提供|上下文)|参考(?:内容|文档|资料|信息|原文|上下文)(?:明确)?(?:指出|未|没有|显示|要求)|"
    r"原文(?:指出|明确指出|未|没有|提到|中)|按照(?:参考)?(?:内容|文档|资料|信息|原文)|参考(?:上文|以下)|当前材料)",
)
META_TOX_LEN_THRESHOLD = 80

# 句级清洗：命中以下任一短语的整句 (句号 / 分号 / 换行边界) 将被剔除
_SCRUB_FORBIDDEN_PATTERNS = [
    # 内部标签词泄漏（最优先）
    r"事实锚答案",
    r"事实锚",
    r"事实基准",
    r"Fact\s*Baseline",
    r"Target\s*Answer",
    r"精炼答案",
    r"事实集封闭",
    r"Fact[-\s]*Set\s*Closure",
    r"OUTOFSCOPE",
    r"批评反馈",
    r"\bCritique\b",
    # 元信息引用前缀
    r"根据参考内容",
    r"参考内容明确指出",
    r"参考内容要求",
    r"参考内容显示",
    r"参考内容(?:未|没有)(?:提及|明确)?",
    r"参考上下文",
    r"参考(?:文档|资料|信息|文件|上文|上述|以下)",
    r"根据参考(?:内容|文档|资料|信息|文件)?",
    r"根据提供(?:的参考)?(?:内容|文档|资料|信息|文件)?",
    r"根据给定(?:的参考)?(?:内容|文档|资料|信息|文件)?",
    r"原文明确指出",
    r"原文指出",
    r"原文要求",
    r"原文提到",
    r"根据原文",
    r"按照原文",
    r"原文中",
    r"根据文档",
    r"根据流程图",
    r"根据表格",
    r"当前材料",
    r"给定(?:材料|上下文|文档)",
]
SCRUB_FORBIDDEN_REGEX = re.compile("|".join(_SCRUB_FORBIDDEN_PATTERNS))

# 反向表述：'应检查并重新确认 ... 是否正确' 这种回避型句式 → 仅打 warning
REVERSE_PHRASING_REGEX = re.compile(
    r"应检查并重新确认[^。；\n]*?是否(?:正确|准确|有效)"
)


_THINKING_ANSWER_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_ONLY_RE = re.compile(
    r"<think>.*?</think>\s*",
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)


def extract_llm_answer_text(text) -> str:
    """从 reasoning 模型输出中提取 ``<answer>`` 正文，去掉 ``<think>`` 块。"""
    if text is None:
        return ""
    s = str(text)
    m = _THINKING_ANSWER_RE.search(s)
    if m:
        return m.group(1).strip()
    s = _THINKING_ONLY_RE.sub("", s).strip()
    s = _ANSWER_TAG_RE.sub("", s).strip()
    return s


_GARBAGE_QUESTION_PREFIX = "`, `...`, `"
_BAD_QUESTION_EXACT = frozenset({
    "`, `...`, `",
    "...",
    ", ...,",
    "改进后的问题",
    "你改进后的问题",
    "改进后问题...",
    "`, `...`, and `",
    "\\n...\\n",
})
_REFINER_LEAK_PATTERNS = [
    re.compile(r"\[Analysis Start\]", re.I),
    re.compile(r"\[Improved Question Start\]", re.I),
    re.compile(r"\[Fact Closure Check", re.I),
    re.compile(r"\[Critique Start\]", re.I),
    re.compile(r"Fact Closure Check JSON", re.I),
    re.compile(r"granularity_match", re.I),
    re.compile(r"scenario_preservation", re.I),
    re.compile(r"all_constraints_in_answer_or_context", re.I),
    re.compile(r"\bI will (output|generate|analyze|formulate|structure)\b", re.I),
    re.compile(r"Let's (check|double|verify|analyze)", re.I),
    re.compile(r"\bI'll output\b", re.I),
]
_IMPROVED_Q_CLOSED = re.compile(
    r"\[Improved Question Start\]\s*(.*?)\s*\[Improved Question End\]",
    re.DOTALL | re.IGNORECASE,
)
_IMPROVED_Q_OPEN = re.compile(
    r"\[Improved Question Start\]\s*(.*?)(?:\s*\[Fact Closure Check|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_PROMPT_PLACEHOLDER_QUESTIONS = frozenset({
    "...",
    "你改进后的问题",
    "改进后的问题",
    "改进后问题...",
})


def _meaningful_char_count(text: str) -> int:
    return sum(1 for c in text if c.isalnum() or ("\u4e00" <= c <= "\u9fff"))


def has_refiner_leak(text: str) -> bool:
    if not text:
        return True
    for pat in _REFINER_LEAK_PATTERNS:
        if pat.search(text):
            return True
    ascii_words = len(re.findall(r"\b[a-zA-Z]{4,}\b", text))
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if len(text) > 150 and ascii_words >= 12 and cn < 60:
        return True
    return False


def extract_improved_question_from_refiner(text: str) -> str | None:
    """从 QuestionRefiner / QuestionRefinePrompt 的 LLM 响应中提取改进后问题。"""
    if not text:
        return None
    response = extract_llm_answer_text(text) or str(text).strip()
    if not response:
        return None

    candidates: list[str] = []

    for pat in (
        _IMPROVED_Q_CLOSED,
        _IMPROVED_Q_OPEN,
        re.compile(
            r"【Improved Question Start】\s*(.*?)\s*【Improved Question End】",
            re.DOTALL | re.IGNORECASE,
        ),
    ):
        m = pat.search(response)
        if m:
            candidates.append(m.group(1).strip().strip('"'))

    # 仅当 marker 前存在完整中文问句、且 marker 后主要是推理泄漏时，保留 marker 前内容
    for sep in (
        "\n[Analysis Start]",
        "\n[Improved Question Start]",
        "[Analysis Start]",
        "[Improved Question Start]",
        "[Fact Closure Check Start]",
        "\n\n请从下列选项中选择正确的一项",
    ):
        if sep in response:
            head = response.split(sep)[0].strip().lstrip("`").strip(" ,:")
            if head:
                candidates.append(head)

    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in _PROMPT_PLACEHOLDER_QUESTIONS:
            continue
        if cand.startswith(_GARBAGE_QUESTION_PREFIX):
            cand = cand[len(_GARBAGE_QUESTION_PREFIX) :].lstrip("\n")
        if is_valid_export_question(cand):
            return cand
    return None


def is_valid_export_question(question: str) -> bool:
    """判断问题是否可作为 SFT instruction 导出（过滤 Refiner 泄漏与占位符）。"""
    inst = (question or "").strip()
    if not inst or inst in _BAD_QUESTION_EXACT:
        return False
    if inst.startswith(_GARBAGE_QUESTION_PREFIX):
        inst = inst[len(_GARBAGE_QUESTION_PREFIX) :].lstrip("\n")
    if not inst or inst in _BAD_QUESTION_EXACT:
        return False
    if "改进后的问题" in inst and _meaningful_char_count(inst) < 20:
        return False
    if has_refiner_leak(inst):
        return False
    if _meaningful_char_count(inst) < 8 or len(inst) < 10:
        return False
    if inst.count("`") >= 3 and _meaningful_char_count(inst) < 15:
        return False
    if re.fullmatch(r"[`\s\.,;:\-…\.\n\\]+", inst):
        return False
    return True


def normalize_export_question(question: str) -> str | None:
    """清洗并规范化可导出的 instruction；无效时返回 None。"""
    if not question:
        return None
    salvaged = extract_improved_question_from_refiner(question)
    if salvaged and is_valid_export_question(salvaged):
        return salvaged
    inst = question.strip()
    if inst.startswith(_GARBAGE_QUESTION_PREFIX):
        inst = inst[len(_GARBAGE_QUESTION_PREFIX) :].lstrip("\n")
    if is_valid_export_question(inst):
        return inst
    return None


_BAD_OUTPUT_EXACT = frozenset({"...", "…", "...."})


def is_valid_export_output(output: str) -> bool:
    out = (output or "").strip()
    if not out or out in _BAD_OUTPUT_EXACT:
        return False
    if _meaningful_char_count(out) < 6 and len(out) < 20:
        return False
    if re.fullmatch(r"[\.\s…,，;；]+", out):
        return False
    return True


def is_outofscope_or_meta_tox(text) -> tuple[bool, str]:
    """判断答案是否属于"无效托词答案"，应被上游 DROP。

    规则：
    1) 命中 ``OUTOFSCOPE`` token → DROP
    2) 以 :data:`META_TOX_PREFIX_REGEX` 开头且整段长度 < :data:`META_TOX_LEN_THRESHOLD`
       → DROP（"参考内容未提及…"这类无效答案）

    Returns:
        ``(should_drop, reason)``，``reason`` ∈ ``{"", "outofscope", "meta_tox_prefix"}``
    """
    if text is None:
        return False, ""
    s = str(text).strip()
    if not s:
        return False, ""
    if OUTOFSCOPE_TOKEN in s.upper():
        return True, "outofscope"
    if META_TOX_PREFIX_REGEX.match(s) and len(s) < META_TOX_LEN_THRESHOLD:
        return True, "meta_tox_prefix"
    return False, ""


def scrub_meta_labels(text) -> tuple[str, list[str]]:
    """从答案中按句剔除内部标签 / 元信息引用句。

    切句规则：句号 / 分号 / 感叹号 / 问号 / 换行作为句界。命中任意 :data:`SCRUB_FORBIDDEN_REGEX`
    模式的句子整句剔除。

    Returns:
        ``(cleaned_text, hit_phrases)``，``hit_phrases`` 用于 logging / 统计。
    """
    if not isinstance(text, str) or not text.strip():
        return text or "", []
    sentences = re.split(r"(?<=[。；;！\!\?？\n])", text)
    kept: list[str] = []
    hits: list[str] = []
    for sent in sentences:
        if not sent.strip():
            kept.append(sent)
            continue
        m = SCRUB_FORBIDDEN_REGEX.search(sent)
        if m:
            hits.append(m.group(0))
            continue
        kept.append(sent)
    cleaned = "".join(kept).strip()
    return cleaned, hits


def has_reverse_phrasing(text) -> bool:
    """是否含有 "应检查并重新确认 ... 是否正确" 这种回避型反向表述。"""
    if not isinstance(text, str) or not text:
        return False
    return REVERSE_PHRASING_REGEX.search(text) is not None


# --- 训练/评测装饰符 strip（✅ ⚠️ 等），与 scrub_meta_labels 正交 ---
_LINE_LEADING_EMOJI = re.compile(
    r"(?m)^[\s\u200b\u200c\u200d\ufeff]*"
    r"[✅✔☑⚠⚠️🔴📌❗❌▶►]+[\s\u200b]*"
)
# 「✅ 验证方法：」等装饰性小节前缀（保留冒号后的实质内容）
_FANCY_SECTION_PREFIX = re.compile(
    r"(?m)^\s*(?:[✅✔☑⚠⚠️🔴📌❗❌]+\s*)?"
    r"(验证方法|局限说明|优先操作|应急处置流程?|备注)\s*[：:]\s*"
)

def strip_training_decorators(text: str) -> tuple[str, list[str]]:
    """去掉 ✅/⚠️ 等装饰符与非用户要求的栏目式抬头，避免 SFT / RAG 学坏。

    Returns:
        ``(cleaned, stripped_tokens)`` 用于日志
    """
    if not isinstance(text, str) or not text.strip():
        return text or "", []
    hits: list[str] = []
    s = text
    for m in _LINE_LEADING_EMOJI.finditer(s):
        hits.append((m.group(0) or "")[:16])
    s = _LINE_LEADING_EMOJI.sub("", s)
    s2 = _FANCY_SECTION_PREFIX.sub("", s)
    if s2 != s:
        hits.append("[fancy_section_prefix]")
        s = s2
    s = re.sub(r"[✅⚠✔☑📌❗❌🔴▶►]{2,}", "", s)
    cleaned = "\n".join(ln.rstrip() for ln in s.split("\n")).strip()
    return cleaned, hits[:20]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _split_content_units(text: str) -> list[str]:
    """Split Chinese/English technical prose into reusable quote-detection units."""
    if not isinstance(text, str):
        return []
    parts = re.split(r"[。；;！？\n，、,：:]+|(?:\s*-\s*)", text)
    return [p.strip() for p in parts if len(_compact_text(p)) >= 6]


def looks_like_answer_as_quote(answer: str, context: str) -> bool:
    """Detect high verbatim overlap with context, a proxy for answer-as-quote dumping.

    This intentionally requires several long copied units or very high copied coverage
    to avoid flagging short, legitimate command/path answers.

    Thresholds are tuned so that answers which largely restate procedure lines from the
    same doc (high overlap but still on-topic) trigger less often; pure chunk-dump
    patterns still trip when coverage is very high.
    """
    if not isinstance(answer, str) or not isinstance(context, str):
        return False
    ans = answer.strip()
    ctx_compact = _compact_text(context)
    ans_compact = _compact_text(ans)
    # 太短：多半是结论句或单条命令，跳过高重叠告警
    if len(ans_compact) < 180 or len(ctx_compact) < len(ans_compact):
        return False

    copied_len = 0
    copied_units = 0
    for unit in _split_content_units(ans):
        unit_compact = _compact_text(unit)
        if unit_compact and unit_compact in ctx_compact:
            copied_units += 1
            copied_len += len(unit_compact)

    copied_ratio = copied_len / max(len(ans_compact), 1)
    if copied_units >= 6 and copied_ratio >= 0.48:
        return True
    if copied_units >= 4 and copied_ratio >= 0.58:
        return True
    if copied_units >= 3 and copied_ratio >= 0.76:
        return True
    return False


_AVOID_QUESTION_REGEX = re.compile(r"(应避免|避免|禁止|不得|切勿|严禁|请勿|不能|不应)")
_DIRECT_AVOID_QUESTION_REGEX = re.compile(
    r"(?:应避免|避免|禁止|不得|切勿|严禁|请勿|不能|不应)"
    r"[^？?。；;\n]{0,24}(?:哪|什么|哪些|何种|执行|操作|动作|事项|行为)"
)
# 「不能忽略哪些 / 不应遗漏什么」类：期望正向列举检查项，不要求答案再堆「禁止/切勿」措辞
_FALSE_AVOID_POLARITY_QUESTION_REGEX = re.compile(
    r"(?:不能|不应|切勿|勿|避免)(?:忽略|遗漏|跳过|缺失|省去)|"
    r"(?:需|需要|应|必须)(?:关注|检查|核实|包含|补齐)(?:哪|什么|哪些)"
)
_PURPOSE_AVOID_REGEX = re.compile(r"为(?:了)?避免|以避免|才能避免")
_AVOID_ANSWER_REGEX = re.compile(r"(应避免|避免|禁止|不得|切勿|严禁|请勿|不能|不应|不要)")
_PREFER_ONLY_REGEX = re.compile(r"^\s*(优先操作|最应优先|应优先|优先|首先|先)")
_CAN_QUESTION_REGEX = re.compile(r"(能否|是否可以|可否|可以.*吗|能不能)")
_CAN_ANSWER_REGEX = re.compile(
    r"(可以|不能|不可以|不得|禁止|不应|请勿|无法|不建议|可重试|不可重试|能重试|不能重试|即可)"
)


# 「若不能，应如何」类条件句里的「不能」不是禁止类发问，避免误触发 avoid 极性检查
_CONDITIONAL_IF_CANNOT_SUB = re.compile(r"(?:若|如果|如其?)不能\s*[，,、]?\s*")


def detect_question_answer_polarity_issue(question: str, answer: str) -> str:
    """Return issue code when answer polarity likely conflicts with the question."""
    if not isinstance(question, str) or not isinstance(answer, str):
        return ""
    q = question.strip()
    a = answer.strip()
    if not q or not a:
        return ""
    q_for_avoid = _CONDITIONAL_IF_CANNOT_SUB.sub("", q)
    if _FALSE_AVOID_POLARITY_QUESTION_REGEX.search(q):
        # 与「应避免哪一步」不同：此类问法答案常用「应检查…」，不做否定极性强检
        pass
    elif _AVOID_QUESTION_REGEX.search(q_for_avoid):
        q_direct = re.sub(
            r"(?:为(?:了)?避免|以避免|才能避免)[^？?，,。；;\n]*",
            "",
            q_for_avoid,
        )
        # "为避免业务中断，应如何..." 是目的状语，答案给正向动作即可；
        # 只有直接询问"应避免/禁止哪项操作"时才做负向极性检查。
        if _PURPOSE_AVOID_REGEX.search(q) and not _DIRECT_AVOID_QUESTION_REGEX.search(q_direct):
            pass
        elif not _DIRECT_AVOID_QUESTION_REGEX.search(q_direct):
            pass
        else:
            if _PREFER_ONLY_REGEX.search(a) and not _AVOID_ANSWER_REGEX.search(a[:160]):
                return "avoid_question_answered_as_prefer"
            if not _AVOID_ANSWER_REGEX.search(a[:220]):
                return "avoid_question_without_negative_answer"
    if _CAN_QUESTION_REGEX.search(q) and not _CAN_ANSWER_REGEX.search(a[:160]):
        return "can_question_without_can_cannot"
    return ""


_EVALUATOR_TONE_REGEX = re.compile(
    r"(最可能|最应优先|应优先(?:检查|执行|收集|确认)以下|"
    r"当前维护流程中可能存在以下违反|可能存在以下违反|"
    r"评分[：:]|不通过[：:]|需要改进[：:])"
)


def detect_evaluator_tone(answer: str) -> list[str]:
    """Detect evaluator / grading wording leaked into the training answer."""
    if not isinstance(answer, str) or not answer:
        return []
    return [m.group(0) for m in _EVALUATOR_TONE_REGEX.finditer(answer)]


_EVALUATOR_TONE_REPLACEMENTS: tuple[tuple[re.Pattern, str], ...] = (
    # Critique 残留（常见于模型把评分头带入答案）
    (re.compile(r"评分[：:][^。；;\n]+(?:[。；;\n]|$)"), ""),
    (re.compile(r"不通过[：:][^。；;\n]+(?:[。；;\n]|$)"), ""),
    (re.compile(r"需要改进[：:][^。；;\n]+(?:[。；;\n]|$)"), ""),
    (re.compile(r"最可能的原因是"), "原因是"),
    (re.compile(r"最可能被忽略的"), "被忽略的"),
    (re.compile(r"最可能与"), "与"),
    (re.compile(r"最可能由"), "由"),
    (re.compile(r"最可能对应"), "对应"),
    (re.compile(r"最可能"), ""),
    (re.compile(r"最应优先"), "应"),
    (re.compile(r"应优先(检查|执行|收集|确认)以下"), r"应\1以下"),
    (
        re.compile(r"当前维护流程中可能存在以下违反[^：:\n]*[：:]?"),
        "需要处理以下不符合要求的操作点：",
    ),
    (
        re.compile(r"可能存在以下违反[^：:\n]*[：:]?"),
        "存在以下不符合要求的操作点：",
    ),
)


def strip_evaluator_tone(text: str) -> tuple[str, list[str]]:
    """去掉/中性化答案中的评估腔、排序腔表达。"""
    if not isinstance(text, str) or not text:
        return text or "", []
    hits = detect_evaluator_tone(text)
    if not hits:
        return text, []
    s = text
    for pattern, repl in _EVALUATOR_TONE_REPLACEMENTS:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+([，。；：])", r"\1", s)
    s = re.sub(r"^[，、；：\s]+", "", s.strip())
    return s, hits[:20]


_SHELL_CONSTRUCT_REGEX = re.compile(
    r"(?:^|\n)\s*(?:for\b|while\b|(?:if\b.*;\s*)?then\b|;\s*do\b)",
    re.IGNORECASE | re.MULTILINE,
)
_AWK_TOKEN = re.compile(r"\bawk\b", re.IGNORECASE)
_GREP_E_FLAG = re.compile(r"grep\s+-E\b", re.IGNORECASE)


def looks_like_fabricated_shell_block(answer: str, anchor: str) -> bool:
    """若答案出现较复杂 shell 结构，而参考答案中不出现同类 token，视作疑似编造脚本。

    覆盖：for/while/do 多行结构、answer 含 ``awk`` / ``grep -E`` 而 anchor 不含（文档常仅一条简单命令）。
    """
    if not isinstance(answer, str) or not isinstance(anchor, str):
        return False
    if not answer.strip():
        return False
    if _SHELL_CONSTRUCT_REGEX.search(answer) and not _SHELL_CONSTRUCT_REGEX.search(anchor):
        return True
    if _AWK_TOKEN.search(answer) and not _AWK_TOKEN.search(anchor):
        return True
    if _GREP_E_FLAG.search(answer) and not _GREP_E_FLAG.search(anchor):
        return True
    return False
