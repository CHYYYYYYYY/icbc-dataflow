"""
Prompts and helpers for technical-document QA pipeline (grounding, rewrite, reverse Q, rubric).
"""
from __future__ import annotations

import json
import re
from typing import Any

from dataflow.core.prompt import PromptABC
from dataflow.utils.meta_label_filters import extract_llm_answer_text
from dataflow.utils.registry import PROMPT_REGISTRY


def _try_parse_json_string(s: str) -> dict[str, Any] | None:
    """Parse a JSON object from a string that may include markdown fences."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            out = json.loads(s[i : j + 1])
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_json_llm_response(text: str | None) -> dict[str, Any] | None:
    """Strip thinking tags / markdown fences and parse JSON object from model output."""
    if not text or not isinstance(text, str):
        return None
    # reasoning 模型常在 <think> 里含 { }，直接 brace-slice 会误解析失败 → 默认 DROP
    candidates = [
        extract_llm_answer_text(text),
        re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip(),
        text.strip(),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result = _try_parse_json_string(candidate)
        if result:
            return result
    return None


def _norm_verdict(v: str | None, allowed: set[str], default: str) -> str:
    if not v or not isinstance(v, str):
        return default
    x = v.strip().upper().replace(" ", "")
    if x in allowed:
        return x
    return default


@PROMPT_REGISTRY.register()
class AnswerGroundingFilterPrompt(PromptABC):
    """Answer grounding: KEEP / REWRITE / DROP + evidence JSON."""

    def build_prompt(self, context: str, question: str, answer: str) -> str:
        return f"""你是一个严格的技术文档QA质量审核专家。

请判断以下「答案」是否真正来自「参考原文」，并给出处理建议。

## 参考原文：
{context}

## 问题：
{question}

## 待审核答案：
{answer}

---

## 审核标准：

### 直接判 DROP（丢弃）：
- 答案出现："参考内容未提供"、"无法根据本文档回答"、
  "缺乏规范依据"、"原文未提及"、"无法确定"等表述
- 答案核心事实与原文明确矛盾
- 答案引入了原文完全不存在的概念作为核心答案

### 判 REWRITE（需要改写）：
- 答案方向正确，但加入了原文不支持的具体细节
- 答案在原文基础上过度扩写

### 判 KEEP（保留）：
- 答案内容可以在原文中找到明确对应
- 没有添加原文不存在的信息
- 简洁准确，与原文高度一致

## 输出格式（严格JSON）：
{{
  "verdict": "KEEP / REWRITE / DROP",
  "reason": "一句话说明判断原因",
  "problematic_parts": ["有问题的片段1", "片段2"],
  "groundable_parts": ["有原文支撑的部分"],
  "anchor_sentences": ["原文中对应的逐字支撑句1", "支撑句2"]
}}
"""


@PROMPT_REGISTRY.register()
class AnswerRewriterPrompt(PromptABC):
    """Rewrite answer strictly from source; output REWRITTEN / DROP JSON."""

    def build_prompt(
        self,
        context: str,
        question: str,
        answer: str,
        grounding_result: dict[str, Any],
    ) -> str:
        gr = grounding_result or {}
        return f"""你是一个技术文档答案改写专家。

任务：将以下答案改写为严格基于原文的版本。

## 参考原文：
{context}

## 问题：
{question}

## 原始答案：
{answer}

## 审核发现的问题：
- 有问题的部分：{gr.get("problematic_parts", [])}
- 原文中有支撑的部分：{gr.get("groundable_parts", [])}
- 原文对应支撑句：{chr(10).join(f"  • {s}" for s in (gr.get("anchor_sentences") or [])) or gr.get("original_evidence", "（无）")}

## 改写规则：

**必须做到：**
1. 每一句话都必须能在原文中找到直接依据
2. 用原文的表述方式，可适当精简，不能改变意思
3. 答案不超过原文对应内容的1.5倍长度

**绝对禁止：**
1. 禁止出现："参考内容未提供"、"无法根据文档回答"等边界声明
2. 禁止添加原文中没有的具体数字、命令、步骤
3. 禁止基于"常识"或"行业经验"补充信息

**特殊情况：**
- 原文完全找不到对应内容 → verdict: "DROP"，不改写
- 原文只有部分答案 → 只回答有依据的部分，其余不提

## 输出格式（严格JSON）：
{{
  "verdict": "REWRITTEN / DROP",
  "rewritten_answer": "改写后的答案（DROP时为null）",
  "anchor_sentences": ["原文支撑句1（逐字引用）", "句子2"],
  "dropped_content": "删除了什么内容及原因"
}}
"""



@PROMPT_REGISTRY.register()
class RubricScorerPrompt(PromptABC):
    """HC veto + SC scores + optional OC; JSON output."""

    HARD_CONSTRAINTS_BY_TYPE: dict[str, list[tuple[str, str]]] = {
        "命令查询": [
            ("HC-1", "仅当问题要求给出可执行命令时，答案中的命令字符串必须正确"),
            ("HC-2", "答案必须直接回应问题，不能答非所问"),
            ("HC-3", "答案不包含原文中不存在的虚构命令或参数"),
        ],
        "参数确认": [
            ("HC-1", "仅当问题要求确认参数或数值时，答案中的参数值必须正确"),
            ("HC-2", "仅当问题涉及单位或适用条件时，答案必须说明对应单位或适用条件"),
            ("HC-3", "答案不包含原文不支持的固定值"),
        ],
        "故障处理": [
            ("HC-1", "仅当问题要求处理路径时，答案中的关键处理路径不能错误"),
            ("HC-2", "答案必须直接对应问题描述的故障场景"),
            ("HC-3", "答案不包含原文不存在的处理方法"),
        ],
        "差异对比": [
            ("HC-1", "仅当问题是对比题时，答案不能混淆被比较对象"),
            ("HC-2", "仅当问题明确列出多个对比维度时，答案不能遗漏核心对比维度"),
            ("HC-3", "答案不包含原文不支持的差异描述"),
        ],
        "版本特性": [
            ("HC-1", "仅当问题涉及版本范围时，答案不能错误扩大或缩小版本范围"),
            ("HC-2", "仅当问题询问特性表现或操作方式时，答案不能答非所问"),
            ("HC-3", "答案不包含该版本范围之外的特性描述"),
        ],
        "default": [
            ("HC-1", "答案内容与标准答案在核心事实上一致"),
            ("HC-2", "答案直接回应问题，不答非所问"),
            ("HC-3", "答案不包含原文不存在的虚构信息"),
        ],
    }

    OPTIONAL_CHECKS_BY_TYPE: dict[str, list[tuple[str, str]]] = {
        "故障处理": [
            ("OC-1", "是否说明了适用场景、前置条件或风险提示"),
            ("OC-2", "是否给出了恢复验证方法"),
            ("OC-3", "是否补充了后续排查或收尾步骤"),
        ],
        "参数确认": [
            ("OC-1", "是否说明了参数的适用范围"),
            ("OC-2", "是否说明了单位、默认值或生效条件"),
            ("OC-3", "是否给出了验证方式"),
        ],
        "命令查询": [
            ("OC-1", "是否说明了命令的适用场景"),
            ("OC-2", "是否补充了注意事项或前置条件"),
            ("OC-3", "是否给出了验证方法或结果判断方式"),
        ],
        "差异对比": [
            ("OC-1", "是否补充了适用范围或版本边界"),
            ("OC-2", "是否解释了差异成立的条件"),
            ("OC-3", "是否给出了对比后的使用建议"),
        ],
        "版本特性": [
            ("OC-1", "是否说明了版本适用范围"),
            ("OC-2", "是否补充了启用前提或限制条件"),
            ("OC-3", "是否给出了验证方式"),
        ],
        "default": [
            ("OC-1", "是否说明了适用范围"),
            ("OC-2", "是否提供了注意事项或前置条件"),
            ("OC-3", "是否给出了验证方法或后续步骤"),
        ],
    }

    def build_prompt(
        self,
        context: str,
        question: str,
        answer: str,
        standard_answer: str,
        question_type: str = "default",
        anchor_sentences: list[str] | None = None,
    ) -> str:
        qc = (question_type or "default").strip()
        if qc not in self.HARD_CONSTRAINTS_BY_TYPE:
            qc = "default"
        hc_list = self.HARD_CONSTRAINTS_BY_TYPE[qc]
        hc_text = "\n".join(f"{hc_id}：{rule}" for hc_id, rule in hc_list)
        oc_list = self.OPTIONAL_CHECKS_BY_TYPE.get(qc, self.OPTIONAL_CHECKS_BY_TYPE["default"])
        oc_text = "\n".join(f"{oc_id}（0-1分）：{rule}" for oc_id, rule in oc_list)

        anchors_text = ""
        if anchor_sentences:
            lines = [s for s in anchor_sentences if isinstance(s, str) and s.strip()]
            if lines:
                anchors_text = "## 答案在原文中的支撑句：\n" + "\n".join(
                    f'  -「{s}」' for s in lines
                )

        return f"""你是一个技术文档QA评测专家，请对以下问答对进行Rubric评分。

## 原文参考：
{context}
{anchors_text}

## 问题：
{question}

## 标准答案（经溯源的可信版本）：
{standard_answer}

## 待评分答案（精炼后的版本，可能与标准答案有差异）：
{answer}

## 题型：
{qc}

---

## 第一部分：硬约束（HC）—— 仅检查一票否决项
{hc_text}

判定要求：
- 先判断每条 HC 是否适用于当前问题，输出 `applicable: true/false`
- 若 `applicable=false`，则该条 `pass` 填 null，`reason` 说明为何不适用
- 只有 `applicable=true 且 pass=false` 才算硬约束失败
- 若存在任意硬约束失败，则最终 `verdict` 必须为 `FAIL`

## 第二部分：软约束（SC）—— 核心质量分（满分 16 分）

评分强制要求：
- 4 分是"几乎无瑕疵"的满分档，只要命中任何一条减分项，一律不得给 4 分
- 3 分代表"方向正确但存在明显瑕疵"，不是"还不错"的保底分
- 默认向下卡分，遇到拿不准时降一档，不要无脑给 3-4 分

SC-1 原文溯源性（0-4 分）—— 可信程度（**不惩罚同义改写和合理的结构重组**）
  评分宽松方向：本维度只惩罚"编造新事实"或"与原文矛盾"；**同义改写 / 术语替换 / 分点重排 / 合并同义句 / 适度精简 / 添加原文可直接推出的物理结构细化**（例如把"光接口"细化为"光接口法兰面"，这是器件物理结构常识性延伸）都**不得视为原文不支持**
  4 分：答案中的每一句陈述都能在原文或支撑句中找到明确对应；允许同义改写和结构重组；没有引入原文无法直接推出的新事实
  3 分：主体对齐原文，允许若干同义改写；最多出现 1-2 处"轻度合理化补充"（例如"目视确认"、"无松脱/歪斜"这类不矛盾、不伪造阈值的操作常识补全）
  2 分：核心事实正确，但出现了原文**没有**的**具体事实**（例如具体命令、具体数值、具体版本号、具体时长阈值、具体根因机制链），而这些具体事实是在原文中找不到依据的
  1 分：部分可溯源，另含较多编造的具体事实或原文不支持的具体数值 / 命令 / 机制
  0 分：大量编造具体事实、与原文明确矛盾，或整体内容换了一套原文未曾提及的说法
  **判分流程提示**：请先判断"是否引入了原文中找不到的**具体新事实**（命令 / 参数值 / 阈值 / 机制细节 / 版本号）"——若没有，**至少给 3 分**；仅在出现这些具体编造时才扣到 2 分及以下

SC-2 实用价值（0-4 分）—— 能否直接用于决策或排障
  4 分：面向真实运维 / 故障 / 调优 / 迁移场景，给出的判断或决策可以被值班人员直接采用，颗粒度到了"当下应该做什么 / 不做什么"
  3 分：方向正确、场景明确，但决策颗粒度不够（例如给了方向但没给阈值、没给优先级、没给取舍依据）
  2 分：只停留在概念解释或文档复述，对实际决策帮助有限
  1 分：与真实运维场景几乎不相关，属于纯背诵或纯概念
  0 分：答非所问，或仅是对文档标题/章节的重复，无任何决策价值
  **SC-3 豁免类型的 SC-2 评分口径调整**：对于命中 SC-3 豁免说明的两类问题，SC-2 的"实用价值"不以"能否给出排障命令"衡量，而以"是否提供了准确的知识/机制/决策依据"衡量，评分锚点整体上移 1 档：
    (a) 纯知识记忆型（仅问定义/参数名/默认值/概念边界）：答案给出准确的定义或边界说明即可得 3 分；若答案还能说明该知识点的运维适用场景则得 4 分；答案只是抄录原文片段、不提炼关键结论则 2 分
    (b) 机制-决策型（问根因机制/最佳实践选型/合规策略/差异对比）：答案给出了可支撑决策的机制解释或选型依据即可得 3 分；若能给出明确的决策结论或对比判断则得 4 分；若只是把相关概念罗列一遍、没有解释其决策含义则 2 分

SC-3 可执行性（0-4 分）—— 能否直接照做
  4 分：至少一条命令 / SQL / 系统视图查询 / 参数修改可以直接复制执行，步骤闭合（前置条件 + 执行 + 验证方式齐全），运维人员可以无障碍落地
  3 分：命令或步骤存在，但缺少验证方式、前置条件未明确，或其中一步仅用自然语言描述
  2 分：只给出方向性动作（例如"修改参数 X"），没有具体命令或取值；或命令不完整、需要大量补全
  1 分：几乎无可执行信息，只是文字说明
  0 分：无任何可执行内容；或命令明显错误、无法运行
  豁免说明：以下两类问题本身不要求给出可执行命令，可在 reason 中注明类别并按 4 分处理，但 SC-2 必须真正给出决策 / 机制 / 依据解释，否则回落到常规评分：
    (a) 纯知识记忆型：仅问定义 / 参数名 / 默认值 / 概念边界
    (b) 机制-决策型：问根因机制 / 最佳实践的选型依据 / 权限与合规策略 / 差异对比结论等，答案以解释或决策依据为主，不以命令落地为交付物

SC-4 精确完整性（0-4 分）—— 关键事实 + 无水分
  4 分：所有关键事实均准确，完整覆盖问题真正需要的信息，且无冗余表述、无礼貌语、无与问题无关的铺垫段
  3 分：关键事实正确，但存在一处次要事实偏差，或包含少量冗余（铺垫段 / 总结段 / 重复说明）
  2 分：核心事实基本正确，但存在多处冗余或次要事实偏差
  1 分：关键事实存在明显错误，或答非所问的冗余段落占比超过 1/3
  0 分：关键事实错误，或未真正回答问题

## 第三部分：可选加分（OC）—— 按题型的加分项（满分3分）
{oc_text}

---

## 输出格式（严格JSON）：
{{
  "hard_constraints": {{
    "HC-1": {{"applicable": true, "pass": true, "reason": "判断依据"}},
    "HC-2": {{"applicable": true, "pass": true, "reason": "判断依据"}},
    "HC-3": {{"applicable": true, "pass": true, "reason": "判断依据"}}
  }},
  "hard_pass": true,
  "soft_scores": {{
    "SC-1": {{"score": 0, "reason": "评分依据"}},
    "SC-2": {{"score": 0, "reason": "评分依据"}},
    "SC-3": {{"score": 0, "reason": "评分依据"}},
    "SC-4": {{"score": 0, "reason": "评分依据"}}
  }},
  "optional_scores": {{
    "OC-1": {{"score": 0, "reason": "评分依据"}},
    "OC-2": {{"score": 0, "reason": "评分依据"}},
    "OC-3": {{"score": 0, "reason": "评分依据"}}
  }},
  "total_soft_score": 0,
  "total_optional_score": 0,
  "final_score": 0,
  "verdict": "PASS/FAIL",
  "main_issue": "最主要的问题是什么（一句话）"
}}

注意：
1. 只要有任意一个 `applicable=true and pass=false` 的 HC，`verdict` 必须为 `FAIL`
2. `applicable=false` 不算 fail，也不参与 `hard_pass` 判定
3. `total_soft_score` 的满分为 16 分（SC-1 ~ SC-4 各 0-4 分之和）
4. `verdict` 判定为 `PASS` 必须同时满足以下全部条件，任意一条不满足即为 `FAIL`：
   - `hard_pass=true`（所有适用的 HC 均通过）
   - `total_soft_score >= 12`
   - `SC-1.score >= 3`（原文溯源性至少"主体可对应原文"）
   - `SC-3.score >= 3`（可执行性至少"命令或步骤存在但未完全闭合"；符合 SC-3 豁免说明的题型按 4 分视为满足）
   - **SC-2 门槛按题型分档（两类题型不同）**：
     * 普通题型（非 SC-3 豁免）：`SC-2.score >= 3`（实用价值必须达到"方向正确、场景明确"）
     * SC-3 豁免类型（纯知识记忆型 / 机制-决策型）：`SC-2.score >= 2`（只要答案提供了准确的知识或机制解释即可；这类题型本身不以排障命令为交付物，用普通门槛会系统性误杀）
     * 判断依据：若 SC-3 的 reason 字段注明了"(a) 纯知识记忆型"或"(b) 机制-决策型"豁免，则本条自动适用宽松门槛
5. 反作弊原则：默认向下卡分；不得因为"答案看起来专业 / 排版漂亮 / 字数多"就给 4 分。SC-1 / SC-3 的 4 分必须逐条对应到原文句子或可执行命令，给不出具体证据时必须降到 3 分及以下
"""


@PROMPT_REGISTRY.register()
def normalize_grounding_verdict(raw: str | None) -> str:
    return _norm_verdict(raw, {"KEEP", "REWRITE", "DROP"}, "DROP")
