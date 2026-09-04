from __future__ import annotations

import re

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.general_text import AnswerCritiquePrompt
from dataflow.utils.meta_label_filters import extract_llm_answer_text
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


def _extract_section(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.compile(
        re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
        re.DOTALL,
    )
    match = pattern.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def _extract_field(block: str, field_name: str) -> str:
    """提取 'field_name [(alias)] [：:] value' 中的 value。
    支持中文括号 / 英文括号包裹的 alias，例如 '强制重写（Force Rewrite）：是'。
    """
    pattern = re.compile(
        rf"{re.escape(field_name)}\s*(?:[（(][^）)]*[）)])?\s*[：:]\s*(.+)"
    )
    for line in (block or "").splitlines():
        match = pattern.search(line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _is_failed_dimension(name: str, rating: str) -> bool:
    negative_map = {
        "completeness": {"不完整"},
        "accuracy": {"不准确"},
        "relevance": {"针对性弱"},
        "structure": {"不清晰"},
        "source_fidelity": {"无支撑"},
        "meta_info": {"不通过"},
        "structure_compression": {"不通过"},
    }
    return rating in negative_map.get(name, set())


def _parse_executability(block: str) -> dict:
    """解析 Executability Check 区块（维度 7，round 3 引入）。

    返回 {applicable, has_executable_step, step_is_complete, failed} 四个布尔字段。
    """
    if not block:
        return {
            "applicable": True,
            "has_executable_step": True,
            "step_is_complete": True,
            "failed": False,
        }
    applicability = _extract_field(block, "适用性")
    has_step = _extract_field(block, "has_executable_step")
    step_complete = _extract_field(block, "step_is_complete")
    is_applicable = applicability != "不适用"
    pass_token = "通过"
    has_step_ok = (not is_applicable) or has_step.startswith(pass_token)
    step_complete_ok = (not is_applicable) or step_complete.startswith(pass_token)
    return {
        "applicable": is_applicable,
        "has_executable_step": has_step_ok,
        "step_is_complete": step_complete_ok,
        "failed": is_applicable and not (has_step_ok and step_complete_ok),
    }


def parse_answer_critique(critique_text: str) -> dict:
    if not isinstance(critique_text, str) or not critique_text.strip():
        return {}
    # reasoning 模型把分析过程放在 <think> 里，实际评估内容在 <answer> 里
    critique_text = extract_llm_answer_text(critique_text) or critique_text

    sections = {
        "completeness": _extract_section(
            critique_text,
            "[Completeness Check Start]",
            "[Completeness Check End]",
        ),
        "accuracy": _extract_section(
            critique_text,
            "[Accuracy Check Start]",
            "[Accuracy Check End]",
        ),
        "relevance": _extract_section(
            critique_text,
            "[Relevance Check Start]",
            "[Relevance Check End]",
        ),
        "structure": _extract_section(
            critique_text,
            "[Structure Check Start]",
            "[Structure Check End]",
        ),
        "source_fidelity": _extract_section(
            critique_text,
            "[Source Fidelity Check Start]",
            "[Source Fidelity Check End]",
        ),
        "meta_info": _extract_section(
            critique_text,
            "[Meta Info Check Start]",
            "[Meta Info Check End]",
        ),
        "executability": _extract_section(
            critique_text,
            "[Executability Check Start]",
            "[Executability Check End]",
        ),
        "structure_compression": _extract_section(
            critique_text,
            "[Structure Check2 Start]",
            "[Structure Check2 End]",
        ),
        "overall": _extract_section(
            critique_text,
            "[Overall Assessment Start]",
            "[Overall Assessment End]",
        ),
        "suggestion": _extract_section(
            critique_text,
            "[Suggestion Start]",
            "[Suggestion End]",
        ),
    }

    dimensions = {}
    failed_dimensions = []
    # 标准 6 维度 + 结构压缩（评分通过/不通过）
    for name in (
        "completeness",
        "accuracy",
        "relevance",
        "structure",
        "source_fidelity",
        "meta_info",
        "structure_compression",
    ):
        block = sections[name]
        rating = _extract_field(block, "评分")
        reason = _extract_field(block, "说明")
        failed = _is_failed_dimension(name, rating)
        dimensions[name] = {
            "rating": rating,
            "reason": reason,
            "failed": failed,
        }
        if failed:
            failed_dimensions.append(name)

    # 维度 7 可操作性硬门槛（特殊：三字段）
    exec_block = sections["executability"]
    exec_dim = _parse_executability(exec_block)
    dimensions["executability"] = exec_dim
    if exec_dim["failed"]:
        failed_dimensions.append("executability")

    overall_rating = _extract_field(sections["overall"], "总体评估")
    issue_count_text = _extract_field(sections["overall"], "问题数量")
    force_rewrite_text = _extract_field(sections["overall"], "强制重写")
    if not force_rewrite_text:
        force_rewrite_text = _extract_field(sections["overall"], "Force Rewrite")
    force_rewrite = force_rewrite_text.startswith("是") if force_rewrite_text else False

    try:
        issue_count = int(re.search(r"\d+", issue_count_text or "").group(0))
    except AttributeError:
        issue_count = len(failed_dimensions)

    needs_improvement = (
        overall_rating == "需要改进"
        or issue_count > 0
        or bool(failed_dimensions)
        or force_rewrite
    )
    return {
        "dimensions": dimensions,
        "failed_dimensions": failed_dimensions,
        "issue_count": issue_count,
        "overall_assessment": overall_rating,
        "needs_improvement": needs_improvement,
        "force_rewrite": force_rewrite,
        "suggestion": sections["suggestion"],
    }


@prompt_restrict(AnswerCritiquePrompt)
@OPERATOR_REGISTRY.register()
class AnswerCritiqueEvaluator(OperatorABC):
    """Generate answer critiques and parse them into a summary."""

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = AnswerCritiquePrompt()

    def run(
        self,
        storage: DataFlowStorage,
        input_context_key: str = "raw_content",
        input_question_key: str = "rough_question",
        input_answer_key: str = "clean_answer",
        output_critique_key: str = "answer_critique",
        output_summary_key: str = "answer_critique_summary",
        output_issue_count_key: str = "answer_critique_issue_count",
        output_needs_improvement_key: str = "answer_critique_needs_improvement",
        output_force_rewrite_key: str = "answer_critique_force_rewrite",
    ):
        df = storage.read("dataframe").copy()
        prompts = [
            self.prompt.build_prompt(
                str(row.get(input_context_key, "") or ""),
                str(row.get(input_question_key, "") or ""),
                str(row.get(input_answer_key, "") or ""),
            )
            for _, row in df.iterrows()
        ]
        critiques = self.llm_serving.generate_from_input(user_inputs=prompts)
        summaries = [parse_answer_critique(text) for text in critiques]
        issue_counts = [
            summary.get("issue_count", 0) if isinstance(summary, dict) else 0
            for summary in summaries
        ]
        needs_improvement = [
            bool(summary.get("needs_improvement", False))
            if isinstance(summary, dict)
            else False
            for summary in summaries
        ]
        force_rewrite = [
            bool(summary.get("force_rewrite", False))
            if isinstance(summary, dict)
            else False
            for summary in summaries
        ]

        df[output_critique_key] = critiques
        df[output_summary_key] = summaries
        df[output_issue_count_key] = issue_counts
        df[output_needs_improvement_key] = needs_improvement
        df[output_force_rewrite_key] = force_rewrite

        n_force = sum(force_rewrite)
        if n_force:
            self.logger.info(
                "[AnswerCritiqueEvaluator] force_rewrite=%d/%d (issue_count avg=%.2f)",
                n_force,
                len(critiques),
                sum(issue_counts) / max(len(issue_counts), 1),
            )

        storage.write(df)
        return [
            output_critique_key,
            output_summary_key,
            output_issue_count_key,
            output_needs_improvement_key,
            output_force_rewrite_key,
        ]
