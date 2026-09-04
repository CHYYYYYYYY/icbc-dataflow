from __future__ import annotations

from typing import Any

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.tech_doc_qa import RubricScorerPrompt, parse_json_llm_response
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


def _infer_hard_pass(result: dict[str, Any]) -> bool:
    hp = result.get("hard_pass")
    if isinstance(hp, bool):
        return hp
    hc = result.get("hard_constraints")
    if not isinstance(hc, dict) or not hc:
        return False
    for v in hc.values():
        if not isinstance(v, dict):
            return False
        applicable = v.get("applicable", True)
        if applicable is False:
            continue
        if v.get("pass") is not True:
            return False
    return True


def _infer_soft_total(result: dict[str, Any]) -> float:
    ts = result.get("total_soft_score")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    ss = result.get("soft_scores")
    if not isinstance(ss, dict):
        return 0.0
    total = 0.0
    for v in ss.values():
        if isinstance(v, dict) and "score" in v:
            try:
                total += float(v["score"])
            except (TypeError, ValueError):
                pass
    return total


def _get_sc_score(result: dict[str, Any], sc_key: str) -> float:
    ss = result.get("soft_scores")
    if not isinstance(ss, dict):
        return 0.0
    v = ss.get(sc_key)
    if not isinstance(v, dict):
        return 0.0
    try:
        return float(v.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _infer_final_score(result: dict[str, Any]) -> float:
    fs = result.get("final_score")
    if fs is not None:
        try:
            return float(fs)
        except (TypeError, ValueError):
            pass
    if not _infer_hard_pass(result):
        return 0.0
    try:
        return float(result.get("total_soft_score", 0) or 0) + float(
            result.get("total_optional_score", 0) or 0
        )
    except (TypeError, ValueError):
        return 0.0


def _is_sc3_exempt(result: dict[str, Any]) -> bool:
    """判断 SC-3 是否命中豁免说明（纯知识记忆型 / 机制-决策型）。

    依据：LLM 按提示词要求在 SC-3 reason 中注明了豁免类别，且 SC-3 score == 4。
    同时检查关键词（兼容 LLM 未精确写类别字母的情况）。
    """
    ss = result.get("soft_scores")
    if not isinstance(ss, dict):
        return False
    sc3 = ss.get("SC-3")
    if not isinstance(sc3, dict):
        return False
    try:
        score = float(sc3.get("score", 0) or 0)
    except (TypeError, ValueError):
        return False
    if score < 4:
        return False
    reason = str(sc3.get("reason", "") or "").lower()
    _EXEMPT_KEYWORDS = (
        "(a)", "(b)",
        "纯知识记忆", "机制-决策", "机制决策",
        "知识记忆型", "机制型", "决策型",
        "不要求", "不需要", "无需命令", "豁免",
    )
    return any(kw in reason for kw in _EXEMPT_KEYWORDS)


def _normalize_question_type(raw: str | None) -> str:
    if not raw or not isinstance(raw, str):
        return "default"
    s = raw.strip()
    if s in RubricScorerPrompt.HARD_CONSTRAINTS_BY_TYPE:
        return s
    return "default"


@prompt_restrict(RubricScorerPrompt)
@OPERATOR_REGISTRY.register()
class RubricScorer(OperatorABC):
    """Rubric: HC veto + soft score threshold; standard=clean, candidate=refined."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        min_soft_score: float = 12.0,
        filter_on_run: bool = False,
        min_sc1_score: float = 3.0,
        min_sc2_score: float = 3.0,
        min_sc2_score_exempt: float = 2.0,
        min_sc3_score: float = 3.0,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = RubricScorerPrompt()
        self.min_soft_score = min_soft_score
        self.filter_on_run = filter_on_run
        self.min_sc1_score = min_sc1_score
        self.min_sc2_score = min_sc2_score
        self.min_sc2_score_exempt = min_sc2_score_exempt
        self.min_sc3_score = min_sc3_score

    def run(
        self,
        storage: DataFlowStorage,
        input_context_key: str = "raw_content",
        input_question_key: str = "refined_reverse_question",
        input_standard_answer_key: str = "clean_answer",
        input_answer_key: str = "refined_answer",
        input_type_key: str = "question_type",
        input_anchors_key: str = "anchor_sentences",
        output_rubric_key: str = "rubric_result",
        output_score_key: str = "rubric_score",
        output_verdict_key: str = "rubric_verdict",
    ):
        df = storage.read("dataframe").copy()
        prompts: list[str] = []
        for _, row in df.iterrows():
            anchors = row.get(input_anchors_key)
            if not isinstance(anchors, list):
                anchors = []
            qtype = _normalize_question_type(row.get(input_type_key))
            prompts.append(
                self.prompt.build_prompt(
                    context=str(row.get(input_context_key, "") or ""),
                    question=str(row.get(input_question_key, "") or ""),
                    answer=str(row.get(input_answer_key, "") or ""),
                    standard_answer=str(row.get(input_standard_answer_key, "") or ""),
                    question_type=qtype,
                    anchor_sentences=anchors,
                )
            )
        responses = self.llm_serving.generate_from_input(
            user_inputs=prompts,
            system_prompt="你是一个技术文档QA评测专家。",
        )
        rubric_results: list[dict] = []
        scores: list[float] = []
        verdicts: list[str] = []
        for resp in responses:
            result = parse_json_llm_response(resp)
            if not result:
                rubric_results.append({})
                scores.append(0.0)
                verdicts.append("FAIL")
                continue
            hard_pass = _infer_hard_pass(result)
            soft_total = _infer_soft_total(result)
            final_score = _infer_final_score(result)
            result = {**result, "hard_pass": hard_pass, "total_soft_score": soft_total}
            sc1 = _get_sc_score(result, "SC-1")
            sc2 = _get_sc_score(result, "SC-2")
            sc3 = _get_sc_score(result, "SC-3")
            # SC-3 豁免类型（纯知识记忆型/机制-决策型）SC-2 门槛降为 exempt 档
            sc2_threshold = (
                self.min_sc2_score_exempt
                if _is_sc3_exempt(result)
                else self.min_sc2_score
            )
            pass_ok = (
                hard_pass
                and soft_total >= self.min_soft_score
                and sc1 >= self.min_sc1_score
                and sc2 >= sc2_threshold
                and sc3 >= self.min_sc3_score
            )
            final_verdict = "PASS" if pass_ok else "FAIL"
            result["verdict"] = final_verdict
            rubric_results.append(result)
            scores.append(final_score)
            verdicts.append(final_verdict)

        df[output_rubric_key] = rubric_results
        df[output_score_key] = scores
        df[output_verdict_key] = verdicts

        pass_n = sum(1 for v in verdicts if v == "PASS")
        fail_n = len(verdicts) - pass_n
        self.logger.info(
            f"[RubricScorer] verdict distribution: PASS={pass_n}/{len(verdicts)}, "
            f"FAIL={fail_n}/{len(verdicts)}, filter_on_run={self.filter_on_run}"
        )

        if self.filter_on_run:
            df = df[df[output_verdict_key] == "PASS"].reset_index(drop=True)
        else:
            self.logger.info(
                "[RubricScorer] filter_on_run=False: all rows (including FAIL) are kept with full "
                f"`{output_rubric_key}`/`{output_verdict_key}` for downstream auditing. "
                "Downstream stages should filter on verdict explicitly."
            )
        storage.write(df)
        return [output_rubric_key, output_score_key, output_verdict_key]
