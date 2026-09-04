from __future__ import annotations

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.tech_doc_qa import AnswerRewriterPrompt, parse_json_llm_response
from dataflow.utils.meta_label_filters import is_outofscope_or_meta_tox
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


def _evidence_to_anchors(evidence) -> list[str]:
    if evidence is None:
        return []
    if isinstance(evidence, str) and evidence.strip():
        return [evidence.strip()]
    if isinstance(evidence, list):
        return [str(x).strip() for x in evidence if str(x).strip()]
    return []


@prompt_restrict(AnswerRewriterPrompt)
@OPERATOR_REGISTRY.register()
class AnswerRewriter(OperatorABC):
    """Rewrite REWRITE verdict rows strictly from source; KEEP/DROP pass through."""

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = AnswerRewriterPrompt()

    def run(
        self,
        storage: DataFlowStorage,
        input_context_key: str = "raw_content",
        input_question_key: str = "rough_question",
        input_answer_key: str = "initial_answer",
        input_grounding_key: str = "grounding_result",
        input_verdict_key: str = "grounding_verdict",
        output_answer_key: str = "clean_answer",
        output_anchors_key: str = "anchor_sentences",
        output_stage_key: str = "rewrite_stage",
    ):
        df = storage.read("dataframe").copy()
        n = len(df)
        clean_answers: list = [None] * n
        anchor_list: list[list[str]] = [[] for _ in range(n)]
        final_verdicts = ["DROP"] * n

        rewrite_prompts: list[str] = []
        rewrite_row_positions: list[int] = []

        outofscope_drop = 0
        meta_tox_drop = 0

        for pos, (_, row) in enumerate(df.iterrows()):
            verdict = str(row.get(input_verdict_key, "") or "").strip().upper()
            initial_ans = str(row.get(input_answer_key, "") or "")
            should_drop_initial, drop_reason_initial = is_outofscope_or_meta_tox(
                initial_ans
            )
            if should_drop_initial:
                final_verdicts[pos] = "DROP"
                clean_answers[pos] = None
                if drop_reason_initial == "outofscope":
                    outofscope_drop += 1
                else:
                    meta_tox_drop += 1
                continue

            if verdict == "KEEP":
                clean_answers[pos] = row.get(input_answer_key)
                gr = row.get(input_grounding_key) or {}
                if not isinstance(gr, dict):
                    gr = {}
                # 优先使用 anchor_sentences 列表；向后兼容 original_evidence 字符串（旧格式）
                raw_anchors = gr.get("anchor_sentences")
                if isinstance(raw_anchors, list) and raw_anchors:
                    anchor_list[pos] = [str(a).strip() for a in raw_anchors if str(a).strip()]
                else:
                    anchor_list[pos] = _evidence_to_anchors(gr.get("original_evidence"))
                final_verdicts[pos] = "KEEP"
            elif verdict == "DROP":
                final_verdicts[pos] = "DROP"
            else:
                rewrite_row_positions.append(pos)
                rewrite_prompts.append(
                    self.prompt.build_prompt(
                        context=str(row.get(input_context_key, "") or ""),
                        question=str(row.get(input_question_key, "") or ""),
                        answer=initial_ans,
                        grounding_result=row.get(input_grounding_key)
                        if isinstance(row.get(input_grounding_key), dict)
                        else {},
                    )
                )

        if rewrite_prompts:
            responses = self.llm_serving.generate_from_input(
                user_inputs=rewrite_prompts,
                system_prompt="你是一个技术文档答案改写专家。",
            )
        else:
            responses = []

        for pos, resp in zip(rewrite_row_positions, responses):
            result = parse_json_llm_response(resp)
            if not result:
                final_verdicts[pos] = "DROP"
                continue
            rv = str(result.get("verdict", "") or "").strip().upper()
            if rv == "DROP":
                final_verdicts[pos] = "DROP"
            else:
                ans = result.get("rewritten_answer")
                anchors = result.get("anchor_sentences") or []
                if not isinstance(anchors, list):
                    anchors = []
                anchors = [str(a).strip() for a in anchors if str(a).strip()]
                ans_text = ans if isinstance(ans, str) else str(ans or "")
                should_drop_rewritten, drop_reason_rewritten = (
                    is_outofscope_or_meta_tox(ans_text)
                )
                if should_drop_rewritten:
                    final_verdicts[pos] = "DROP"
                    clean_answers[pos] = None
                    anchor_list[pos] = []
                    if drop_reason_rewritten == "outofscope":
                        outofscope_drop += 1
                    else:
                        meta_tox_drop += 1
                    continue
                clean_answers[pos] = ans_text
                anchor_list[pos] = anchors
                final_verdicts[pos] = "REWRITTEN"

        df[output_answer_key] = clean_answers
        df[output_anchors_key] = anchor_list
        df[output_stage_key] = final_verdicts
        storage.write(df)
        if outofscope_drop or meta_tox_drop:
            self.logger.info(
                "[AnswerRewriter] tox-drop summary: outofscope=%d meta_tox_prefix=%d (total_in=%d)",
                outofscope_drop,
                meta_tox_drop,
                n,
            )
        return [output_answer_key, output_anchors_key, output_stage_key]
