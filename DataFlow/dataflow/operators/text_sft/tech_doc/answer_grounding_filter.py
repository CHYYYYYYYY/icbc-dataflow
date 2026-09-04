import pandas as pd

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.tech_doc_qa import (
    AnswerGroundingFilterPrompt,
    normalize_grounding_verdict,
    parse_json_llm_response,
)
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@prompt_restrict(AnswerGroundingFilterPrompt)
@OPERATOR_REGISTRY.register()
class AnswerGroundingFilter(OperatorABC):
    """Judge whether the answer is grounded in the source text."""

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = AnswerGroundingFilterPrompt()

    def run(
        self,
        storage: DataFlowStorage,
        input_context_key: str = "raw_content",
        input_question_key: str = "rough_question",
        input_answer_key: str = "initial_answer",
        output_verdict_key: str = "grounding_verdict",
        output_grounding_key: str = "grounding_result",
    ):
        df = storage.read("dataframe")
        prompts = [
            self.prompt.build_prompt(
                context=str(row.get(input_context_key, "") or ""),
                question=str(row.get(input_question_key, "") or ""),
                answer=str(row.get(input_answer_key, "") or ""),
            )
            for _, row in df.iterrows()
        ]
        responses = self.llm_serving.generate_from_input(
            user_inputs=prompts,
            system_prompt="你是一个严格的技术文档QA质量审核专家。",
        )
        verdicts: list[str] = []
        groundings: list[dict] = []
        for resp in responses:
            result = parse_json_llm_response(resp)
            if not result:
                verdicts.append("DROP")
                groundings.append({})
                continue
            v = normalize_grounding_verdict(result.get("verdict"))
            verdicts.append(v)
            groundings.append(result)

        df = df.copy()
        df[output_verdict_key] = verdicts
        df[output_grounding_key] = groundings
        storage.write(df)
        return [output_verdict_key, output_grounding_key]
