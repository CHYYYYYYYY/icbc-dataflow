import json
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.meta_label_filters import (
    extract_improved_question_from_refiner,
    extract_llm_answer_text,
    is_valid_export_question,
)
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import QuestionCritiquePrompt, QuestionRefinePrompt
from dataflow.core.prompt import prompt_restrict


_FACT_CLOSURE_BLOCK_PATTERNS = [
    re.compile(
        r"\[Fact\s+Closure\s+Check\s+Start\](.*?)\[Fact\s+Closure\s+Check\s+End\]",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"【Fact\s+Closure\s+Check\s+Start】(.*?)【Fact\s+Closure\s+Check\s+End】",
        re.DOTALL | re.IGNORECASE,
    ),
]
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*?\"new_facts_introduced\"[^{}]*?\}", re.DOTALL)


def _parse_fact_closure_check(llm_response: str) -> dict | None:
    """Try to extract the fact_closure_check JSON object from LLM response.

    Returns the parsed dict on success, or None if not found / unparseable.
    """
    if not llm_response:
        return None
    # reasoning 模型的有效内容在 <answer> 里，避免 think 块里的 JSON 被误捕
    llm_response = extract_llm_answer_text(llm_response) or llm_response
    block_text = None
    for pat in _FACT_CLOSURE_BLOCK_PATTERNS:
        m = pat.search(llm_response)
        if m:
            block_text = m.group(1)
            break
    if block_text is None:
        block_text = llm_response

    fence = _JSON_FENCE_PATTERN.search(block_text)
    json_str = None
    if fence:
        json_str = fence.group(1)
    else:
        m2 = _JSON_OBJECT_PATTERN.search(block_text)
        if m2:
            json_str = m2.group(0)

    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def _closure_violates(closure: dict | None) -> tuple[bool, str]:
    """Return (violates, reason). Violates if new_facts_introduced or replaced_facts non-empty,
    or all_constraints_in_answer_or_context is explicitly false."""
    if not isinstance(closure, dict):
        return False, ""
    nf = closure.get("new_facts_introduced")
    rf = closure.get("replaced_facts")
    all_ok = closure.get("all_constraints_in_answer_or_context")
    if isinstance(nf, list) and len(nf) > 0:
        return True, f"new_facts={nf!r}"
    if isinstance(rf, list) and len(rf) > 0:
        return True, f"replaced_facts={rf!r}"
    if all_ok is False:
        return True, "all_constraints_in_answer_or_context=false"
    return False, ""


@prompt_restrict(
    QuestionCritiquePrompt,
    QuestionRefinePrompt
)
@OPERATOR_REGISTRY.register()
class QuestionRefiner(OperatorABC):
    """
    两阶段优化问题质量：
    第一阶段：根据上下文对生成的问题进行批评
    第二阶段：基于批评反馈改进问题
    
    批评维度：
    1. 元信息检查：问题是否包含章节、作者、段落等元信息
    2. 问题具体性：问题是否太泛泛，是否像人类口吻
    3. 运维价值：问题对运维人员是否有实际价值
    """
    
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.critique_prompt = QuestionCritiquePrompt()
        self.refine_prompt = QuestionRefinePrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "两阶段优化问题质量：第一阶段根据上下文对生成的问题进行批评，第二阶段基于批评反馈改进问题。\n\n"
                "批评维度：\n"
                "1. 元信息检查：问题是否包含章节、作者、段落等元信息\n"
                "2. 问题具体性：问题是否太泛泛，是否像人类口吻\n"
                "3. 运维价值：问题对运维人员是否有实际价值\n\n"
                "初始化参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n\n"
                "运行参数：\n"
                "- input_context_key：输入上下文字段名，默认为'raw_content'\n"
                "- input_question_key：输入问题字段名，默认为'question'\n"
                "- output_question_key：输出优化后问题字段名，默认为'refined_question'\n"
                "- output_critique_key：输出批评字段名，默认为'question_critique'\n\n"
                "输出参数：\n"
                "- 包含优化后问题的DataFrame\n"
                "- 返回包含优化后问题字段名的列表"
            )
        elif lang == "en":
            return (
                "Two-stage question quality optimization: First stage critiques generated questions based on context, "
                "second stage refines questions based on critique feedback.\n\n"
                "Critique Dimensions:\n"
                "1. Meta-info Check: Whether question contains chapter, author, paragraph references\n"
                "2. Specificity Check: Whether question is too vague, resembles human speech\n"
                "3. Value Check: Whether question has practical value for operations staff\n\n"
                "Initialization Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n\n"
                "Run Parameters:\n"
                "- input_context_key: Field name for input context, default is 'raw_content'\n"
                "- input_question_key: Field name for input question, default is 'question'\n"
                "- output_question_key: Field name for refined question, default is 'refined_question'\n"
                "- output_critique_key: Field name for critique, default is 'question_critique'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing refined questions\n"
                "- List containing refined question field name for subsequent operator reference"
            )
        else:
            return "QuestionRefiner improves question quality through two-stage critique and refinement process."

    def generate_critique(self, contexts: list, questions: list) -> list:
        """批量生成问题批评"""
        critique_prompts = [
            self.critique_prompt.build_prompt(ctx, q) 
            for ctx, q in zip(contexts, questions)
        ]
        critique_responses = self.llm_serving.generate_from_input(critique_prompts)
        return critique_responses

    def _extract_refined_question(self, llm_response: str, original_question: str) -> str | None:
        """从 LLM 响应中提取改进后的问题；提取失败或无效时返回 None（由调用方回退 original）。"""
        extracted = extract_improved_question_from_refiner(llm_response or "")
        if extracted and is_valid_export_question(extracted):
            return extracted

        # 不再将整个 <answer> 正文或 Analysis 段后全文当作问题（易泄漏英文推理）
        if llm_response:
            self.logger.debug(
                "Refiner output did not yield a valid improved question; fallback to original. "
                "preview=%s",
                extract_llm_answer_text(llm_response)[:160],
            )
        return None

    def generate_refined_question(
        self,
        contexts: list,
        questions: list,
        critiques: list,
        answers: list | None = None,
    ) -> list:
        """批量生成优化后的问题"""
        if answers is None:
            answers = [""] * len(questions)
        refine_prompts = [
            self.refine_prompt.build_prompt(ctx, q, c, a)
            for ctx, q, c, a in zip(contexts, questions, critiques, answers)
        ]
        refined_questions = self.llm_serving.generate_from_input(refine_prompts)

        # 提取改进后的问题（含事实闭包兜底）
        extracted_questions = []
        extraction_stats = {'success': 0, 'fallback': 0}
        closure_stats = {'ok': 0, 'violation': 0, 'missing': 0}

        for rq, orig_q in zip(refined_questions, questions):
            extracted = self._extract_refined_question(rq, orig_q)

            if not extracted:
                self.logger.warning(
                    f"Failed to extract refined question, fallback to original: {orig_q[:50]}..."
                )
                self.logger.debug(f"LLM response was: {rq[:200]}...")
                extracted_questions.append(orig_q)
                extraction_stats['fallback'] += 1
                continue

            if not is_valid_export_question(extracted):
                self.logger.warning(
                    "[QuestionRefiner] extracted question invalid, fallback to original. "
                    "extracted=%s orig=%s",
                    str(extracted)[:80],
                    str(orig_q)[:80],
                )
                extracted = orig_q
                extraction_stats['fallback'] += 1

            closure = _parse_fact_closure_check(rq or "")
            if closure is None:
                closure_stats['missing'] += 1
                extracted_questions.append(extracted)
                extraction_stats['success'] += 1
                continue

            violates, reason = _closure_violates(closure)
            if violates:
                self.logger.warning(
                    "[QuestionRefiner] fact-closure violation, fallback to original. "
                    "reason=%s orig=%s refined=%s",
                    reason,
                    str(orig_q)[:80],
                    str(extracted)[:80],
                )
                extracted_questions.append(orig_q)
                closure_stats['violation'] += 1
                extraction_stats['fallback'] += 1
            else:
                extracted_questions.append(extracted)
                closure_stats['ok'] += 1
                extraction_stats['success'] += 1

        total = len(refined_questions)
        if extraction_stats['fallback'] > 0:
            self.logger.info(
                f"Question extraction stats: {extraction_stats['success']}/{total} successful, "
                f"{extraction_stats['fallback']}/{total} used original question"
            )
        self.logger.info(
            "[QuestionRefiner] fact-closure: ok=%d violation=%d missing=%d (total=%d)",
            closure_stats['ok'],
            closure_stats['violation'],
            closure_stats['missing'],
            total,
        )

        return extracted_questions

    def run(self, 
            storage: DataFlowStorage, 
            input_context_key: str = 'raw_content',
            input_question_key: str = 'question',
            input_answer_key: str | None = None,
            output_question_key: str = 'refined_question',
            output_critique_key: str = 'question_critique'):
        """
        运行问题优化流程。
        
        Args:
            storage: DataFlow存储对象
            input_context_key: 输入上下文字段名
            input_question_key: 输入问题字段名
            input_answer_key: 输入答案字段名（可选）；当流水线中已有答案时传入，
                              用于事实闭包校验；TextQAPipeline（问题先于答案生成）
                              不传此参数。
            output_question_key: 输出优化后问题字段名
            output_critique_key: 输出批评字段名
        """
        df = storage.read('dataframe')
        
        contexts = df.get(input_context_key).to_list()
        questions = df.get(input_question_key).to_list()

        answers: list | None = None
        if input_answer_key and input_answer_key in df.columns:
            answers = [str(v) if v is not None else "" for v in df.get(input_answer_key).to_list()]
            self.logger.info(
                "[QuestionRefiner] using existing answers from '%s' for fact-closure check.",
                input_answer_key,
            )
        
        self.logger.info(f'Generating critiques for {len(questions)} questions...')
        critique_responses = self.generate_critique(contexts, questions)
        self.logger.info(f'Generated critiques for questions.')
        
        self.logger.info(f'Refining questions based on critiques...')
        refined_questions = self.generate_refined_question(contexts, questions, critique_responses, answers)
        self.logger.info(f'Refined questions generated.')
        
        df[output_critique_key] = critique_responses
        df[output_question_key] = refined_questions
        
        storage.write(df)
        self.logger.info(f'Refined questions saved to storage.')
        
        return [output_question_key, output_critique_key]

