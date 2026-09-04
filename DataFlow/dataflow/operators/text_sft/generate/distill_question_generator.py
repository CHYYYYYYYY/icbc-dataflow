"""
DistillQuestionGenerator: 基于标签/主题蒸馏生成高质量领域问题的算子

用途：根据给定的标签或主题，使用 LLM 生成指定数量的领域相关问题，
     适用于构建领域知识问答数据集。
支持两步出题模式：先分析文本信号选定题型，再约束出题（减少无关题型冗余）。
"""

import json
import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.meta_label_filters import extract_llm_answer_text
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import DistillQuestionGeneratorPrompt, TextContentTypeAnalyzerPrompt
from dataflow.core.prompt import prompt_restrict


def extract_json_array(model_output: str) -> list:
    """
    从模型输出中提取 JSON 数组

    Args:
        model_output: 模型的原始输出字符串

    Returns:
        提取的问题列表，如果解析失败返回空列表
    """
    # reasoning 模型输出会包含 <think>...</think>，需先提取 <answer> 正文
    clean = extract_llm_answer_text(model_output or "")
    candidates = [clean, model_output or ""] if clean else [model_output or ""]
    for text in candidates:
        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        json_pattern = r'\[[\s\S]*?\]'
        for match in re.findall(json_pattern, text):
            try:
                result = json.loads(match)
                if isinstance(result, list) and len(result) > 0:
                    return result
            except json.JSONDecodeError:
                continue
    return []


_OBJECTIVE_AXIS_B_TYPES = ("FillBlank", "MultipleChoice")


def _merge_allowed_types_with_objective(
    allowed_types: list | None,
    *,
    ensure: bool,
) -> list | None:
    """两步分析未返回客观题型时，仍并入 FillBlank/MultipleChoice，避免出题阶段无法选客观题。"""
    if not ensure or allowed_types is None:
        return allowed_types
    seen: set[str] = set()
    merged: list[str] = []
    for t in list(allowed_types) + list(_OBJECTIVE_AXIS_B_TYPES):
        s = str(t).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        merged.append(s)
    return merged


def extract_json_object(model_output: str) -> dict:
    """
    从模型输出中提取 JSON 对象（用于文本信号分析结果）

    Args:
        model_output: 模型的原始输出字符串

    Returns:
        解析后的字典，失败返回空字典
    """
    # reasoning 模型输出会包含 <think>...</think>，需先提取 <answer> 正文
    clean = extract_llm_answer_text(model_output or "")
    candidates = [clean, model_output or ""] if clean else [model_output or ""]
    for text in candidates:
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        json_pattern = r'\{[\s\S]*?\}'
        for match in re.findall(json_pattern, text):
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    return {}


@prompt_restrict(DistillQuestionGeneratorPrompt)
@OPERATOR_REGISTRY.register()
class DistillQuestionGenerator(OperatorABC):
    """
    领域问题蒸馏生成算子：基于标签/主题生成高质量领域问题

    参数：
        llm_serving: LLM 服务对象，需实现 LLMServingABC 接口
        current_tag: 当前标签/主题名称
        tag_path: 标签完整链路（可选）
        count: 每个上下文生成的问题数量，默认为 10
        existing_questions: 已有问题列表，用于避免重复
        two_step_type_selection: 是否启用两步出题模式（先分析文本信号选定题型，再约束出题）
        ensure_objective_questions: 是否在约束中保证少量客观题（FillBlank+MultipleChoice 合计有下限且含上限），默认 True
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        current_tag: str,
        tag_path: str = "",
        count: int = 10,
        existing_questions: list = None,
        two_step_type_selection: bool = False,
        ensure_objective_questions: bool = True,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.current_tag = current_tag
        self.tag_path = tag_path
        self.count = count
        self.existing_questions = existing_questions or []
        self.two_step_type_selection = two_step_type_selection
        self.ensure_objective_questions = ensure_objective_questions
        self.prompts = DistillQuestionGeneratorPrompt(
            count=count,
            existing_questions=existing_questions,
            ensure_objective_questions=ensure_objective_questions,
        )
        if two_step_type_selection:
            self.type_analyzer_prompt = TextContentTypeAnalyzerPrompt()
        self.logger.info(
            f'Initialized {self.__class__.__name__} with tag="{current_tag}", '
            f'count={count}, two_step={two_step_type_selection}, '
            f'objective_quota={ensure_objective_questions}'
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于标签/主题和参考上下文蒸馏生成高质量领域问题，适用于构建领域知识问答数据集。\n"
                "特点：\n"
                "- 问题与标签主题紧密相关\n"
                "- 基于参考上下文生成更具体的问题\n"
                "- 场景 × 题型双轴覆盖（10类场景 × 11类题型）\n"
                "- 支持客观题型：FillBlank（填空题）和 MultipleChoice（单选题），含答案字段\n"
                "- 支持两步出题模式：先分析文本信号选定题型，再约束出题\n"
                "- 避免重复或高度相似的问题\n\n"
                "初始化参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- current_tag：当前标签/主题名称\n"
                "- tag_path：标签完整链路（可选）\n"
                "- count：每个上下文生成的问题数量，默认为 10\n"
                "- existing_questions：已有问题列表，用于避免重复\n"
                "- two_step_type_selection：是否启用两步出题模式，默认 False\n"
                "- ensure_objective_questions：是否在本批中约束客观题条数（下限+上限），默认 True；关闭则仅保留客观题上限\n\n"
                "运行参数：\n"
                "- input_context_key：输入上下文字段名，默认为 'raw_content'\n"
                "- output_question_key：输出问题字段名，默认为 'question'\n"
                "- output_context_key：输出上下文字段名，默认为 'raw_content'\n\n"
                "输出参数：\n"
                "- 包含 'question' 和 'raw_content' 字段的 DataFrame\n"
                "- 每个问题占一行，FillBlank 题含 distill_blank_answer，MultipleChoice 含 distill_options/distill_correct_answer"
            )
        elif lang == "en":
            return (
                "Generate high-quality domain questions based on tags/topics and reference context.\n"
                "Features:\n"
                "- Questions closely related to tag topics\n"
                "- Dual-axis coverage (10 scenarios × 11 question types)\n"
                "- Supports objective types: FillBlank and MultipleChoice with answer fields\n"
                "- Optional two-step mode: analyze text signals first, then constrain question generation\n"
                "- Avoid duplicate or highly similar questions\n\n"
                "Init Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- current_tag: Current tag/topic name\n"
                "- tag_path: Full tag path (optional)\n"
                "- count: Number of questions per context, default is 10\n"
                "- existing_questions: List of existing questions to avoid duplication\n"
                "- two_step_type_selection: Enable two-step type selection mode, default False\n"
                "- ensure_objective_questions: Enforce objective question count bounds (min+max), default True\n\n"
                "Run Parameters:\n"
                "- input_context_key: Input context field name, default is 'raw_content'\n"
                "- output_question_key: Output question field name, default is 'question'\n"
                "- output_context_key: Output context field name, default is 'raw_content'"
            )
        else:
            return "DistillQuestionGenerator generates domain questions from tags/topics and context."

    def _analyze_context_types(self, contexts: list) -> list[list]:
        """
        两步模式第一步：调用 LLM 分析每个 context 的文本信号，返回适合的题型列表

        Returns:
            每个 context 对应的 suitable_types 列表（列表的列表）
        """
        self.logger.info(f"[two-step] Analyzing text signals for {len(contexts)} contexts...")
        analysis_inputs = [self.type_analyzer_prompt.build_prompt(ctx) for ctx in contexts]
        analysis_outputs = self.llm_serving.generate_from_input(analysis_inputs)

        results = []
        for i, output in enumerate(analysis_outputs):
            parsed = extract_json_object(output)
            suitable_types = parsed.get("suitable_types", [])
            text_signals = parsed.get("text_signals", [])
            if not suitable_types:
                suitable_types = None  # 解析失败时不约束
                self.logger.warning(f"[two-step] Context {i}: type analysis parse failed, using unrestricted types")
            else:
                self.logger.debug(f"[two-step] Context {i}: signals={text_signals}, types={suitable_types}")
            results.append(suitable_types)

        return results

    def run(
        self,
        storage: DataFlowStorage,
        input_context_key: str = "raw_content",
        output_question_key: str = "question",
        output_context_key: str = "raw_content"
    ):
        """
        运行问题蒸馏生成算子

        Args:
            storage: DataFlow 存储对象
            input_context_key: 输入上下文的字段名
            output_question_key: 输出问题的字段名
            output_context_key: 输出上下文的字段名

        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__} for tag '{self.current_tag}'...")

        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} contexts")

        contexts = [str(row.get(input_context_key, '')) for _, row in dataframe.iterrows()]

        # 两步模式：先分析文本信号，获取每个 context 的适合题型
        if self.two_step_type_selection:
            allowed_types_per_context = self._analyze_context_types(contexts)
        else:
            allowed_types_per_context = [None] * len(contexts)

        # 构建出题 prompts（携带题型约束）
        llm_inputs = []
        for ctx, allowed_types in zip(contexts, allowed_types_per_context):
            merged_types = _merge_allowed_types_with_objective(
                allowed_types,
                ensure=self.ensure_objective_questions and self.two_step_type_selection,
            )
            llm_input = self.prompts.build_prompt(
                current_tag=self.current_tag,
                tag_path=self.tag_path,
                context=ctx,
                allowed_types=merged_types,
            )
            llm_inputs.append(llm_input)

        self.logger.info("Generating questions from contexts...")
        outputs = self.llm_serving.generate_from_input(llm_inputs)
        self.logger.info("Question generation completed.")

        result_records = []
        for idx, output in enumerate(outputs):
            questions_data = extract_json_array(output)
            source_context = dataframe[input_context_key].iloc[idx]

            for q_item in questions_data:
                question = ""
                scenario = ""
                question_type = ""
                blank_answer = ""
                options = None
                correct_answer = ""

                if isinstance(q_item, str) and q_item.strip():
                    question = q_item.strip()
                elif isinstance(q_item, dict):
                    question = str(q_item.get("question", "")).strip()
                    scenario = str(q_item.get("scenario", "")).strip()
                    question_type = str(q_item.get("question_type", "")).strip()
                    blank_answer = str(q_item.get("blank_answer", "")).strip()
                    raw_options = q_item.get("options")
                    if isinstance(raw_options, list):
                        options = raw_options
                    correct_answer = str(q_item.get("correct_answer", "")).strip()

                if question:
                    record = {
                        output_question_key: question,
                        output_context_key: source_context,
                    }
                    if scenario:
                        record["distill_scenario"] = scenario
                    if question_type:
                        record["distill_question_type"] = question_type
                    if blank_answer:
                        record["distill_blank_answer"] = blank_answer
                    if options is not None:
                        record["distill_options"] = json.dumps(options, ensure_ascii=False)
                    if correct_answer:
                        record["distill_correct_answer"] = correct_answer
                    result_records.append(record)

        output_df = pd.DataFrame(result_records)
        storage.write(output_df)
        self.logger.info(f"Generated {len(output_df)} questions from {len(dataframe)} contexts")

        out_keys = [output_question_key, output_context_key]
        for extra_col in (
            "distill_scenario",
            "distill_question_type",
            "distill_blank_answer",
            "distill_options",
            "distill_correct_answer",
        ):
            if extra_col in output_df.columns:
                out_keys.append(extra_col)

        return out_keys
