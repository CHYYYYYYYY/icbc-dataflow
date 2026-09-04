"""
TextAnswerGenerator: 基于问题和上下文生成答案的算子

用途：接收问题列表和对应的上下文，使用 LLM 生成准确的答案，适用于构建 SFT 数据集。
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import TextAnswerGeneratorPrompt
from dataflow.core.prompt import prompt_restrict


@prompt_restrict(TextAnswerGeneratorPrompt)
@OPERATOR_REGISTRY.register()
class TextAnswerGenerator(OperatorABC):
    """
    答案生成算子：基于问题和上下文生成准确的答案
    
    参数：
        llm_serving: LLM 服务对象，需实现 LLMServingABC 接口
        custom_prompt: 用户自定义的额外约束（可选）
        output_format: 输出格式说明（可选）
    """
    
    def __init__(
        self, 
        llm_serving: LLMServingABC, 
        custom_prompt: str = "",
        output_format: str = ""
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.custom_prompt = custom_prompt
        self.output_format = output_format
        self.prompts = TextAnswerGeneratorPrompt(
            custom_prompt=custom_prompt,
            output_format=output_format
        )
        self.logger.info(f'Initialized {self.__class__.__name__}')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于问题和参考内容生成准确的答案，适用于构建 SFT 数据集。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- custom_prompt：用户自定义的额外约束（可选）\n"
                "- output_format：输出格式说明（可选）\n"
                "- input_question_key：输入问题字段名，默认为 'question'\n"
                "- input_content_key：输入上下文字段名，默认为 'raw_content'\n"
                "输出参数：\n"
                "- 包含 'instruction'、'output' 和 'raw_content' 字段的 DataFrame\n"
                "- instruction 为问题，output 为生成的答案"
            )
        elif lang == "en":
            return (
                "Generate accurate answers based on questions and reference content for SFT dataset.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- custom_prompt: User-defined additional constraints (optional)\n"
                "- output_format: Output format specification (optional)\n"
                "- input_question_key: Input question field name, default is 'question'\n"
                "- input_content_key: Input context field name, default is 'raw_content'\n"
                "Output Parameters:\n"
                "- DataFrame containing 'instruction', 'output', and 'raw_content' fields\n"
                "- instruction is the question, output is the generated answer"
            )
        else:
            return "TextAnswerGenerator generates answers from questions and context for SFT datasets."

    def run(
        self, 
        storage: DataFlowStorage, 
        input_question_key: str = "question",
        input_content_key: str = "raw_content",
        output_instruction_key: str = "instruction",
        output_answer_key: str = "output",
        output_content_key: str = "raw_content",
        preserve_input_columns: bool = True,
    ):
        """
        运行答案生成算子
        
        Args:
            storage: DataFlow 存储对象
            input_question_key: 输入问题的字段名，默认为 'question'
            input_content_key: 输入上下文的字段名，默认为 'raw_content'
            output_instruction_key: 输出指令（问题）的字段名，默认为 'instruction'
            output_answer_key: 输出答案的字段名，默认为 'output'
            output_content_key: 输出原始内容的字段名，默认为 'raw_content'
            preserve_input_columns: True 时在原表上追加/覆盖输出列并保留其它列；False 时仅写出三列（与旧版一致）
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        # 读取数据
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} rows")
        
        # 检查必要字段
        if input_question_key not in dataframe.columns:
            raise ValueError(f"Input data missing '{input_question_key}' field")
        if input_content_key not in dataframe.columns:
            raise ValueError(f"Input data missing '{input_content_key}' field")
        
        # 准备 LLM 输入
        llm_inputs = []
        for _, row in dataframe.iterrows():
            question = row.get(input_question_key, '')
            content = row.get(input_content_key, '')
            llm_input = self.prompts.build_prompt(text=content, question=question)
            llm_inputs.append(llm_input)
        
        # 调用 LLM 生成答案
        self.logger.info("Generating answers...")
        outputs = self.llm_serving.generate_from_input(llm_inputs)
        self.logger.info("Answer generation completed.")
        
        if not preserve_input_columns:
            result_records = []
            for idx, output in enumerate(outputs):
                answer = output.strip() if output else ""
                result_records.append({
                    output_instruction_key: dataframe[input_question_key].iloc[idx],
                    output_answer_key: answer,
                    output_content_key: dataframe[input_content_key].iloc[idx]
                })
            output_df = pd.DataFrame(result_records)
            storage.write(output_df)
            self.logger.info(f"Generated {len(output_df)} answers")
            return [output_instruction_key, output_answer_key, output_content_key]

        df = dataframe.copy()
        cleaned = [(o.strip() if o else "") for o in outputs]
        df[output_instruction_key] = df[input_question_key].values
        df[output_answer_key] = cleaned
        if output_content_key != input_content_key:
            df[output_content_key] = df[input_content_key].values
        storage.write(df)
        self.logger.info(f"Generated {len(df)} answers (preserved input columns)")
        
        return [output_instruction_key, output_answer_key, output_content_key]

