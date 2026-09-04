"""
TextQuestionGenerator: 从文本中生成多个高质量问题的算子

用途：基于给定的文本内容，使用 LLM 生成指定数量的问题，适用于构建 SFT 数据集。
"""

import json
import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import TextQuestionGeneratorPrompt
from dataflow.core.prompt import prompt_restrict


def extract_json_array(model_output: str) -> list:
    """
    从模型输出中提取 JSON 数组
    
    Args:
        model_output: 模型的原始输出字符串
    
    Returns:
        提取的问题列表，如果解析失败返回空列表
    """
    # 尝试直接解析
    try:
        result = json.loads(model_output.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 数组模式
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, model_output)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            continue
    
    return []


@prompt_restrict(TextQuestionGeneratorPrompt)
@OPERATOR_REGISTRY.register()
class TextQuestionGenerator(OperatorABC):
    """
    问题生成算子：从文本中生成多个高质量问题
    
    参数：
        llm_serving: LLM 服务对象，需实现 LLMServingABC 接口
        number: 每段文本生成的问题数量，默认为 5
        custom_prompt: 用户自定义的额外生成要求
    """
    
    def __init__(
        self, 
        llm_serving: LLMServingABC, 
        number: int = 5,
        custom_prompt: str = ""
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.number = number
        self.custom_prompt = custom_prompt
        self.prompts = TextQuestionGeneratorPrompt(
            number=number, 
            custom_prompt=custom_prompt
        )
        self.logger.info(f'Initialized {self.__class__.__name__} with number={number}')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "从给定文本中生成多个高质量问题，适用于构建 SFT 数据集。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- number：每段文本生成的问题数量，默认为 5\n"
                "- custom_prompt：用户自定义的额外生成要求\n"
                "- input_key：输入文本字段名，默认为 'text'\n"
                "输出参数：\n"
                "- 包含 'question' 和 'raw_content' 字段的 DataFrame\n"
                "- 每个问题占一行，raw_content 保存原始文本"
            )
        elif lang == "en":
            return (
                "Generate multiple high-quality questions from given text for SFT dataset construction.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- number: Number of questions to generate per text, default is 5\n"
                "- custom_prompt: User-defined additional generation requirements\n"
                "- input_key: Input text field name, default is 'text'\n"
                "Output Parameters:\n"
                "- DataFrame containing 'question' and 'raw_content' fields\n"
                "- Each question is one row, raw_content stores the original text"
            )
        else:
            return "TextQuestionGenerator generates multiple questions from text for SFT datasets."

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str = "text",
        output_question_key: str = "question",
        output_content_key: str = "raw_content"
    ):
        """
        运行问题生成算子
        
        Args:
            storage: DataFlow 存储对象
            input_key: 输入文本的字段名，默认为 'text'
            output_question_key: 输出问题的字段名，默认为 'question'
            output_content_key: 输出原始内容的字段名，默认为 'raw_content'
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        # 读取数据
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} rows")
        
        # 准备 LLM 输入
        llm_inputs = []
        for _, row in dataframe.iterrows():
            text = row.get(input_key, '')
            llm_input = self.prompts.build_prompt(text=text)
            llm_inputs.append(llm_input)
        
        # 调用 LLM 生成问题
        self.logger.info("Generating questions...")
        outputs = self.llm_serving.generate_from_input(llm_inputs)
        self.logger.info("Question generation completed.")
        
        # 解析输出并展开为多行
        result_records = []
        for idx, output in enumerate(outputs):
            questions = extract_json_array(output)
            raw_content = dataframe[input_key].iloc[idx]
            
            for question in questions:
                if isinstance(question, str) and question.strip():
                    result_records.append({
                        output_question_key: question.strip(),
                        output_content_key: raw_content
                    })
        
        # 保存结果
        output_df = pd.DataFrame(result_records)
        storage.write(output_df)
        self.logger.info(f"Generated {len(output_df)} questions from {len(dataframe)} texts")
        
        return [output_question_key, output_content_key]

