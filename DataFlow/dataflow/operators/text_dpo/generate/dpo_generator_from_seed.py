"""
DPO 多模型回复生成算子
参考 distilabel 的 TextGeneration 和 DataFlow 的 SFTGeneratorSeed 逻辑
用于从文本 chunk 生成 question，然后让多个模型生成 answer，作为 DPO 训练的基础数据
"""
import re
import json
from typing import List, Union
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC


def extract_json_object(model_output: str):
    """提取第一个包含 instruction 字段的 JSON 对象"""
    json_pattern = r'\{[^}]*\}'
    matches = re.findall(json_pattern, model_output)
    for match in matches:
        try:
            obj = json.loads(match)
            if 'instruction' in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


# Question 生成的 Prompt 模板（支持多语言，模型会根据输入内容语言自动适配）
QUESTION_GENERATION_PROMPT = """You are tasked with creating high-quality question based on the provided context.
Please generate one question based on the provided context, focusing on diversity, relevance, and clarity.

Requirements:
1. Generate exactly one distinct and well-formed question.
2. The question must be based on the context and include enough background for clarity.
3. The question language should match the context language (e.g., Chinese context → Chinese question).
4. Output must follow this JSON format:
{{
    "instruction": "QUESTION"
}}

Examples:
{{
    "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points."
}}
{{
    "instruction": "机器学习中的监督学习和无监督学习有什么区别？请结合实际应用场景进行说明。"
}}
{{
    "instruction": "How can we use Python to calculate the GCD (greatest common divisor) of five numbers and express each number in terms of the GCD?"
}}

{custom_section}

Now, based on the following context, please generate one question:
"""

# Answer 生成的 Prompt 模板（支持多语言）
ANSWER_GENERATION_PROMPT = """Based on the following context, please answer the question accurately and comprehensively.
Your answer language should match the question language.

Context:
{context}

Question:
{question}

Please provide a detailed and helpful answer:"""


@OPERATOR_REGISTRY.register()
class DPOResponseGenerator(OperatorABC):
    """
    DPO 多模型回复生成算子（从文本 chunk 生成 QA 对）
    
    功能：
    - 从文本 chunk 生成 question（使用 question_llm_serving）
    - 让多个模型根据文本 chunk 内容生成多个 answer
    - 支持使用多个 LLM serving 实例生成多样化回复
    """
    
    def __init__(
        self,
        question_llm_serving: LLMServingABC = None,
        answer_llm_serving_list: Union[List[LLMServingABC], LLMServingABC] = None,
        num_generations_per_model: int = 1,
        custom_prompt: str = None,
        model_names: List[str] = None
    ):
        """
        初始化 DPO 回复生成器
        
        Args:
            question_llm_serving: 用于生成 question 的 LLM 服务实例
            answer_llm_serving_list: 用于生成 answer 的 LLM 服务列表
            num_generations_per_model: 每个模型为每条指令生成的回复数量
            custom_prompt: 自定义的 question 生成提示词
            model_names: 可选的模型名称列表，用于标识每个模型
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        
        self.question_llm_serving = question_llm_serving
        
        if answer_llm_serving_list is None:
            self.answer_llm_serving_list = []
        elif isinstance(answer_llm_serving_list, list):
            self.answer_llm_serving_list = answer_llm_serving_list
        else:
            self.answer_llm_serving_list = [answer_llm_serving_list]
        
        self.num_generations_per_model = num_generations_per_model
        self.custom_prompt = custom_prompt or ""
        self.model_names = model_names or [f"model_{i}" for i in range(len(self.answer_llm_serving_list))]
        
        self.logger.info(f'{self.__class__.__name__} initialized with {len(self.answer_llm_serving_list)} answer models.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "DPO 多模型回复生成算子。从文本 chunk 生成 question，然后让多个模型生成 answer。\n"
                "输入参数：\n"
                "- question_llm_serving：用于生成问题的 LLM 服务\n"
                "- answer_llm_serving_list：用于生成答案的 LLM 服务列表\n"
                "- num_generations_per_model：每个模型生成的答案数量\n"
                "- custom_prompt：自定义问题生成提示词\n"
                "- model_names：模型名称列表\n"
                "- input_key：输入文本 chunk 字段名\n"
                "- output_instruction_key：输出问题字段名\n"
                "- output_generations_key：输出回复列表字段名\n"
                "- output_models_key：输出模型名称列表字段名"
            )
        else:
            return (
                "DPO multi-model response generator. Generates question from text chunk, "
                "then generates multiple answers using different models."
            )

    def _build_question_prompt(self, content: str) -> str:
        """构建生成 question 的 prompt"""
        custom_section = f"Additional instruction:\n{self.custom_prompt}\n" if self.custom_prompt else ""
        full_prompt = QUESTION_GENERATION_PROMPT.format(custom_section=custom_section)
        return f"<|im_start|>system\n{full_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant"

    def _build_answer_prompt(self, context: str, question: str) -> str:
        """构建生成 answer 的 prompt"""
        return ANSWER_GENERATION_PROMPT.format(context=context, question=question)

    def _generate_questions(self, raw_contents: List[str]) -> List[str]:
        """从文本 chunk 生成 questions"""
        question_prompts = [self._build_question_prompt(content) for content in raw_contents]
        
        self.logger.info(f"Generating questions for {len(raw_contents)} chunks...")
        outputs = self.question_llm_serving.generate_from_input(question_prompts)
        
        questions = []
        for idx, output in enumerate(outputs):
            result = extract_json_object(output)
            if result and 'instruction' in result:
                questions.append(result['instruction'])
            else:
                questions.append(f"请根据以下内容回答问题：{raw_contents[idx][:100]}...")
                self.logger.warning(f"Failed to parse question for index {idx}, using default.")
        
        return questions

    def _generate_answers(self, raw_contents: List[str], questions: List[str]) -> List[dict]:
        """让多个模型生成 answers"""
        results = [{"generations": [], "generation_models": []} for _ in questions]
        
        answer_prompts = [
            self._build_answer_prompt(context=content, question=question)
            for content, question in zip(raw_contents, questions)
        ]
        
        for model_idx, llm_serving in enumerate(self.answer_llm_serving_list):
            model_name = self.model_names[model_idx] if model_idx < len(self.model_names) else f"model_{model_idx}"
            
            for gen_idx in range(self.num_generations_per_model):
                self.logger.info(f"Generating answers with {model_name} ({gen_idx + 1}/{self.num_generations_per_model})...")
                
                responses = llm_serving.generate_from_input(answer_prompts)
                
                for i, response in enumerate(responses):
                    results[i]["generations"].append(response.strip() if response else "")
                    results[i]["generation_models"].append(model_name)
        
        return results

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "raw_content",
        output_instruction_key: str = "instruction",
        output_generations_key: str = "generations",
        output_models_key: str = "generation_models"
    ):
        """
        运行回复生成算子
        
        Args:
            storage: DataFlow 存储对象
            input_key: 输入文本 chunk 的字段名，默认为 'raw_content'
            output_instruction_key: 输出问题的字段名，默认为 'instruction'
            output_generations_key: 输出回复列表的字段名，默认为 'generations'
            output_models_key: 输出模型名称列表的字段名，默认为 'generation_models'
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        dataframe = storage.read("dataframe")
        
        if input_key not in dataframe.columns:
            raise ValueError(f"输入数据缺少 '{input_key}' 字段")
        
        raw_contents = dataframe[input_key].tolist()
        self.logger.info(f"Processing {len(raw_contents)} text chunks...")
        
        # Step 1: 生成 questions
        questions = self._generate_questions(raw_contents)
        self.logger.info(f"Generated {len(questions)} questions.")
        
        # Step 2: 让多个模型生成 answers
        generation_results = self._generate_answers(raw_contents, questions)
        
        # 更新 dataframe
        dataframe[output_instruction_key] = questions
        dataframe[output_generations_key] = [r["generations"] for r in generation_results]
        dataframe[output_models_key] = [r["generation_models"] for r in generation_results]
        
        storage.write(dataframe)
        
        total_generations = sum(len(r["generations"]) for r in generation_results)
        self.logger.info(f"Generated {len(questions)} questions and {total_generations} answers for {len(raw_contents)} chunks.")
        
        return [output_instruction_key, output_generations_key, output_models_key]