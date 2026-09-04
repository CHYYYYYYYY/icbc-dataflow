"""
DPO 多模型回复生成算子
参考 distilabel 的 TextGeneration 和多模型生成策略
用于为每个指令生成多个候选回复，作为 DPO 训练的基础数据
"""
import json
from typing import List, Optional, Union
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC


@OPERATOR_REGISTRY.register()
class DPOResponseGenerator(OperatorABC):
    """
    DPO 多模型/多参数回复生成算子
    
    功能：
    - 为每个指令生成多个候选回复
    - 支持使用多个 LLM serving 实例（模拟多模型）
    - 支持使用不同的生成参数（如 temperature）来增加多样性
    - 输出包含 instruction 和 generations 列表的数据
    
    用法示例：
    ```python
    from dataflow.serving import VLLMServing
    
    # 创建多个 serving 实例
    llm_serving_list = [
        VLLMServing(model_path="model_a", temperature=0.3),
        VLLMServing(model_path="model_a", temperature=0.9),
        VLLMServing(model_path="model_b", temperature=0.7),
    ]
    
    generator = DPOResponseGenerator(
        llm_serving_list=llm_serving_list,
        num_generations_per_model=1
    )
    ```
    """
    
    def __init__(
        self,
        llm_serving_list: Union[List[LLMServingABC], LLMServingABC] = None,
        num_generations_per_model: int = 1,
        system_prompt: str = None,
        model_names: List[str] = None
    ):
        """
        初始化 DPO 回复生成器
        
        Args:
            llm_serving_list: LLM 服务列表，可以是单个或多个 LLMServingABC 实例
            num_generations_per_model: 每个模型为每条指令生成的回复数量
            system_prompt: 可选的系统提示词
            model_names: 可选的模型名称列表，用于标识每个模型
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        
        # 处理 llm_serving_list
        if llm_serving_list is None:
            self.llm_serving_list = []
        elif isinstance(llm_serving_list, list):
            self.llm_serving_list = llm_serving_list
        else:
            self.llm_serving_list = [llm_serving_list]
        
        self.num_generations_per_model = num_generations_per_model
        self.system_prompt = system_prompt or "你是一个有帮助的AI助手。请根据用户的问题给出详细、准确的回答。"
        
        # 设置模型名称
        if model_names:
            self.model_names = model_names
        else:
            self.model_names = [f"model_{i}" for i in range(len(self.llm_serving_list))]
        
        self.logger.info(f'{self.__class__.__name__} initialized with {len(self.llm_serving_list)} models.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "DPO 多模型回复生成算子。为每个指令使用多个模型或不同参数生成多个候选回复。\n"
                "输入参数：\n"
                "- llm_serving_list: LLM服务列表，支持多个模型实例\n"
                "- num_generations_per_model: 每个模型生成的回复数量，默认为1\n"
                "- system_prompt: 系统提示词\n"
                "- model_names: 模型名称列表\n"
                "输出参数：\n"
                "- 包含 instruction, generations, generation_models 字段的 DataFrame"
            )
        else:
            return (
                "DPO multi-model response generator. Generates multiple candidate responses "
                "for each instruction using multiple models or different parameters.\n"
                "Input Parameters:\n"
                "- llm_serving_list: List of LLM serving instances\n"
                "- num_generations_per_model: Number of responses per model, default 1\n"
                "- system_prompt: System prompt\n"
                "- model_names: List of model names\n"
                "Output Parameters:\n"
                "- DataFrame containing instruction, generations, generation_models fields"
            )

    def _generate_responses(self, instructions: List[str]) -> List[dict]:
        """
        为指令列表生成多个回复
        
        Returns:
            包含 generations 和 generation_models 的字典列表
        """
        results = []
        
        # 初始化每条指令的结果
        for instruction in instructions:
            results.append({
                "generations": [],
                "generation_models": []
            })
        
        # 对每个 LLM serving 实例生成回复
        for model_idx, llm_serving in enumerate(self.llm_serving_list):
            model_name = self.model_names[model_idx] if model_idx < len(self.model_names) else f"model_{model_idx}"
            
            for gen_idx in range(self.num_generations_per_model):
                self.logger.info(f"Generating responses with {model_name} (generation {gen_idx + 1}/{self.num_generations_per_model})...")
                
                # 批量生成回复
                responses = llm_serving.generate_from_input(
                    user_inputs=instructions,
                    system_prompt=self.system_prompt
                )
                
                # 将回复添加到结果中
                for i, response in enumerate(responses):
                    results[i]["generations"].append(response.strip() if response else "")
                    results[i]["generation_models"].append(model_name)
        
        return results

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        output_generations_key: str = "generations",
        output_models_key: str = "generation_models"
    ):
        """
        运行回复生成算子
        
        Args:
            storage: DataFlow 存储对象
            input_instruction_key: 输入指令字段名
            output_generations_key: 输出生成列表字段名
            output_models_key: 输出模型名称列表字段名
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        # 读取输入数据
        dataframe = storage.read("dataframe")
        
        if input_instruction_key not in dataframe.columns:
            raise ValueError(f"输入数据缺少 '{input_instruction_key}' 字段")
        
        instructions = dataframe[input_instruction_key].tolist()
        self.logger.info(f"Generating responses for {len(instructions)} instructions...")
        
        # 生成回复
        generation_results = self._generate_responses(instructions)
        
        # 更新 dataframe
        dataframe[output_generations_key] = [r["generations"] for r in generation_results]
        dataframe[output_models_key] = [r["generation_models"] for r in generation_results]
        
        # 写入存储
        storage.write(dataframe)
        
        total_generations = sum(len(r["generations"]) for r in generation_results)
        self.logger.info(f"Generated {total_generations} total responses for {len(instructions)} instructions.")
        
        return [output_generations_key, output_models_key]