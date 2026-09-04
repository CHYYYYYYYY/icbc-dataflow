"""
DPO 数据格式化算子
参考 distilabel 的 FormatTextGenerationDPO 实现
将评分后的数据转换为 DPO 训练格式
"""
import hashlib
from typing import List, Optional, Tuple
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd


@OPERATOR_REGISTRY.register()
class FormatDPO(OperatorABC):
    """
    DPO 数据格式化算子
    
    功能：
    - 将包含多个回复和评分的数据转换为 DPO 训练格式
    - 根据评分选择 chosen（最高分）和 rejected（最低分）回复
    - 支持设置最小评分差异阈值，确保 chosen 和 rejected 有足够区分度
    - 输出符合 TRL DPOTrainer 格式的数据
    
    输出格式：
    {
        "prompt": "用户指令",
        "prompt_id": "指令的哈希ID",
        "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "chosen_rating": 5,
        "chosen_model": "model_name",
        "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "rejected_rating": 2,
        "rejected_model": "model_name"
    }
    """
    
    def __init__(
        self,
        min_rating_diff: float = 1.0,
        keep_ties: bool = False
    ):
        """
        初始化 DPO 格式化器
        
        Args:
            min_rating_diff: chosen 和 rejected 之间的最小评分差异，默认为 1.0
            keep_ties: 当最高分和最低分相同时是否保留该样本，默认为 False
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        
        self.min_rating_diff = min_rating_diff
        self.keep_ties = keep_ties
        
        self.logger.info(f'{self.__class__.__name__} initialized with min_rating_diff={min_rating_diff}, keep_ties={keep_ties}')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "DPO 数据格式化算子。将评分后的多回复数据转换为 DPO 训练格式。\n"
                "输入参数：\n"
                "- min_rating_diff: chosen 和 rejected 之间的最小评分差异，默认为 1.0\n"
                "- keep_ties: 当最高分和最低分相同时是否保留该样本，默认为 False\n"
                "输出参数：\n"
                "- 包含 prompt, prompt_id, chosen, rejected, chosen_rating, rejected_rating 等字段的 DataFrame"
            )
        else:
            return (
                "DPO data formatter. Converts rated multi-response data to DPO training format.\n"
                "Input Parameters:\n"
                "- min_rating_diff: Minimum rating difference between chosen and rejected, default 1.0\n"
                "- keep_ties: Whether to keep samples when highest and lowest ratings are equal, default False\n"
                "Output Parameters:\n"
                "- DataFrame containing prompt, prompt_id, chosen, rejected, chosen_rating, rejected_rating fields"
            )

    def _generate_prompt_id(self, prompt: str) -> str:
        """生成 prompt 的唯一 ID"""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _select_chosen_rejected(
        self,
        generations: List[str],
        ratings: List[float],
        models: List[str] = None
    ) -> Optional[Tuple[dict, dict]]:
        """
        根据评分选择 chosen 和 rejected
        
        Args:
            generations: 生成的回复列表
            ratings: 对应的评分列表
            models: 对应的模型名称列表
        
        Returns:
            (chosen_info, rejected_info) 或 None（如果不满足条件）
        """
        if not generations or not ratings:
            return None
        
        if len(generations) != len(ratings):
            self.logger.warning("generations 和 ratings 长度不匹配")
            return None
        
        # 找到最高分和最低分的索引
        max_rating = max(ratings)
        min_rating = min(ratings)
        
        # 检查评分差异
        if max_rating - min_rating < self.min_rating_diff:
            if not self.keep_ties:
                return None
        
        # 选择最高分作为 chosen（如果有多个相同最高分，选第一个）
        chosen_idx = ratings.index(max_rating)
        
        # 选择最低分作为 rejected（如果有多个相同最低分，选最后一个，避免与 chosen 相同）
        rejected_idx = len(ratings) - 1 - ratings[::-1].index(min_rating)
        
        # 如果 chosen 和 rejected 是同一个，尝试选择其他的
        if chosen_idx == rejected_idx and len(generations) > 1:
            # 找第二低分
            sorted_indices = sorted(range(len(ratings)), key=lambda i: ratings[i])
            for idx in sorted_indices:
                if idx != chosen_idx:
                    rejected_idx = idx
                    break
        
        # 如果仍然相同，返回 None
        if chosen_idx == rejected_idx:
            if not self.keep_ties:
                return None
        
        chosen_info = {
            "response": generations[chosen_idx],
            "rating": ratings[chosen_idx],
            "model": models[chosen_idx] if models and chosen_idx < len(models) else "unknown"
        }
        
        rejected_info = {
            "response": generations[rejected_idx],
            "rating": ratings[rejected_idx],
            "model": models[rejected_idx] if models and rejected_idx < len(models) else "unknown"
        }
        
        return chosen_info, rejected_info

    def _format_conversation(self, instruction: str, response: str) -> List[dict]:
        """将指令和回复格式化为对话格式"""
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        input_generations_key: str = "generations",
        input_ratings_key: str = "ratings",
        input_models_key: str = "generation_models",
        output_prompt_key: str = "prompt",
        output_prompt_id_key: str = "prompt_id",
        output_chosen_key: str = "chosen",
        output_rejected_key: str = "rejected",
        output_chosen_rating_key: str = "chosen_rating",
        output_rejected_rating_key: str = "rejected_rating",
        output_chosen_model_key: str = "chosen_model",
        output_rejected_model_key: str = "rejected_model"
    ):
        """
        运行格式化算子
        
        Args:
            storage: DataFlow 存储对象
            input_instruction_key: 输入指令字段名
            input_generations_key: 输入生成列表字段名
            input_ratings_key: 输入评分列表字段名
            input_models_key: 输入模型名称列表字段名
            output_*: 各输出字段名
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        # 读取输入数据
        dataframe = storage.read("dataframe")
        
        # 验证必需字段
        required_keys = [input_instruction_key, input_generations_key, input_ratings_key]
        for key in required_keys:
            if key not in dataframe.columns:
                raise ValueError(f"输入数据缺少 '{key}' 字段")
        
        instructions = dataframe[input_instruction_key].tolist()
        generations_list = dataframe[input_generations_key].tolist()
        ratings_list = dataframe[input_ratings_key].tolist()
        
        # 模型名称是可选的
        if input_models_key in dataframe.columns:
            models_list = dataframe[input_models_key].tolist()
        else:
            models_list = [None] * len(instructions)
        
        self.logger.info(f"Formatting {len(instructions)} samples to DPO format...")
        
        # 格式化数据
        formatted_data = []
        skipped_count = 0
        
        for instruction, generations, ratings, models in zip(
            instructions, generations_list, ratings_list, models_list
        ):
            result = self._select_chosen_rejected(generations, ratings, models)
            
            if result is None:
                skipped_count += 1
                continue
            
            chosen_info, rejected_info = result
            
            formatted_data.append({
                output_prompt_key: instruction,
                output_prompt_id_key: self._generate_prompt_id(instruction),
                output_chosen_key: self._format_conversation(instruction, chosen_info["response"]),
                output_rejected_key: self._format_conversation(instruction, rejected_info["response"]),
                output_chosen_rating_key: chosen_info["rating"],
                output_rejected_rating_key: rejected_info["rating"],
                output_chosen_model_key: chosen_info["model"],
                output_rejected_model_key: rejected_info["model"]
            })
        
        # 创建新的 DataFrame
        result_df = pd.DataFrame(formatted_data)
        
        # 写入存储
        storage.write(result_df)
        
        self.logger.info(
            f"Formatted {len(formatted_data)} DPO samples. "
            f"Skipped {skipped_count} samples due to insufficient rating difference."
        )
        
        return [
            output_prompt_key, output_prompt_id_key,
            output_chosen_key, output_rejected_key,
            output_chosen_rating_key, output_rejected_rating_key,
            output_chosen_model_key, output_rejected_model_key
        ]
