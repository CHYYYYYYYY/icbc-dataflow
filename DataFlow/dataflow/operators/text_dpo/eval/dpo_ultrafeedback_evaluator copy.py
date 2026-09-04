"""
DPO UltraFeedback 评估算子
参考 distilabel 的 UltraFeedback 任务实现
用于对多个生成的回复进行质量评分
"""
import json
import re
from typing import List, Optional
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC


@OPERATOR_REGISTRY.register()
class DPOUltraFeedbackEvaluator(OperatorABC):
    """
    DPO UltraFeedback 评估算子
    
    功能：
    - 使用 LLM 对多个生成的回复进行质量评分
    - 支持多个评估维度：overall-rating, helpfulness, honesty, instruction-following, truthfulness
    - 输出每个回复的评分和评分理由
    
    用法示例：
    ```python
    from dataflow.serving import VLLMServing
    
    evaluator = DPOUltraFeedbackEvaluator(
        llm_serving=VLLMServing(model_path="judge_model"),
        aspect="overall-rating"
    )
    ```
    """
    
    ASPECTS = {
        "overall-rating": {
            "description": "综合评估回复的整体质量",
            "criteria": """评估标准：
1分 - 回复完全不相关、有害或完全错误
2分 - 回复有重大问题，信息错误或不完整
3分 - 回复基本正确但有明显改进空间
4分 - 回复质量良好，准确且有帮助
5分 - 回复优秀，完全满足要求，信息准确、全面、有帮助"""
        },
        "helpfulness": {
            "description": "评估回复对用户的帮助程度",
            "criteria": """评估标准：
1分 - 完全没有帮助，无法解决用户问题
2分 - 帮助有限，只解决了部分问题
3分 - 有一定帮助，但缺乏深度或细节
4分 - 很有帮助，基本满足用户需求
5分 - 极其有帮助，超出预期地解决了问题"""
        },
        "honesty": {
            "description": "评估回复的诚实度和不确定性表达",
            "criteria": """评估标准：
1分 - 自信地给出错误信息
2分 - 给出不确定的错误信息
3分 - 给出正确信息但表达过度不确定
4分 - 诚实地表达不确定性，信息基本正确
5分 - 完全诚实，正确表达确定性/不确定性"""
        },
        "instruction-following": {
            "description": "评估回复是否严格遵循指令要求",
            "criteria": """评估标准：
1分 - 完全忽视指令要求
2分 - 部分遵循指令，但有重要遗漏
3分 - 基本遵循指令，有小的偏差
4分 - 良好地遵循指令要求
5分 - 完美地遵循所有指令要求"""
        },
        "truthfulness": {
            "description": "评估回复内容的真实性和准确性",
            "criteria": """评估标准：
1分 - 包含严重的事实错误或虚假信息
2分 - 包含多处事实错误
3分 - 基本准确但有轻微错误
4分 - 信息准确可靠
5分 - 完全准确，引用可靠来源"""
        }
    }
    
    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        aspect: str = "overall-rating"
    ):
        """
        初始化 UltraFeedback 评估器
        
        Args:
            llm_serving: 用于评估的 LLM 服务实例
            aspect: 评估维度，可选值：overall-rating, helpfulness, honesty, 
                   instruction-following, truthfulness
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        
        if aspect not in self.ASPECTS:
            raise ValueError(f"不支持的评估维度: {aspect}. 支持的维度: {list(self.ASPECTS.keys())}")
        
        self.llm_serving = llm_serving
        self.aspect = aspect
        self.aspect_info = self.ASPECTS[aspect]
        
        self.logger.info(f'{self.__class__.__name__} initialized with aspect: {aspect}')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "DPO UltraFeedback 评估算子。使用 LLM 对多个生成的回复进行质量评分。\n"
                "输入参数：\n"
                "- llm_serving: 用于评估的 LLM 服务实例\n"
                "- aspect: 评估维度 (overall-rating, helpfulness, honesty, instruction-following, truthfulness)\n"
                "输出参数：\n"
                "- 包含 ratings, rationales 字段的 DataFrame"
            )
        else:
            return (
                "DPO UltraFeedback evaluator. Uses LLM to rate multiple generated responses.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving instance for evaluation\n"
                "- aspect: Evaluation aspect (overall-rating, helpfulness, honesty, instruction-following, truthfulness)\n"
                "Output Parameters:\n"
                "- DataFrame containing ratings, rationales fields"
            )

    def _build_evaluation_prompt(self, instruction: str, generations: List[str]) -> str:
        """构建评估 prompt"""
        generations_text = ""
        for i, gen in enumerate(generations, 1):
            generations_text += f"\n### 回复 {i}:\n{gen}\n"
        
        return f"""请评估以下指令的多个回复:

## 指令:
{instruction}

## 待评估的回复:
{generations_text}

请对每个回复进行评分，严格按以下 JSON 格式输出:
{{
    "ratings": [回复1评分, 回复2评分, ...],
    "rationales": ["回复1的评分理由", "回复2的评分理由", ...]
}}

注意: ratings 中的评分必须是 1-5 的整数。"""

    def _build_system_prompt(self) -> str:
        """构建系统 prompt"""
        return f"""你是一个专业的文本质量评估专家。你的任务是评估 AI 助手对指令的多个回复。
评估维度: {self.aspect_info['description']}

{self.aspect_info['criteria']}

你需要为每个回复给出:
1. 一个 1-5 的整数评分
2. 简短的评分理由

请严格按照 JSON 格式输出你的评估结果。"""

    def _parse_evaluation_response(self, response: str, num_generations: int) -> dict:
        """
        解析 LLM 的评估响应
        
        Args:
            response: LLM 的原始响应
            num_generations: 期望的评分数量
        
        Returns:
            包含 ratings 和 rationales 的字典
        """
        default_result = {
            "ratings": [3] * num_generations,
            "rationales": ["解析失败，使用默认评分"] * num_generations
        }
        
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                ratings = result.get("ratings", [])
                rationales = result.get("rationales", [])
                
                # 验证和修正 ratings
                validated_ratings = []
                for r in ratings:
                    if isinstance(r, (int, float)) and 1 <= r <= 5:
                        validated_ratings.append(int(r))
                    else:
                        validated_ratings.append(3)
                
                # 确保数量匹配
                while len(validated_ratings) < num_generations:
                    validated_ratings.append(3)
                validated_ratings = validated_ratings[:num_generations]
                
                # 确保 rationales 数量匹配
                while len(rationales) < num_generations:
                    rationales.append("无评分理由")
                rationales = rationales[:num_generations]
                
                return {
                    "ratings": validated_ratings,
                    "rationales": rationales
                }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to parse evaluation response: {e}")
        
        return default_result

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        input_generations_key: str = "generations",
        output_ratings_key: str = "ratings",
        output_rationales_key: str = "rationales"
    ):
        """
        运行评估算子
        
        Args:
            storage: DataFlow 存储对象
            input_instruction_key: 输入指令字段名
            input_generations_key: 输入生成列表字段名
            output_ratings_key: 输出评分列表字段名
            output_rationales_key: 输出评分理由列表字段名
        
        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")
        
        # 读取输入数据
        dataframe = storage.read("dataframe")
        
        if input_instruction_key not in dataframe.columns:
            raise ValueError(f"输入数据缺少 '{input_instruction_key}' 字段")
        if input_generations_key not in dataframe.columns:
            raise ValueError(f"输入数据缺少 '{input_generations_key}' 字段")
        
        instructions = dataframe[input_instruction_key].tolist()
        generations_list = dataframe[input_generations_key].tolist()
        
        self.logger.info(f"Evaluating {len(instructions)} instruction-response pairs...")
        
        # 构建评估 prompts
        system_prompt = self._build_system_prompt()
        evaluation_prompts = []
        for instruction, generations in zip(instructions, generations_list):
            prompt = self._build_evaluation_prompt(instruction, generations)
            evaluation_prompts.append(prompt)
        
        # 批量评估
        responses = self.llm_serving.generate_from_input(
            user_inputs=evaluation_prompts,
            system_prompt=system_prompt
        )
        
        # 解析评估结果
        all_ratings = []
        all_rationales = []
        for i, (response, generations) in enumerate(zip(responses, generations_list)):
            result = self._parse_evaluation_response(response, len(generations))
            all_ratings.append(result["ratings"])
            all_rationales.append(result["rationales"])
        
        # 更新 dataframe
        dataframe[output_ratings_key] = all_ratings
        dataframe[output_rationales_key] = all_rationales
        
        # 写入存储
        storage.write(dataframe)
        
        self.logger.info(f"Evaluation completed for {len(instructions)} samples.")
        
        return [output_ratings_key, output_rationales_key]