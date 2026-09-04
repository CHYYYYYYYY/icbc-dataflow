"""
DPO (Direct Preference Optimization) 相关的 Prompt 模板
参考 distilabel 的 UltraFeedback 实现
"""
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class DPOUltraFeedbackPrompt(PromptABC):
    """
    UltraFeedback 风格的评估 Prompt，用于对多个 LLM 生成的回复进行评分。
    支持多个评估维度：overall-rating, helpfulness, honesty, instruction-following, truthfulness
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
    
    def __init__(self, aspect: str = "overall-rating"):
        super().__init__()
        if aspect not in self.ASPECTS:
            raise ValueError(f"不支持的评估维度: {aspect}. 支持的维度: {list(self.ASPECTS.keys())}")
        self.aspect = aspect
        self.aspect_info = self.ASPECTS[aspect]
    
    def build_system_prompt(self) -> str:
        return f"""你是一个专业的文本质量评估专家。你的任务是评估 AI 助手对指令的多个回复。
评估维度: {self.aspect_info['description']}

{self.aspect_info['criteria']}

你需要为每个回复给出:
1. 一个 1-5 的整数评分
2. 简短的评分理由

请严格按照 JSON 格式输出你的评估结果。"""

    def build_prompt(self, instruction: str, generations: list) -> str:
        """
        构建评估 prompt
        
        Args:
            instruction: 原始指令
            generations: 多个生成的回复列表
        
        Returns:
            格式化的评估 prompt
        """
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

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "UltraFeedback 风格的 DPO 评估 Prompt，用于对多个 LLM 生成的回复进行评分。"
                "支持 overall-rating, helpfulness, honesty, instruction-following, truthfulness 等评估维度。"
            )
        else:
            return (
                "UltraFeedback-style DPO evaluation prompt for rating multiple LLM responses. "
                "Supports aspects: overall-rating, helpfulness, honesty, instruction-following, truthfulness."
            )