"""
EvolInstructGenerator: 基于 Evol-Instruct 方法的指令进化算子

用途：将简单指令进化为更复杂的版本，用于生成高质量的 SFT 训练数据。
参考论文：WizardLM: Empowering Large Language Models to Follow Complex Instructions

进化策略：
1. 深度进化 (Depth Evolution):
   - 添加约束 (Constraints): 给指令添加更多限制条件
   - 深化问题 (Deepen): 增加问题的深度和广度
   - 具体化 (Concretizing): 将通用概念替换为具体概念
   - 多步推理 (Reasoning): 要求多步推理过程
2. 广度进化 (Breadth Evolution):
   - 创建同领域新问题: 基于原问题创建相关但不同的问题
"""

import random
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import EvolInstructPrompt
from dataflow.core.prompt import prompt_restrict


@prompt_restrict(EvolInstructPrompt)
@OPERATOR_REGISTRY.register()
class EvolInstructGenerator(OperatorABC):
    """
    Evol-Instruct 指令进化算子：将简单指令进化为更复杂的版本

    参数：
        llm_serving: LLM 服务对象，需实现 LLMServingABC 接口
        evolution_type: 进化类型，'depth'/'breadth'/'all'
        depth_strategies: 深度进化策略列表
        generate_answer: 是否同时生成答案
        num_evolutions: 每条指令进化次数
        use_context: 是否使用上下文约束进化（启用后进化的问题会严格基于上下文）
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        evolution_type: str = 'all',
        depth_strategies: list = None,
        generate_answer: bool = True,
        num_evolutions: int = 1,
        use_context: bool = False
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.evolution_type = evolution_type
        self.depth_strategies = depth_strategies
        self.generate_answer = generate_answer
        self.num_evolutions = num_evolutions
        self.use_context = use_context
        self.prompts = EvolInstructPrompt(
            evolution_type=evolution_type,
            depth_strategies=depth_strategies,
            use_context=use_context
        )
        self.logger.info(
            f'Initialized {self.__class__.__name__} with '
            f'evolution_type={evolution_type}, generate_answer={generate_answer}, use_context={use_context}'
        )
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于 Evol-Instruct 方法将简单指令进化为更复杂的版本，适用于构建高质量 SFT 数据集。\n"
                "特点：\n"
                "- 深度进化：添加约束、深化问题、具体化、多步推理\n"
                "- 广度进化：创建同领域新问题\n"
                "- 可选择是否同时生成答案\n"
                "- 支持基于上下文约束进化（启用 use_context 后，进化不会脱离原始上下文）\n\n"
                "初始化参数：\n"
                "- llm_serving：LLM 服务对象\n"
                "- evolution_type：进化类型，'depth'/'breadth'/'all'\n"
                "- depth_strategies：深度进化策略列表\n"
                "- generate_answer：是否生成答案，默认 True\n"
                "- num_evolutions：每条指令进化次数，默认 1\n"
                "- use_context：是否使用上下文约束进化，默认 False\n\n"
                "运行参数：\n"
                "- input_instruction_key：输入指令字段名，默认 'instruction'\n"
                "- input_context_key：输入上下文字段名，默认 'context'（启用 use_context 时使用）\n"
                "- output_instruction_key：输出进化指令字段名，默认 'evolved_instruction'\n"
                "- output_answer_key：输出答案字段名，默认 'output'\n"
            )
        elif lang == "en":
            return (
                "Evolve simple instructions into complex versions using Evol-Instruct method.\n"
                "Features:\n"
                "- Depth evolution: constraints, deepen, concretizing, reasoning\n"
                "- Breadth evolution: create new problems in same domain\n"
                "- Optional answer generation\n"
                "- Support context-constrained evolution (with use_context=True)\n\n"
                "Init Parameters:\n"
                "- llm_serving: LLM serving object\n"
                "- evolution_type: 'depth'/'breadth'/'all'\n"
                "- depth_strategies: list of depth strategies\n"
                "- generate_answer: whether to generate answers, default True\n"
                "- num_evolutions: number of evolutions per instruction, default 1\n"
                "- use_context: whether to use context constraint, default False\n\n"
                "Run Parameters:\n"
                "- input_instruction_key: input instruction field, default 'instruction'\n"
                "- input_context_key: input context field, default 'context' (used when use_context=True)\n"
                "- output_instruction_key: output evolved instruction field\n"
                "- output_answer_key: output answer field, default 'output'\n"
            )
        else:
            return "EvolInstructGenerator evolves instructions using Evol-Instruct method."

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        input_context_key: str = "context",
        output_instruction_key: str = "evolved_instruction",
        output_answer_key: str = "output",
        output_original_key: str = "original_instruction"
    ):
        """
        运行 Evol-Instruct 指令进化算子

        Args:
            storage: DataFlow 存储对象
            input_instruction_key: 输入指令字段名
            input_context_key: 输入上下文字段名（启用 use_context 时用于约束进化范围）
            output_instruction_key: 输出进化指令字段名
            output_answer_key: 输出答案字段名
            output_original_key: 输出原始指令字段名

        Returns:
            输出字段名列表
        """
        self.logger.info(f"Running {self.__class__.__name__}...")

        # 读取数据
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} instructions")

        # 准备进化 prompts
        evol_prompts = []
        original_instructions = []

        for _, row in dataframe.iterrows():
            instruction = row.get(input_instruction_key, '')
            # 获取上下文（如果启用了 use_context）
            context = None
            if self.use_context and input_context_key in row:
                context = str(row[input_context_key]).strip() if row[input_context_key] else None

            # 为每条指令生成 num_evolutions 个进化 prompt
            for _ in range(self.num_evolutions):
                evol_prompt = self.prompts.build_prompt(instruction=instruction, context=context)
                evol_prompts.append(evol_prompt)
                original_instructions.append(instruction)
        
        # 调用 LLM 进行指令进化
        self.logger.info(f"Evolving {len(evol_prompts)} instructions...")
        evolved_instructions = self.llm_serving.generate_from_input(evol_prompts)
        self.logger.info("Instruction evolution completed.")

        # 清理进化后的指令（去除可能的前缀标记）
        cleaned_instructions = []
        for ei in evolved_instructions:
            # 清理可能残留的标记（支持中英文）
            ei = ei.strip()
            markers = [
                '#Rewritten Prompt#:', '#Created Prompt#:',
                'Rewritten Prompt:', 'Created Prompt:',
                '#改写后的问题#:', '#创作的问题#:',
                '改写后的问题:', '创作的问题:'
            ]
            for marker in markers:
                if ei.startswith(marker):
                    ei = ei[len(marker):].strip()
            cleaned_instructions.append(ei)

        # 如果需要生成答案
        answers = []
        if self.generate_answer:
            self.logger.info("Generating answers for evolved instructions...")
            answers = self.llm_serving.generate_from_input(cleaned_instructions)
            self.logger.info("Answer generation completed.")

        # 构建结果
        result_records = []
        for idx, (evol_inst, orig_inst) in enumerate(zip(cleaned_instructions, original_instructions)):
            record = {
                output_instruction_key: evol_inst,
                output_original_key: orig_inst
            }
            if self.generate_answer and idx < len(answers):
                record[output_answer_key] = answers[idx]
            result_records.append(record)

        # 保存结果
        output_df = pd.DataFrame(result_records)
        storage.write(output_df)
        self.logger.info(f"Generated {len(output_df)} evolved instructions")

        output_keys = [output_instruction_key, output_original_key]
        if self.generate_answer:
            output_keys.append(output_answer_key)
        return output_keys

