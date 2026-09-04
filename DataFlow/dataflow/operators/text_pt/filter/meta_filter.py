import numpy as np
import pandas as pd
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import LLMServingABC
from dataflow.operators.text_pt import MetaSampleEvaluator


@OPERATOR_REGISTRY.register()
class MetaFilter(OperatorABC):
    """
    Filter data based on MetaSampleEvaluator multi-dimensional quality scores.
    Combines instruction, input, and output fields for evaluation.
    """

    def __init__(self, 
                 min_score: float = 3.0, 
                 max_score: float = 5.0, 
                 llm_serving: LLMServingABC = None, 
                 dimensions: list[dict] = None):
        """
        Initialize MetaFilter.
        
        Args:
            min_score: Minimum average score threshold, default is 3.0
            max_score: Maximum average score threshold, default is 5.0
            llm_serving: LLM serving instance
            dimensions: Custom dimensions list for MetaPrompt evaluation.
                       If None, uses default dimensions from MetaSampleEvaluator.
        """
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.dimensions = dimensions
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")
        
        # Initialize MetaSampleEvaluator with custom dimensions if provided
        if dimensions is not None:
            self.scorer = MetaSampleEvaluator(llm_serving=llm_serving, dimensions=dimensions)
        else:
            self.scorer = MetaSampleEvaluator(llm_serving=llm_serving)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于MetaSampleEvaluator多维度打分器的得分对数据进行过滤。通过LLM评估问答对的多个维度质量，返回综合质量得分。\n\n"
                "初始化参数：\n"
                "- min_score: 最低平均分数阈值，默认为3.0\n"
                "- max_score: 最高平均分数阈值，默认为5.0\n"
                "- llm_serving: LLM服务实例\n"
                "- dimensions: MetaPrompt的自定义维度列表，如果为None则使用默认维度\n\n"
                "运行参数：\n"
                "- input_instruction_key: 输入指令字段名（如问题）\n"
                "- input_input_key: 输入内容字段名（如上下文）\n"
                "- input_output_key: 输出内容字段名（如答案）\n"
                "- output_key: 输出分数字段名，默认为'MetaScore'（六个维度得分的均值）\n\n"
                "过滤逻辑：保留平均分数在[min_score, max_score]范围内的数据"
            )
        else:
            return (
                "Filter data using scores from the MetaSampleEvaluator. Evaluate QA pair quality across multiple dimensions using LLM and return a comprehensive quality score.\n\n"
                "Initialization Parameters:\n"
                "- min_score: Minimum average score threshold, default is 3.0\n"
                "- max_score: Maximum average score threshold, default is 5.0\n"
                "- llm_serving: LLM serving instance\n"
                "- dimensions: Custom dimensions list for MetaPrompt, uses default if None\n\n"
                "Run Parameters:\n"
                "- input_instruction_key: Input instruction field name (e.g., question)\n"
                "- input_input_key: Input content field name (e.g., context)\n"
                "- input_output_key: Output content field name (e.g., answer)\n"
                "- output_key: Output score field name, default is 'MetaScore' (mean of 6 dimension scores)\n\n"
                "Filter Logic: Keep data with average scores in [min_score, max_score] range"
            )

    def _combine_fields(self, dataframe: pd.DataFrame, 
                        input_instruction_key: str, 
                        input_input_key: str, 
                        input_output_key: str) -> pd.DataFrame:
        """
        Combine instruction, input, and output fields into a single text field for evaluation.
        
        Format: "Question: {instruction}\nContext: {input}\nAnswer: {output}"
        """
        combined_texts = []
        for _, row in dataframe.iterrows():
            instruction = row.get(input_instruction_key, '')
            input_text = row.get(input_input_key, '')
            output_text = row.get(input_output_key, '')
            
            combined = f"Question: {instruction}\nContext: {input_text}\nAnswer: {output_text}"
            combined_texts.append(combined)
        
        df_copy = dataframe.copy()
        df_copy['_meta_combined_text'] = combined_texts
        return df_copy

    def run(self, 
            storage: DataFlowStorage, 
            input_instruction_key: str, 
            input_input_key: str, 
            input_output_key: str, 
            output_key: str = 'MetaScore'):
        """
        Run the MetaFilter.
        
        Args:
            storage: DataFlow storage object
            input_instruction_key: Field name for instruction/question
            input_input_key: Field name for input/context
            input_output_key: Field name for output/answer
            output_key: Field name for output score (mean of all dimensions)
        """
        self.logger.info(
            f"Running {self.__class__.__name__} with"
            f"input_instruction_key={input_instruction_key}, "
            f"input_input_key={input_input_key}, "
            f"input_output_key={input_output_key}, "
            f"output_key={output_key}..."
        )
        
        dataframe = storage.read("dataframe")
        
        # Combine fields into single text for evaluation
        df_combined = self._combine_fields(
            dataframe, input_instruction_key, input_input_key, input_output_key
        )
        
        # Get multi-dimensional scores using MetaSampleEvaluator
        scores = self.scorer.eval(df_combined, '_meta_combined_text')
        
        # Calculate mean score across all dimensions for each sample
        mean_scores = [np.nanmean(score_list) for score_list in scores]
        
        # Add individual dimension scores and mean score to dataframe
        score_df = pd.DataFrame(scores, columns=self.scorer.output_columns)
        dataframe = pd.concat([dataframe, score_df], axis=1)
        dataframe[output_key] = mean_scores
        
        # Filter based on mean score
        filtered_dataframe = dataframe[
            (dataframe[output_key] >= self.min_score) & 
            (dataframe[output_key] <= self.max_score)
        ]
        
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [output_key] + self.scorer.output_columns

