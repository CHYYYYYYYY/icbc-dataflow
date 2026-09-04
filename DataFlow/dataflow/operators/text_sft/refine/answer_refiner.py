import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import AnswerCritiquePrompt, AnswerRefinePrompt
from dataflow.core.prompt import prompt_restrict
from dataflow.utils.meta_label_filters import (
    detect_evaluator_tone,
    detect_question_answer_polarity_issue,
    extract_llm_answer_text,
    has_reverse_phrasing,
    looks_like_answer_as_quote,
    looks_like_fabricated_shell_block,
    scrub_meta_labels as _scrub_meta_labels,
    strip_evaluator_tone,
    strip_training_decorators,
)


@prompt_restrict(
    AnswerCritiquePrompt,
    AnswerRefinePrompt
)
@OPERATOR_REGISTRY.register()
class AnswerRefiner(OperatorABC):
    """
    两阶段优化答案质量：
    第一阶段：根据上下文和问题对生成的答案进行批评
    第二阶段：基于批评反馈改进答案
    
    批评维度（6个）：
    1. 完整性：答案是否包含了原文中的所有关键步骤/要点
    2. 准确性：答案内容是否正确，无事实性错误
    3. 针对性：答案是否直接回答问题，与问题焦点一致
    4. 结构清晰度：答案是否分点分段，结构清晰易读
    5. 原文支撑：答案是否基于原文，非编造内容
    6. 可操作性：对于处理类问题，是否提供具体可执行的步骤
    """
    
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.critique_prompt = AnswerCritiquePrompt()
        self.refine_prompt = AnswerRefinePrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "两阶段优化答案质量：第一阶段根据上下文和问题对生成的答案进行批评，"
                "第二阶段基于批评反馈改进答案。\n\n"
                "批评维度：\n"
                "1. 完整性：答案是否包含所有关键步骤/要点\n"
                "2. 准确性：答案是否正确，无事实性错误\n"
                "3. 针对性：答案是否直接回答问题\n"
                "4. 结构清晰度：答案是否分点分段，易于阅读\n"
                "5. 原文支撑：答案是否基于原文\n"
                "6. 可操作性：是否提供具体可执行的步骤\n\n"
                "初始化参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n\n"
                "运行参数：\n"
                "- input_context_key：输入上下文字段名，默认为'raw_content'\n"
                "- input_question_key：输入问题字段名，默认为'question'\n"
                "- input_answer_key：输入答案字段名，默认为'answer'\n"
                "- output_answer_key：输出优化后答案字段名，默认为'refined_answer'\n"
                "- output_critique_key：输出批评字段名，默认为'answer_critique'\n\n"
                "输出参数：\n"
                "- 包含优化后答案的DataFrame\n"
                "- 返回包含优化后答案字段名的列表"
            )
        elif lang == "en":
            return (
                "Two-stage answer quality optimization: First stage critiques generated answers based on context and question, "
                "second stage refines answers based on critique feedback.\n\n"
                "Critique Dimensions:\n"
                "1. Completeness: Whether answer contains all key steps/points\n"
                "2. Accuracy: Whether answer is correct without factual errors\n"
                "3. Relevance: Whether answer directly addresses the question\n"
                "4. Structure: Whether answer is well-organized and readable\n"
                "5. Source Fidelity: Whether answer is based on source text\n"
                "6. Actionability: Whether answer provides executable steps\n\n"
                "Initialization Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n\n"
                "Run Parameters:\n"
                "- input_context_key: Field name for input context, default is 'raw_content'\n"
                "- input_question_key: Field name for input question, default is 'question'\n"
                "- input_answer_key: Field name for input answer, default is 'answer'\n"
                "- output_answer_key: Field name for refined answer, default is 'refined_answer'\n"
                "- output_critique_key: Field name for critique, default is 'answer_critique'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing refined answers\n"
                "- List containing refined answer field name for subsequent operator reference"
            )
        else:
            return "AnswerRefiner improves answer quality through two-stage critique and refinement process."

    def generate_critique(self, contexts: list, questions: list, answers: list) -> list:
        """批量生成答案批评"""
        critique_prompts = [
            self.critique_prompt.build_prompt(ctx, q, a) 
            for ctx, q, a in zip(contexts, questions, answers)
        ]
        critique_responses = self.llm_serving.generate_from_input(critique_prompts)
        return critique_responses

    def _extract_refined_answer(self, llm_response: str, original_answer: str) -> str:
        """
        从 LLM 响应中提取改进后的答案，使用多种 fallback 策略。

        Args:
            llm_response: LLM 返回的完整响应
            original_answer: 原始答案（用于 fallback）

        Returns:
            提取出的改进后答案
        """
        # reasoning 模型把分析过程放在 <think> 里，prompt 模板的占位符
        # (如 [Improved Answer Start]...[Improved Answer End]) 也会出现在 thinking 块里，
        # 若不先剥离则 Strategy1 会误命中模板占位符而非真实输出。
        raw = llm_response or ""
        has_answer_tag = bool(re.search(r'<answer>', raw, re.IGNORECASE))
        clean = extract_llm_answer_text(raw)
        # 若有 <answer> 标签但内容为空 → LLM 未生成答案，直接返回 None
        if has_answer_tag and not clean:
            return None
        response_to_search = clean if clean else raw

        # 占位符黑名单（prompt 模板里的示例内容，不能当真实答案）
        _PLACEHOLDERS = {'...', '你改进后的答案', '改进后的答案', '优化后的答案'}

        # 策略1: 标准格式匹配 [Improved Answer Start]...[Improved Answer End]
        patterns = [
            # 标准英文方括号
            r'\[Improved Answer Start\](.*?)\[Improved Answer End\]',
            # 中文方括号变体
            r'【Improved Answer Start】(.*?)【Improved Answer End】',
            # 可能有空格的变体
            r'\[\s*Improved\s+Answer\s+Start\s*\](.*?)\[\s*Improved\s+Answer\s+End\s*\]',
            # 简化标记
            r'\[Answer Start\](.*?)\[Answer End\]',
            r'【Answer Start】(.*?)【Answer End】',
            # 带冒号的变体
            r'\[Improved Answer Start\]:\s*(.*?)\[Improved Answer End\]',
            # Markdown 代码块内的标记
            r'```[\s\S]*?\[Improved Answer Start\](.*?)\[Improved Answer End\][\s\S]*?```',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_to_search, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted and extracted not in _PLACEHOLDERS:
                    return extracted

        # 策略2: 尝试提取 [Analysis End] 之后的所有内容
        analysis_end_patterns = [
            r'\[Analysis End\]\s*(.*)',
            r'【Analysis End】\s*(.*)',
            r'\[\s*Analysis\s+End\s*\]\s*(.*)',
        ]

        for pattern in analysis_end_patterns:
            match = re.search(pattern, response_to_search, re.DOTALL | re.IGNORECASE)
            if match:
                after_analysis = match.group(1).strip()
                # 清理可能的结束标记
                after_analysis = re.sub(r'\[/?Improved Answer (?:Start|End)\]', '', after_analysis, flags=re.IGNORECASE)
                after_analysis = re.sub(r'【/?Improved Answer (?:Start|End)】', '', after_analysis)
                after_analysis = after_analysis.strip()
                if after_analysis and len(after_analysis) > 20 and after_analysis not in _PLACEHOLDERS:
                    self.logger.debug(f"Extracted answer using Analysis End fallback")
                    return after_analysis

        # 策略3: 如果响应看起来就是改进后的答案（没有使用标记格式）
        if '[Analysis Start]' in response_to_search or '【Analysis Start】' in response_to_search:
            analysis_patterns = [
                r'\[Analysis End\]\s*\n*(.*)',
                r'【Analysis End】\s*\n*(.*)',
            ]
            for pattern in analysis_patterns:
                match = re.search(pattern, response_to_search, re.DOTALL)
                if match:
                    remaining = match.group(1).strip()
                    if remaining and len(remaining) > 20 and remaining not in _PLACEHOLDERS:
                        self.logger.debug(f"Extracted answer after Analysis section")
                        return remaining

        # 策略4: 已在入口 strip thinking，直接使用 response_to_search
        cleaned_response = response_to_search

        if (len(cleaned_response) > 10 and
            '[Analysis' not in cleaned_response and
            '【Analysis' not in cleaned_response and
            not cleaned_response.startswith('```') and
            cleaned_response not in _PLACEHOLDERS):
            self.logger.debug(f"Using cleaned response as answer (no markers found)")
            return cleaned_response

        # 所有策略都失败，返回 None 表示需要使用原答案
        return None

    @staticmethod
    def _critique_allows_subtractive_repair(critique: str) -> bool:
        """Whether critique asks for deletion / narrowing rather than adding facts."""
        if not isinstance(critique, str):
            return False
        patterns = (
            "无支撑",
            "不准确",
            "元信息",
            "删除",
            "不得",
            "不可",
            "仅能确认",
            "当前材料不足",
            "参考上下文仅",
            "未提供",
            "无依据",
        )
        return any(p in critique for p in patterns)

    def _enforce_fact_closure(
        self,
        refined: str,
        anchor: str,
        critique: str = "",
        context: str = "",
        max_context_ratio: float = 1.5,
    ) -> tuple[str, str]:
        """
        原文长度兜底：精炼后的答案不得超过原文字符数的 max_context_ratio 倍（默认 1.5×），
        与 AnswerRewriterPrompt 的改写阶段约束保持一致。

        若 refined 为空，回退到 anchor（clean_answer）。
        若 refined 超长（> max_context_ratio × context 字符数），同样回退到 anchor。

        返回: (final_answer, reason)
          reason ∈ {"ok", "oversize", "empty"}
        """
        if not refined or not isinstance(refined, str):
            return anchor, "empty"
        refined = refined.strip()
        if not refined:
            return anchor, "empty"

        # 基于原文（context）的长度硬兜底，去除空白后比较字符数
        ctx_len = len("".join(ch for ch in (context or "") if not ch.isspace()))
        ref_len = len("".join(ch for ch in refined if not ch.isspace()))
        if ctx_len > 0 and ref_len > max_context_ratio * ctx_len:
            self.logger.warning(
                "[AnswerRefiner] oversize: refined_len=%d > %.1f × context_len=%d; "
                "falling back to clean_answer.",
                ref_len,
                max_context_ratio,
                ctx_len,
            )
            return anchor, "oversize"

        return refined, "ok"

    def generate_refined_answer(self, contexts: list, questions: list,
                                 answers: list, critiques: list) -> list:
        """批量生成优化后的答案"""
        refine_prompts = [
            self.refine_prompt.build_prompt(ctx, q, a, c)
            for ctx, q, a, c in zip(contexts, questions, answers, critiques)
        ]
        refined_answers = self.llm_serving.generate_from_input(refine_prompts)

        # 提取改进后的答案 + 事实集封闭硬兜底 + 元信息扫尾
        extracted_answers: list[str] = []
        extraction_stats = {'success': 0, 'fallback_extract': 0}
        closure_stats = {
            'ok': 0,
            'oversize': 0,
            'empty': 0,
        }
        scrub_stats = {
            'no_hit': 0,
            'sentence_scrubbed': 0,
            'fallback_too_short': 0,
            'reverse_phrasing_warn': 0,
            'decor_stripped': 0,
            'shell_anchor_fallback': 0,
            'quote_dump_warn': 0,
            'polarity_warn': 0,
            'evaluator_tone_warn': 0,
        }

        for ra, orig_a, ctx, question, critique in zip(
            refined_answers, answers, contexts, questions, critiques
        ):
            extracted = self._extract_refined_answer(ra, orig_a)

            if extracted is None:
                self.logger.warning(
                    f"Failed to extract refined answer after all strategies, keeping original: {orig_a[:50]}..."
                )
                self.logger.debug(f"LLM response was: {ra[:200]}...")
                extraction_stats['fallback_extract'] += 1
                extracted = orig_a
            else:
                extraction_stats['success'] += 1

            final_ans, reason = self._enforce_fact_closure(
                extracted, orig_a or "", critique or "", ctx or ""
            )
            closure_stats[reason] = closure_stats.get(reason, 0) + 1

            scrubbed, hits = _scrub_meta_labels(final_ans or "")
            if hits:
                anchor_len = len(orig_a or "")
                allow_subtractive_scrub = self._critique_allows_subtractive_repair(
                    critique or ""
                )
                if (
                    anchor_len > 0
                    and len(scrubbed) < anchor_len * 0.5
                    and not allow_subtractive_scrub
                ):
                    self.logger.warning(
                        "[AnswerRefiner] _scrub_meta_labels removed too much (len=%d, "
                        "anchor=%d, hits=%s); fallback to clean_answer.",
                        len(scrubbed),
                        anchor_len,
                        hits,
                    )
                    scrub_stats['fallback_too_short'] += 1
                    final_ans = orig_a or ""
                else:
                    self.logger.info(
                        "[AnswerRefiner] _scrub_meta_labels hit phrases=%s, scrubbed length=%d",
                        hits,
                        len(scrubbed),
                    )
                    scrub_stats['sentence_scrubbed'] += 1
                    final_ans = scrubbed
            else:
                scrub_stats['no_hit'] += 1

            if has_reverse_phrasing(final_ans):
                self.logger.warning(
                    "[AnswerRefiner] reverse-phrasing pattern detected (e.g. '应检查并重新确认...是否正确'); "
                    "leaving as-is, but flagging for human review."
                )
                scrub_stats['reverse_phrasing_warn'] += 1

            deco, deco_hits = strip_training_decorators(final_ans or "")
            if deco_hits:
                self.logger.info(
                    "[AnswerRefiner] strip_training_decorators: hits=%s", deco_hits[:5]
                )
                scrub_stats['decor_stripped'] += 1
                final_ans = deco

            if looks_like_fabricated_shell_block(final_ans or "", orig_a or ""):
                self.logger.warning(
                    "[AnswerRefiner] suspected fabricated multi-line shell vs anchor; "
                    "fallback to clean_answer."
                )
                final_ans = (orig_a or "").strip()
                scrub_stats['shell_anchor_fallback'] += 1

            if looks_like_answer_as_quote(final_ans or "", ctx or ""):
                self.logger.warning(
                    "[AnswerRefiner] suspected answer-as-quote dump after refinement; "
                    "question=%r answer_prefix=%r",
                    (question or "")[:80],
                    (final_ans or "")[:120],
                )
                scrub_stats['quote_dump_warn'] += 1

            tone_hits = detect_evaluator_tone(final_ans or "")
            if tone_hits:
                scrub_stats['evaluator_tone_warn'] += 1
                toned, tone_strip_hits = strip_evaluator_tone(final_ans or "")
                if tone_strip_hits and toned != final_ans:
                    self.logger.info(
                        "[AnswerRefiner] stripped evaluator-tone hits=%s",
                        tone_strip_hits[:5],
                    )
                    final_ans = toned
                else:
                    self.logger.warning(
                        "[AnswerRefiner] evaluator-tone leak detected: hits=%s answer_prefix=%r",
                        tone_hits[:5],
                        (final_ans or "")[:120],
                    )

            polarity_issue = detect_question_answer_polarity_issue(question or "", final_ans or "")
            if polarity_issue:
                self.logger.warning(
                    "[AnswerRefiner] question-answer polarity issue: %s question=%r answer_prefix=%r",
                    polarity_issue,
                    (question or "")[:80],
                    (final_ans or "")[:120],
                )
                scrub_stats['polarity_warn'] += 1

            extracted_answers.append(final_ans)

        total = len(refined_answers)
        if extraction_stats['fallback_extract'] > 0:
            self.logger.info(
                f"[AnswerRefiner] extraction: success={extraction_stats['success']}/{total}, "
                f"fallback_extract={extraction_stats['fallback_extract']}/{total}"
            )
        self.logger.info(
            "[AnswerRefiner] fact-closure: ok=%d/%d oversize=%d empty=%d",
            closure_stats['ok'],
            total,
            closure_stats['oversize'],
            closure_stats['empty'],
        )
        self.logger.info(
            "[AnswerRefiner] meta-label scrub summary (total=%d): "
            "rows_no_sentence_meta=%d sentence_scrubbed=%d "
            "fallback_too_short=%d reverse_phrasing_warn=%d decor=%d shell_fb=%d "
            "quote_warn=%d polarity_warn=%d evaluator_tone_warn=%d",
            total,
            scrub_stats['no_hit'],
            scrub_stats['sentence_scrubbed'],
            scrub_stats['fallback_too_short'],
            scrub_stats['reverse_phrasing_warn'],
            scrub_stats['decor_stripped'],
            scrub_stats['shell_anchor_fallback'],
            scrub_stats['quote_dump_warn'],
            scrub_stats['polarity_warn'],
            scrub_stats['evaluator_tone_warn'],
        )

        return extracted_answers

    def run(self, 
            storage: DataFlowStorage, 
            input_context_key: str = 'raw_content',
            input_question_key: str = 'question',
            input_answer_key: str = 'answer',
            output_answer_key: str = 'refined_answer',
            output_critique_key: str = 'answer_critique',
            reuse_existing_critique: bool = False):
        """
        运行答案优化流程。
        
        Args:
            storage: DataFlow存储对象
            input_context_key: 输入上下文字段名
            input_question_key: 输入问题字段名
            input_answer_key: 输入答案字段名
            output_answer_key: 输出优化后答案字段名
            output_critique_key: 输出批评字段名
        """
        df = storage.read('dataframe')
        
        contexts = df.get(input_context_key).to_list()
        questions = df.get(input_question_key).to_list()
        answers = df.get(input_answer_key).to_list()
        
        critique_responses = None
        if reuse_existing_critique and output_critique_key in df.columns:
            existing = df.get(output_critique_key).to_list()
            if all(isinstance(item, str) and item.strip() for item in existing):
                critique_responses = existing
                self.logger.info(
                    f"Reusing existing critiques for {len(answers)} answers..."
                )

        if critique_responses is None:
            self.logger.info(f'Generating critiques for {len(answers)} answers...')
            critique_responses = self.generate_critique(contexts, questions, answers)
            self.logger.info(f'Generated critiques for answers.')
        
        self.logger.info(f'Refining answers based on critiques...')
        refined_answers = self.generate_refined_answer(contexts, questions, answers, critique_responses)
        self.logger.info(f'Refined answers generated.')
        
        df[output_critique_key] = critique_responses
        df[output_answer_key] = refined_answers
        
        storage.write(df)
        self.logger.info(f'Refined answers saved to storage.')
        
        return [output_answer_key, output_critique_key]

