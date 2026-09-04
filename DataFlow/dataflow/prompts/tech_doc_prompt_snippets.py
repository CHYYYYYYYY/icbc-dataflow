"""
Shared Chinese prompt fragments for TechDoc QA pipeline — single source of meta-phrase rules.

Used by general_text (critique/refine) and kept aligned with meta_label_filters scrub lists.
"""

# Full regex-level list for AnswerRefinePrompt §6 (精炼阶段必须执行删改).
META_FORBIDDEN_ZH_REFINE_BLOCK = """\
必须删除并改写为直接陈述的元信息短语（regex 级别覆盖以下所有变体）：
- 引用参考内容的：`根据参考内容` / `根据参考` / `参考内容明确指出` / `参考内容要求` / `参考内容显示` / `参考内容(?:未|没有)` / `参考上下文(?:未|没有|仅)?` / `参考文档` / `参考资料` / `参考信息` / `参考文件` / `给定材料` / `给定上下文` / `给定文档`
- 训练答案中禁止「材料边界」托词：`当前材料` / `当前材料仅` / `当前材料未` / `当前材料不足以`
- 引用原文的：`原文指出` / `原文明确指出` / `原文要求` / `原文提到` / `原文中` / `根据原文` / `按照原文`
- 引用文档的：`根据文档` / `文档中` / `根据提供的参考` / `根据给定的参考内容` / `根据提供的信息` / `根据给定的文档`
- 引用结构位点的：`在步骤X` / `根据步骤` / `按照步骤` / `根据流程图` / `根据表格` / `在第X章` / `在第X节` / `第X行` / `上述` / `以下` / `如上文所述`
正确做法：把"根据参考文档指出 A 是 B"或"原文指出 A 是 B"改写为直接陈述"A 是 B"；不要保留任何诸如"参考"、"文档"、"原文"、"本文"、"资料"等字眼。"""

# Abbreviated critique instruction: full patterns live in META_FORBIDDEN_ZH_REFINE_BLOCK / 精炼阶段。
META_FORBIDDEN_ZH_CRITIQUE_BRIEF = """\
答案中不得用「引用文档/章节/步骤/行号/图表/资料/信息」的方式交代出处（含：步骤编号、章节与小节、代码行号、流程图/表格指称，以及「根据参考内容」「原文指出」「参考文档」「参考资料」「根据告警参数」「如上所述」等套话）。应直接陈述事实。若出现任一类套话或结构位点引用，判「不通过」；删改口径与精炼阶段的禁词清单一致。"""
