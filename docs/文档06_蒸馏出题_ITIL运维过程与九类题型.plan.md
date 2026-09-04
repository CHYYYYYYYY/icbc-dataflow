---
name: DistillQuestionGen ITIL-QTypes
overview: 用《运维问答分类》中的 ITIL/运维过程维度 + 九类题型 + 标签思想，替换 Round3 计划里 DistillQuestionGeneratorPrompt 原有的「三档难度 + 旧题型列表」；与现有四条实用性硬约束对齐，并处理「事实型 vs 反文档视角」的张力与纯 JSON 字符串输出的限制。
todos: []
isProject: false
---

# DistillQuestionGeneratorPrompt：新版题型与出题策略（对齐《运维问答分类》）

## 目标

在 [finetune/DataFlow/dataflow/prompts/general_text.py](/home/wugk/finetune/DataFlow/dataflow/prompts/general_text.py) 的 `DistillQuestionGeneratorPrompt.build_prompt` 中，**删除**当前 `## Constraints:` 里第 **4** 条（三档：知识记忆 / 分析思考 / 实际应用）与第 **5** 条（原理分析 / 对比 / 应用实践等旧列表），以及 `## Output Format:` 里「三档每档至少 30%」的表述。

**替换为**下文的双轴分类（运维过程 × 题型）、占比与模板策略，使出题与《运维问答分类》一致，同时保留你 Round3 里已定的四条「实用性硬要求」与负面样本清单（可微调措辞以与新题型一致）。

## 双轴分类（写进 prompt 的正文结构）

### 轴 A：运维过程 / 场景（ITIL + DevOps/SRE）

从文档表格抽象为 **10 类场景标签**（生成时每题在内心先选定一类，用于保证多样；不要求改输出格式则不在 JSON 里暴露）：

- Incident（事件）、Problem（问题）、Change（变更）、Configuration（配置/CMDB）、Release（发布）、Capacity（容量）、Availability（可用性/HA）、Continuity（连续性/灾备）、Security（安全/合规）、DevOpsSRE（自动化/可观测/SLO 等）

**多样性约束（建议写死为可执行规则）**：在一次性生成的 `{self.count}` 条问题中，至少覆盖 **4 个不同**场景轴；其中 **Incident / Problem / Change / Availability** 四类合计占比 **≥ 50%**（运维值班最高频）。

### 轴 B：九类题型（替换原「第 5 条」列表）

与文档「数据集问题类别」对齐，在 prompt 中逐类给出 **运维向合格问法**（避免与「禁止什么是 / 本节介绍」冲突）：

| 题型（英文 id 供内部约束） | 中文名 | 合格问法要点（写入 prompt） |
|---------------------------|--------|------------------------------|
| Factoid | 运维事实型 | 禁止裸「什么是 X」；必须带 **现象/决策点/版本约束**，问「X 在本场景下影响什么 / 与哪类故障相关」 |
| Diagnostic | 诊断型 | 现象 + 日志/指标线索 + 问可能原因或 **先查什么** |
| Procedural | 操作步骤型 | 「如何在 {环境约束} 下完成 {变更/操作}」+ 可验收目标 |
| RootCause | 根因分析型 | 反复/类故障模式 + 问 **机制层根因** 与验证思路 |
| BestPractice | 最佳实践型 | 必须带 **约束**（窗口、RPO/RTO、预算、版本）；禁止无边界的「有哪些最佳实践」 |
| ConfigExample | 配置示例型 | 问「依据文档应改哪些参数/文件项、前后检查点」，**题干不写死目标参数值** |
| ScriptCommand | 脚本/命令型 | 问「给出可复用脚本/命令骨架」；敏感信息用占位符 |
| FaultReproFix | 故障复现与修复 | **强制**「测试/预发/演练环境」措辞；禁止鼓励生产破坏性操作 |
| ComplianceSecurity | 合规/安全咨询 | 权限、加密、审计、最小权限；与上下文中的安全段落绑定 |

**与现有硬约束 4 的对齐**：Factoid、BestPractice 用上面「运维向模板」收口，避免退回文档背诵；FaultReproFix 单独加一句「不得省略环境限定词」。

## 占比与轮换策略（替换原「三档每档 ≥30%」）

按 `{self.count}` 给 **下限/上限**（比「九类各一」更稳，因 count 可能小于 9）：

- **核心排障/变更（Diagnostic + Procedural + RootCause）**：合计 **≥ ceil(0.5 × count)**，且每一类 **≥ 1**（当 `count ≥ 3`）。
- **事实与配置（Factoid + ConfigExample）**：合计 **≤ floor(0.35 × count)**，且 Factoid 单独 **≤ floor(0.2 × count)**，防止定义题回潮。
- **脚本/命令（ScriptCommand）**：若 `context` 中含代码块、命令行、参数表、RMAN/SQL 片段 → **≥ 1**；否则 **≥ 0**（不写死造假场景）。
- **最佳实践（BestPractice）**：**≤ ceil(0.25 × count)**，且每条必须含显式约束从句。
- **故障复现与修复（FaultReproFix）**：**≤ max(1, floor(0.15 × count))**，且一律测试/演练语境。
- **合规/安全（ComplianceSecurity）**：若上下文出现加密、权限、审计、合规关键词 → **≥ 1**；否则 0。

（实现时把 `ceil/floor` 写成自然语言区间即可，例如 count=10：诊断+步骤+根因合计≥5，事实+配置≤3，等。）

## 标签体系（轻量接入）

文档中的 `#定义 #故障诊断` 等标签：**不强制改 JSON 输出结构**时，在 prompt 中要求模型在 **起草阶段** 为每题选定 **1 个主标签 + 1 个场景轴** 用于自检，但最终仍只输出字符串数组（与 [L2478-L2482](/home/wugk/finetune/DataFlow/dataflow/prompts/general_text.py) 现状兼容）。

**可选增强（另起 todo，避免悄悄改数据契约）**：若你希望数据集统计「题型/场景占比」，再把输出改为 `[{"question","qtype","scene","tags"}]`，并同步改 [distill_question_generator.py](/home/wugk/finetune/DataFlow/dataflow/operators/text_sft/generate/distill_question_generator.py) 的解析与下游字段。

## 负面样本与文档示例的取舍

保留原 **6. 负面样本清单** 框架，并 **显式加一条**：禁止把《运维问答分类》里的「什么是闪回恢复区」类 **纯定义模板**原样生成；若题型为 Factoid，只能使用本节「运维事实型」改写模板。

## Workflow 小节同步改两句

当前 [L2426-L2431](/home/wugk/finetune/DataFlow/dataflow/prompts/general_text.py) 写「规划难度分布基础/中级/高级」——改为「先为每条题选定场景轴与题型，再检查占比与硬约束，最后输出」。

## 与 Round3 其余部分的关系

- **不新增前置 gate 算子**、仍以首轮 prompt 收口：不变。
- **AnswerCritique / Rubric** 可按需在后续轮次把 `qtype` 与「可执行性」阈值做软关联；本方案先不动，除非上结构化输出。

## 实施时的单文件落点

只改 `DistillQuestionGeneratorPrompt.build_prompt` 内 `base_prompt` 字符串：**Constraints 第 4–5 条替换**、**Workflow 两处替换**、**Output Format 最后一 bullet 替换**；类 docstring 里「基础/中级/高级」描述建议一并改成「场景×题型覆盖」。
