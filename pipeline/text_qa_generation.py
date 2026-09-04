"""
Text QA Generation Pipeline

使用 TextQuestionGenerator 和 TextAnswerGenerator 算子生成 QA 数据集
流程：
1. 从输入文本生成多个问题
2. 基于问题和原始文本生成答案
3. (可选) 使用 AlpagasusFilter 进行质量过滤
"""

from dataflow.operators.text_sft import TextQuestionGenerator, TextAnswerGenerator
from dataflow.operators.text_sft import AlpagasusFilter
from dataflow.operators.text_pt import MetaFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.operators.text_sft import CondorRefiner, QuestionRefiner
import os
os.environ.setdefault("DF_API_KEY", "")


my_dimensions = [
    # ========== 你关心的核心维度 ==========
    {
        "dimension_name": "Context Faithfulness",
        "description": "评估答案是否忠实于给定的参考内容，答案必须基于上下文内容生成，不能编造或引用外部信息",
        "example_list": [
            {
                "text": """ 1. 预警告警（次要/重要级别）
                            触发阈值：当磁盘使用率达到**设置阈值的80%**时触发，例如：
                            若系统设置的只读阈值为 85%（数据库变为只读的阈值），则预警告警会在 68%（85% × 80%）时触发（来源：ALM-5025126、ALM_AI_StorageThresholdPreAlarm）。
                            若组件磁盘容量超过 70%（默认阈值），触发重要告警；超过 80% 时触发紧急告警（来源：ALM-5023113）。
                            2. 紧急告警
                            触发阈值：当磁盘使用率达到或超过 85% 时，触发以下紧急告警：
                            实例磁盘满告警（ALM-5012960）：数据库进入只读模式，禁止写入操作（来源：ALM-5012960）。
                            数据库只读告警（ALM-5025110、ALM_AI_TransactionReadOnly）：磁盘使用率直接超过 85% 触发（来源：ALM-5025110）。
                            3. 其他场景告警
                            默认阈值告警：部分告警默认阈值为 80%（如 ALM-5023112），连续三次检测到超过此阈值时触发（来源：ALM-5023112）。
                            系统盘告警：关注根目录（/）或 /var/log 目录使用率，若达到 100% 将导致 Agent 无法响应（来源：ALM-5014594）。""",
                "score": "5"
            },
            {
                "text": "当系统磁盘使用率大于或等于1%时，将触发名为'Ops巡检-系统磁盘磁盘使用率'的告警，该告警的ID为501 459 4，推荐告警等级为“重要”。监控系统以每分钟一次的频率采集“系统磁盘磁盘使用率”指标，在15分钟的查询区间内，只要检测到当前值满足表达式“>=”且阈值为1%的条件，即立即触发告警（连续1次满足条件即可）。此告警用于快速识别系统磁盘使用异常增长的情况，建议运维人员在告警触发后立即检查主机层面的磁盘占用情况，排查是否存在日志暴增、临时文件堆积或异常进程写入等问题，并结合具体磁盘路径进行清理或扩容操作。",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Meta-information Avoidance",
        "description": "评估答案是否避免了元信息引用。答案不应该使用'根据提供的参考内容'、'文章中提到'、'作者认为'、'第X章节'等元信息短语，应该直接陈述事实",
        "example_list": [
            {
                "text": "Ustore引擎不支持全量物化视图的创建和使用。",
                "score": "5"
            },
            {
                "text": "在账本数据库的blockchain模式下，当用户对防篡改用户表执行INSERT操作时，系统会自动生成一条记录写入对应用户历史表（即blockchain.<schemaname>_<tablename>_hist）中。根据参考内容中的表5-2《hash_ins与hash_del场景对应关系》，INSERT操作对应的字段填充规则如下：\n\n- **hash_ins字段**：会被填充为所插入数据行的hash值。该hash值是对INSERT操作中新插入的数据行内容进行摘要计算得到的hash16类型值，用于确保插入数据的完整性与不可篡改性。\n- **hash_del字段**：在INSERT操作中不涉及数据删除，因此该字段为空（空值）。\n\n此外，用户历史表中的其他字段也会相应更新：\n- **rec_num字段**：记录该操作在历史表中的执行序号，按时间顺序递增。\n- **pre_hash字段**：保存当前用户历史表在本次操作前的整体摘要信息，用于构建链式数据结构，实现操作历史的可追溯性和防篡改验证。\n\n综上所述，在执行INSERT操作时，用户历史表通过记录新插入行的hash值（hash_ins）、空的hash_del以及前序状态摘要pre_hash，完整地保留了操作行为和数据状态，符合账本数据库基于区块链思想的设计原则。",
                "score": "1"
            },
            {
                "text": "执行 `ALTER INDEX i2 SET (enable_tde = on)` 操作的作用是将一个已存在的非加密索引转换为加密索引。根据参考内容，透明数据加密（TDE）支持对索引进行加密状态的动态转换，该操作正是实现这一功能的具体语法。\n\n### 作用说明：\n- **启用索引加密**：此命令显式地将索引 `i2` 设置为加密索引，即开启其透明数据加密功能。在执行该命令前，若创建索引时未指定 `WITH (enable_tde = on)`，则该索引默认为非加密状态；通过 `ALTER INDEX ... SET (enable_tde = on)` 可将其升级为加密索引。\n- **与加密表协同工作**：该操作要求索引所基于的基表必须是加密表（即使用 `WITH (enable_tde = on)` 创建的表）。只有当基表具备加密属性时，才允许将关联索引设置为加密状态，以确保整个数据路径的一致性与安全性。\n\n### 对索引安全性的影响：\n- **提升数据静态保护能力**：启用 `enable_tde = on` 后，数据库在将索引数据写入磁盘之前会自动进行加密，在从磁盘读取后自动解密。这保证了即使存储介质被非法访问或窃取，攻击者也无法直接获取明文索引内容，从而增强了数据的静态安全防护。\n- **统一密钥管理机制**：加密索引与其基表共享相同的加密算法和主密钥（由密钥管理服务提供），并随基表的密钥轮转而同步更新。这意味着索引的加密密钥生命周期由系统统一管理，避免了密钥分散带来的安全隐患。\n- **透明且无感知的安全增强**：应用程序无需修改任何SQL语句或业务逻辑即可享受加密带来的安全增益。所有插入、查询、更新等操作仍正常执行，加密\/解密过程由数据库内核自动完成。\n\n### 对索引存储的影响：\n- **存储元数据增加**：加密索引的相关配置信息（如 `enable_tde`、`encrypt_algo`、`dek_cipher` 等）会被记录在系统表 `pg_class` 的 `reloptions` 字段中。例如，可通过如下查询查看：\n  ```sql\n  SELECT relname, reloptions FROM pg_class WHERE relname = 'i2';\n  ```\n  返回结果中将包含 `{enable_tde=on, encrypt_algo=AES_128_CTR, dek_cipher=...}` 等字段，表明该索引已启用加密，并指定了具体的加密算法和密文信息。\n- **加密算法默认为AES_128_CTR**：除非特别指定，系统默认使用 `AES_128_CTR` 算法对索引数据进行加密。该算法兼顾性能与安全性，适用于大多数场景下的透明加密需求。\n- **不影响存储结构和访问方式**：虽然数据在物理层面以密文形式存储，但逻辑结构（如B树结构）保持不变，索引的构建、维护和查询优化行为不受影响。存储类型（如USTORE）、压缩选项等其他属性也继续生效。\n\n### 总结：\n`ALTER INDEX i2 SET (enable_tde = on)` 是一种灵活的安全增强手段，允许用户在不重建索引的前提下，动态将其转换为加密状态。它显著提升了索引在持久化存储中的保密性，同时保持了对应用的完全透明性和操作兼容性。该操作依赖于已配置的透明加密环境（包括管理员开启 `enable_tde` 和设置 `tde_key_info`），并且要求基表本身为加密表。一旦启用，索引数据将以加密形式落盘，受到与加密表同等强度的保护，是实现端到端数据静态加密的重要组成部分。",
                "score": "2"
            }
        ]
    },
    
    # ========== 补充的通用维度（可根据需要调整）==========
    {
        "dimension_name": "Completeness",
        "description": "评估答案是否完整回答了问题，覆盖了所有关键信息点",
        "example_list": [
            {
                "text": "Ustore引擎不支持全量物化视图，但支持增量物化视图。增量物化视图可以通过CREATE INCREMENTAL MATERIALIZED VIEW语句创建。",
                "score": "5"
            },
            {
                "text": "不支持。",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Accuracy",
        "description": "评估答案的技术准确性，信息是否正确无误",
        "example_list": [
            {
                "text": "GaussDB支持行存储和列存储两种存储模型，适用于不同的应用场景。",
                "score": "5"
            },
            {
                "text": "所有数据库都只支持行存储模式。",
                "score": "1"
            }
        ]
    },
    {
        "dimension_name": "Clarity",
        "description": "评估答案表达是否清晰、逻辑是否通顺、解决步骤是否详细并且分点作答",
        "example_list": [
            {
                "text": """要判断一个ETCD节点是否因数据文件损坏而无法加入集群，需按照以下步骤进行排查和确认：
                            1、检查集群状态
                            执行命令 cm_ctl query -Cvid，查看ETCD节点状态：
                            若存在节点状态为 "Down" 且其他节点正常（如存在主节点 "StateLeader"），可能涉及数据损坏。
                            若所有节点均未处于"Down"状态，数据库可能已自动修复，无需处理。
                            
                            2、验证数据文件完整性
                            登录异常节点，进入ETCD数据目录（路径需添加 /var/chroot）：
                            cd /var/chroot/usr/local/etcd/member
                            检查关键数据文件是否存在：
                            执行 ls snap 确认是否存在 .snap 文件（快照文件）。
                            执行 ls wal 确认是否存在 .wal 文件（预写式日志）。
                            若文件缺失，则判定为数据文件损坏。
                            
                            3、权限检查
                            若访问目录时提示 Permission denied，需修改权限：
                            chmod 700 /var/chroot/usr/local/etcd""",
                "score": "5"
            },
            {
                "text": "要判断一个ETCD节点是否因数据文件损坏而无法加入集群，需按照以下步骤进行排查和确认：首先，确保已执行环境变量导入操作，命令为 `source /home/Ruby/gauss_env/file`，以保障后续指令正常运行。接着，通过执行 `cm_ctl view | grep etcd` 命令查看当前ETCD服务所在的IP地址与端口号，明确目标节点的通信信息。然后，使用 `curl -kv <ETCD_IP>:<ETCD_PORT>` 检查该节点与其他ETCD节点之间的网络连通性，若网络不通，应联系网络管理员处理；若网络连通正常，则继续深入检查节点状态。进入关键诊断阶段，执行集群状态查看命令（具体命令在流程中未完全列出，但通常为 `etcdctl endpoint status` 或 `cm_ctl` 相关子命令），观察输出结果中该节点是否处于非健康状态或未能成功注册到集群中。如果发现该节点长期处于“未就绪”、“异常退出”或“无法同步数据”等状态，并且排除了网络问题和磁盘使用率过高因素后，结合日志表现如出现“corrupted wal file”、“failed to restore snapshot”或“mismatched revision”等典型错误信息，则可判定为ETCD数据文件损坏导致其无法加入集群。此时，该节点属于少数派节点故障情形，需进一步采取修复措施或替换操作以恢复服务。",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Professional Tone",
        "description": "评估答案是否使用专业、规范的语言，适合作为技术文档或培训材料",
        "example_list": [
            {
                "text": "增量物化视图通过物化视图日志追踪基表的变更，仅刷新自上次刷新以来发生变化的数据。",
                "score": "5"
            },
            {
                "text": "这玩意儿就是记录改了啥，然后只更新改的部分，挺省事的。",
                "score": "2"
            }
        ]
    }
]


class TextQAPipeline():
    """
    两阶段 QA 数据生成 Pipeline
    
    第一阶段: TextQuestionGenerator - 从文本生成多个问题
    第二阶段: TextAnswerGenerator - 为每个问题生成答案
    第三阶段: AlpagasusFilter - 质量过滤（可选）
    """
    
    def __init__(
        self, 
        input_file: str,
        num_questions: int = 3,
        question_custom_prompt: str = "",
        answer_custom_prompt: str = "",
        enable_filter: bool = False,
        min_score: int = 3,
        max_score: int = 5
    ):
        """
        初始化 Pipeline
        
        Args:
            input_file: 输入文件路径，JSON 格式，包含 'text' 字段
            num_questions: 每段文本生成的问题数量
            question_custom_prompt: 问题生成的自定义提示词
            answer_custom_prompt: 答案生成的自定义提示词
            enable_filter: 是否启用质量过滤
            min_score: 过滤最低分数
            max_score: 过滤最高分数
        """
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path="./cache_local_qa",
            file_name_prefix="qa_pipeline_step",
            cache_type="json",
        )
        
        # 初始化 LLM 服务
        llm_serving = APILLMServing_request(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            model_name="qwen-plus",
            max_workers=100
        )
        
        # 第一阶段：问题生成器
        self.question_generator = TextQuestionGenerator(
            llm_serving=llm_serving,
            number=num_questions,
            custom_prompt=question_custom_prompt
        )
        self.question_refiner = QuestionRefiner(
            llm_serving=llm_serving
        )
        # 第二阶段：答案生成器
        self.answer_generator = TextAnswerGenerator(
            llm_serving=llm_serving,
            custom_prompt=answer_custom_prompt
        )
        self.refiner = CondorRefiner(llm_serving=llm_serving)
        # 第三阶段：质量过滤（可选）
        # self.enable_filter = enable_filter
        # if enable_filter:
        #     self.alpagasus_filter = AlpagasusFilter(
        #         min_score=min_score,
        #         max_score=max_score,
        #         llm_serving=llm_serving
        #     )
        self.meta_filter = MetaFilter(
            min_score=min_score,           # 均值最低分
            max_score=max_score,           # 均值最高分
            llm_serving=llm_serving,
            dimensions=my_dimensions  # 可选，不传则使用默认6维度
        )
    
    def forward(self):
        """执行 Pipeline"""
        
        # 步骤 1: 从文本生成多个问题
        print("=" * 50)
        print("Step 1: Generating questions from text...")
        print("=" * 50)
        self.question_generator.run(
            storage=self.storage.step(),
            input_key='text',
            output_question_key='question',
            output_content_key='raw_content',
        )
        self.question_refiner.run(
            storage=self.storage.step()
        )
        # 步骤 2: 为每个问题生成答案
        # print("=" * 50)
        # print("Step 2: Generating answers for questions...")
        # print("=" * 50)
        # self.answer_generator.run(
        #     storage=self.storage.step(),
        #     input_question_key='question',
        #     input_content_key='raw_content',
        #     output_instruction_key='instruction',
        #     output_answer_key='output',
        #     output_content_key='raw_content'
        # )
        # self.meta_filter.run(
        #     storage=self.storage.step(),
        #     input_instruction_key='instruction',  # 问题字段
        #     input_input_key='raw_content',         # 上下文字段
        #     input_output_key='output',             # 答案字段
        #     output_key='MetaScore'                 # 输出均值分数字段
        # )
        # # self.refiner.run(
        #     storage=self.storage.step(),
        #     input_instruction_key='instruction',
        #     input_output_key='output'
        # )
        # # 步骤 3: 质量过滤（可选）
        # if self.enable_filter:
        #     print("=" * 50)
        #     print("Step 3: Filtering low quality QA pairs...")
        #     print("=" * 50)
        #     self.alpagasus_filter.run(
        #         storage=self.storage.step(),
        #         input_instruction_key='instruction',
        #         input_input_key="input",
        #         input_output_key='output'
        #     )
        
        print("=" * 50)
        print("Pipeline completed!")
        print("=" * 50)


if __name__ == "__main__":
    # 配置参数
    pipeline = TextQAPipeline(
        input_file="/home/wugk/finetune/finetune/data/主备版特性指南/output.json",
        num_questions=3,  # 每段文本生成 5 个问题
        question_custom_prompt="你是一名运维领域的专家，生成的问题必须是运维领域中有价值的技术问题",
        answer_custom_prompt="答案必须严格按照上下文生成, 操作步骤要详细不能进行总结概要",
        enable_filter=True,#是否启用质量过滤
        min_score=4.5,
        max_score=5
    )
    
    # 执行 Pipeline
    pipeline.forward()

