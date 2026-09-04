"""QA 生成管道脚本。依赖 DataFlow 包与 DF_API_KEY。"""
import os
import re
import sys
from pathlib import Path

# 未设置 PYTHONPATH 时，从本仓库推断 DataFlow 根目录（…/finetune/DataFlow）
_DATAFLOW_ROOT = Path(__file__).resolve().parents[2] / "DataFlow"
if _DATAFLOW_ROOT.is_dir() and str(_DATAFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATAFLOW_ROOT))

from dataflow.operators.text_sft import (
    TextQuestionGenerator,
    TextAnswerGenerator,
    DistillQuestionGenerator,
    EvolInstructGenerator,
    QuestionRefiner,
    AnswerRefiner,
    AnswerGroundingFilter,
    AnswerCritiqueEvaluator,
    AnswerRewriter,
    RubricScorer,
)
from dataflow.operators.text_pt import MetaFilter
from dataflow.utils.meta_label_filters import is_outofscope_or_meta_tox, extract_llm_answer_text
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

# 使用 APILLMServing_request 前请在环境中设置 DF_API_KEY（勿将密钥写入代码库）
DEFAULT_API_URL = os.environ.get(
    "DF_API_URL",
    "http://ai.wenmodel.com/v1/chat/completions",
)
DEFAULT_MODEL_NAME = os.environ.get("DF_MODEL_NAME", "qwen-plus")
DEFAULT_MAX_WORKERS = int(os.environ.get("DF_MAX_WORKERS", "8"))


question_dimensions = [
    # ========== 你关心的核心维度 ==========
    {
        "dimension_name": "Context Faithfulness",
        "description": "评估问题是否忠实于给定的参考内容，答案必须基于上下文内容生成，不能编造或引用外部信息",
        "example_list": [
            {
                "raw_content": "### 5.5 修复账本数据库  \n\n## 前提条件  \n\n- 系统中需要有审计管理员或者具有审计管理员权限的角色。- 数据库正常运行，并且对防篡改数据库执行了一系列增、删、改等操作，保证在查询时段内有账本操作记录结果产生。  \n\n## 背景信息  \n\n- 当在异常情况或表被损坏时，需要使用ledger_gchain_repair(text, text)接口对全局区块表进行修复，或使用ledger_hist_repair(text, text)接口对用户历史表进行修复，修复后调用全局区块表或用户历史表校验接口结果为true。- 修复用户历史表的接口为pg_catalog.ledger_hist_repair，操作为：SELECT pg_catalog.ledger_hist_repair(schema_name_text,table_name_text);如果修复成功，函数返回修复过程中用户历史表hash的增量。注：对用户表执行闪回DROP时，可使用该函数恢复用户表和用户历史表名称，请参见恢复用户表和用户历史表名称。- 修复全局区块表的接口为pg_catalog.ledger_gchain_repair，操作为：SELECT pg_catalog.ledger_gchain_repair(schema_name_text,table_name_text);如果修复成功，函数返回修复过程中全局区块表中指定表的hash总和。  \n\n## 恢复用户表数据和全局区块表数据  \n\n步骤1 执行历史表修复操作。gaussdb=# SELECT pg_catalog.ledger_hist_repair('ledgernsp','usertable');查询结果如下：ledger_hist_repair84e8bfc3b974e9cf(1 row)该结果表明当前节点用户历史表修复成功，修复造成的用户历史表hash增量为84e8bfc3b974e9cf。  \n\n步骤2 执行全局区块表修复操作。gaussdb=# SELECT pg_catalog.ledger_gchain_repair('ledgernsp','usertable');查询结果如下：ledger_gchain_repaira41714001181a294(1 row)\n\n该结果表明，全局区块表修复成功，且插入一条修复数据，其hash值为a41714001181a294。  \n\n- ---结束  \n\n## 恢复用户表和用户历史表名称  \n\n已通过enable_recyclebin参数和recyclebin_retention_time参数开启闪回DROP功能，恢复用户表和用户历史表名称。示例如下：  \n\nDROP用户表，对用户表执行闪回DROP。使用ledger_hist_repair对用户表、用户历史表进行表名恢复。-- 对用户表执行闪回drop，使用ledger_hist_repair对用户历史表进行表名恢复。gaussdb=# CREATE TABLE ledgernsp.tab2(a int, b text);CREATE TABLEgaussdb=# DROP TABLE ledgernsp.tab2;DROP TABLEgaussdb=# SELECT rcyreld, rcyname, rcyoriginname rcyreld | rcyname | rcyoriginname---16717 | BIN\\$38242338414D\\$42EB978==\\$0 | tab2 16725 | BIN\\$382423384155\\$42EC678==\\$0 | gs_hist_tab2_index 16722 | BIN\\$382423384152\\$42ECC30==\\$0 | ledgernsp_tab2_hist 16720 | BIN\\$382423384150\\$42ED3E0==\\$0 | pg_toast_16717(4 rows)---对用户表执行闪回drop。gaussdb=# TIMECAPSULE TABLE ledgernsp.tab2 TO BEFORE DROP;TimeCapsule Table---使用ledger_hist_repair恢复用户历史表表名。gaussdb=# SELECT ledger_hist_repair('ledgernsp', 'tab2');ledger_hist_repair000000000000000(1 row)gaussdb=# TIMECAPSULE TABLE ledgernsp.tab2 TO BEFORE DROP;TimeCapsule Tablegaussdb=# SELECT ledger_hist_repair('ledgernsp', 'tab2');ledger_hist_repair000000000000000(1 row)gaussdb=# \\d+ ledgernsp.tab2;Table \"ledgernsp.tab2\"Column | Type | Modifiers | Storage | Stats target | Descriptiona | integer | | plain | | b | text | | extended | | hash_1d2d14 | hash16 | | plain | | Has OIDs: noOptions: orientation=row, compression=no, storage_type=USTORE, segment=off, toast.storage_type=USTORE, toast.toast_storage_type=enhanced_toastHistory table name: ledgernsp_tab2_hist---对用户表执行闪回drop，使用ledger_hist_repair对用户表进行表名恢复。gaussdb=# CREATE TABLE ledgernsp.tab3(a int, b text);CREATE TABLEgaussdb=# DROP TABLE ledgernsp.tab3;DROP TABLEgaussdb=# SELECT rcyreld, rcyname, rcyoriginname FROM gs_recyclebin;rcyreld | rcyname | rcyoriginname---17574 | BIN\\$44A4233844A6\\$B18E7A0==\\$0 | tab3 17582 | BIN\\$44A4233844AE\\$B18F488==\\$0 | gs_hist_tab3_index 17579 | BIN\\$44A4233844AB\\$B18F4A0==\\$0 | ledgernsp_tab3_hist 17577 | BIN\\$44A4233844A9\\$B190208==\\$0 | pg_toast_17574(4 rows)---对用户历史表执行闪回drop。\n\ngaussdb=# TIMECAPSULE TABLE blockchain.ledgernsp_tab3_hist TO BEFORE DROP; TimeCapsule Table  \n\n-- 拿到回收站中用户表对应的rcyname，使用ledger_hist_repair恢复用户表表名。  \n\ngaussdb=# SELECT ledger_hist_repair('ledgernsp','BIN$44A4233844A6$B18E7A0==$0'); ledger_hist_repair",
                "text": "执行全局区块表修复后，返回的hash值代表什么？",
                "score": "5",
                "explain": "生成问题可在上下文中找到"
            },
            {
                "raw_content": "### 5.5 修复账本数据库  \n\n## 前提条件  \n\n- 系统中需要有审计管理员或者具有审计管理员权限的角色。- 数据库正常运行，并且对防篡改数据库执行了一系列增、删、改等操作，保证在查询时段内有账本操作记录结果产生。  \n\n## 背景信息  \n\n- 当在异常情况或表被损坏时，需要使用ledger_gchain_repair(text, text)接口对全局区块表进行修复，或使用ledger_hist_repair(text, text)接口对用户历史表进行修复，修复后调用全局区块表或用户历史表校验接口结果为true。- 修复用户历史表的接口为pg_catalog.ledger_hist_repair，操作为：SELECT pg_catalog.ledger_hist_repair(schema_name_text,table_name_text);如果修复成功，函数返回修复过程中用户历史表hash的增量。注：对用户表执行闪回DROP时，可使用该函数恢复用户表和用户历史表名称，请参见恢复用户表和用户历史表名称。- 修复全局区块表的接口为pg_catalog.ledger_gchain_repair，操作为：SELECT pg_catalog.ledger_gchain_repair(schema_name_text,table_name_text);如果修复成功，函数返回修复过程中全局区块表中指定表的hash总和。  \n\n## 恢复用户表数据和全局区块表数据  \n\n步骤1 执行历史表修复操作。gaussdb=# SELECT pg_catalog.ledger_hist_repair('ledgernsp','usertable');查询结果如下：ledger_hist_repair84e8bfc3b974e9cf(1 row)该结果表明当前节点用户历史表修复成功，修复造成的用户历史表hash增量为84e8bfc3b974e9cf。  \n\n步骤2 执行全局区块表修复操作。gaussdb=# SELECT pg_catalog.ledger_gchain_repair('ledgernsp','usertable');查询结果如下：ledger_gchain_repaira41714001181a294(1 row)\n\n该结果表明，全局区块表修复成功，且插入一条修复数据，其hash值为a41714001181a294。  \n\n- ---结束  \n\n## 恢复用户表和用户历史表名称  \n\n已通过enable_recyclebin参数和recyclebin_retention_time参数开启闪回DROP功能，恢复用户表和用户历史表名称。示例如下：  \n\nDROP用户表，对用户表执行闪回DROP。使用ledger_hist_repair对用户表、用户历史表进行表名恢复。-- 对用户表执行闪回drop，使用ledger_hist_repair对用户历史表进行表名恢复。gaussdb=# CREATE TABLE ledgernsp.tab2(a int, b text);CREATE TABLEgaussdb=# DROP TABLE ledgernsp.tab2;DROP TABLEgaussdb=# SELECT rcyreld, rcyname, rcyoriginname rcyreld | rcyname | rcyoriginname---16717 | BIN\\$38242338414D\\$42EB978==\\$0 | tab2 16725 | BIN\\$382423384155\\$42EC678==\\$0 | gs_hist_tab2_index 16722 | BIN\\$382423384152\\$42ECC30==\\$0 | ledgernsp_tab2_hist 16720 | BIN\\$382423384150\\$42ED3E0==\\$0 | pg_toast_16717(4 rows)---对用户表执行闪回drop。gaussdb=# TIMECAPSULE TABLE ledgernsp.tab2 TO BEFORE DROP;TimeCapsule Table---使用ledger_hist_repair恢复用户历史表表名。gaussdb=# SELECT ledger_hist_repair('ledgernsp', 'tab2');ledger_hist_repair000000000000000(1 row)gaussdb=# TIMECAPSULE TABLE ledgernsp.tab2 TO BEFORE DROP;TimeCapsule Tablegaussdb=# SELECT ledger_hist_repair('ledgernsp', 'tab2');ledger_hist_repair000000000000000(1 row)gaussdb=# \\d+ ledgernsp.tab2;Table \"ledgernsp.tab2\"Column | Type | Modifiers | Storage | Stats target | Descriptiona | integer | | plain | | b | text | | extended | | hash_1d2d14 | hash16 | | plain | | Has OIDs: noOptions: orientation=row, compression=no, storage_type=USTORE, segment=off, toast.storage_type=USTORE, toast.toast_storage_type=enhanced_toastHistory table name: ledgernsp_tab2_hist---对用户表执行闪回drop，使用ledger_hist_repair对用户表进行表名恢复。gaussdb=# CREATE TABLE ledgernsp.tab3(a int, b text);CREATE TABLEgaussdb=# DROP TABLE ledgernsp.tab3;DROP TABLEgaussdb=# SELECT rcyreld, rcyname, rcyoriginname FROM gs_recyclebin;rcyreld | rcyname | rcyoriginname---17574 | BIN\\$44A4233844A6\\$B18E7A0==\\$0 | tab3 17582 | BIN\\$44A4233844AE\\$B18F488==\\$0 | gs_hist_tab3_index 17579 | BIN\\$44A4233844AB\\$B18F4A0==\\$0 | ledgernsp_tab3_hist 17577 | BIN\\$44A4233844A9\\$B190208==\\$0 | pg_toast_17574(4 rows)---对用户历史表执行闪回drop。\n\ngaussdb=# TIMECAPSULE TABLE blockchain.ledgernsp_tab3_hist TO BEFORE DROP; TimeCapsule Table  \n\n-- 拿到回收站中用户表对应的rcyname，使用ledger_hist_repair恢复用户表表名。  \n\ngaussdb=# SELECT ledger_hist_repair('ledgernsp','BIN$44A4233844A6$B18E7A0==$0'); ledger_hist_repair",
                "text": "在GaussDB中，如何查看表的详细结构信息？",
                "score": "1",
                "explain": "生成问题无法在上下文中定位"
            },
        ]
    },
    {
        "dimension_name": "Meta-information Avoidance",
        "description": "评估问题是否避免了元信息引用。问题不应该使用'根据提供的参考内容'、'文章中提到'、'作者认为'、'第X章节'等元信息短语，应该直接陈述事实",
        "example_list": [
            {
                "text": "如果DN组件修复失败，在执行步骤6时需要收集哪些关键日志和信息用于技术支持分析",
                "score": "1",
                "explain": "提问时并不清楚步骤6是什么，这是利用元信息生成的"
            },
            {
                "text": "如果DN组件修复失败，需要收集哪些关键日志和信息用于技术支持分析",
                "score": "5",
                "explain": "提问时避免利用元信息生成问题"
            },
            
            {
                "text": "当步骤7中未出现'Timeout exception'时，应如何进一步定位问题",
                "score": "1",
                "explain": "提问时并不清楚步骤7是什么，这是利用元信息生成的"
            },
        ]
    },
    
    # ========== 补充的通用维度（可根据需要调整）==========
    {
        "dimension_name": "Operator Perspective",
        "description": "评估问题是否符合运维人员的提问习惯和视角",
        "example_list": [
            {
                "text": "CN协调器所有实例都显示为Normal代表集群正常吗？",
                "score": "5",
                "explain": "符合运维人员的提问习惯"
            },
            {
                "text": "结合CMServer和ETCD的状态，为什么协调器所有实例都显示为Normal可能是合理的",
                "score": "1",
                "explain": "问题不符合运维人员的提问习惯, 一般人也不会这么问"
            },
            {
                "text": "表格中的'主键'字段是什么意思？",
                "score": "5",
                "explain": "符合运维人员的提问习惯"
            },
            {
                "text": "什么是表格中的'主键'字段所表示的含义？",
                "score": "1",
                "explain": "问题不符合运维人员的提问习惯, 一般人也不会这么问"
            }
        ]
    },
    {
        "dimension_name": "Practical Value",
        "description": "评估问题对实际排障是否有参考价值",
        "example_list": [
            {
                "text": "在处理该告警时，登录ManageOne运维面的访问地址是什么？",
                "score": "1",
                "explain": "常识性问题,实际排障不会问"
            },
            {
                "text": "执行ps -ef | grep dN datanode -z命令的目的是什么",
                "score": "1",
                "explain": "常识性问题,实际排障不会问"
            },
            {
                "text": "内核执行gs_replace命令修复节点会失败可能的原因",
                "score": "1",
                "explain": "该问题对实际排障有实际价值"
            }
        ]
    },
    {
        "dimension_name": "Clarity",
        "description": "评估问题表达是否清晰无歧义，上下文完整",
        "example_list": [
            {
                "text": "主节点和管理节点分别使用哪些IP地址进行通信",
                "score": "1",
                "explain": "该问题有歧义，不建议提问"
            },
            {
                "text": "节点上运行的关键资源有哪些",
                "score": "1",
                "explain": "该问题有歧义，节点可理解为数据库组件，如DN、CN节点等，也可以理解为配置文件或者磁盘与CPU资源"
            }
        ]
    },
    {
        "dimension_name": "Specificity Balance",
        "description": "评估问题的具体程度是否适中，是否过泛或者太具体",
        "example_list": [
            {
                "text": "Python脚本check.py中变量volatility的值是多少",
                "score": "1",
                "explain": "问题问的太具体了，问到脚本中某一个参数了，一般不会这么问。"
            },
            {
                "text": "PostgreSQL集群中显示'pg_replication'复制状态意味着什么机制在运行？",
                "score": "1",
                "explain": "原文没有专门提到pg_replication，这个只是在复制问题的步骤中出现在SQL语句中，虽然回答的没什么问题，但很少会问这么具体某个进程。"
            }
        ]
    }
]

answer_dimensions = [
    # ========== 你关心的核心维度 ==========
    {
        "dimension_name": "Completeness",
        "description": "评估答案是否包含了原文中的所有关键步骤/要点",
        "example_list": [
            {
                "text": "要判断一个ETCD节点是否因数据文件损坏而无法加入集群，需按照以下步骤进行排查和确认：首先，确保已执行环境变量导入操作，命令为 `source /home/Ruby/gauss_env/file`，以保障后续指令正常运行。接着，通过执行 `cm_ctl view | grep etcd` 命令查看当前ETCD服务所在的IP地址与端口号，明确目标节点的通信信息。然后，使用 `curl -kv <ETCD_IP>:<ETCD_PORT>` 检查该节点与其他ETCD节点之间的网络连通性，若网络不通，应联系网络管理员处理；若网络连通正常，则继续深入检查节点状态。进入关键诊断阶段，执行集群状态查看命令（具体命令在流程中未完全列出，但通常为 `etcdctl endpoint status` 或 `cm_ctl` 相关子命令），观察输出结果中该节点是否处于非健康状态或未能成功注册到集群中。如果发现该节点长期处于“未就绪”、“异常退出”或“无法同步数据”等状态，并且排除了网络问题和磁盘使用率过高因素后，结合日志表现如出现“corrupted wal file”、“failed to restore snapshot”或“mismatched revision”等典型错误信息，则可判定为ETCD数据文件损坏导致其无法加入集群。此时，该节点属于少数派节点故障情形，需进一步采取修复措施或替换操作以恢复服务。",
                "score": "2",
                "explain": "答案未分点"
            },
            {
                "text": """要判断一个ETCD节点是否因数据文件损坏而无法加入集群，可以按照以下步骤:
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
                "score": "5",
                "explain": "答案步骤详细"
            },
        ]
    },
    {
        "dimension_name": "Meta-information Avoidance",
        "description": "评估答案是否避免了元信息引用。答案不应该使用'根据提供的参考内容'、'文章中提到'、'作者认为'、'第X章节'等元信息短语，应该直接陈述事实",
        "example_list": [
            {
                "text": "根据提供的上下文，在向使用RANGE策略的一级分区表中新增一级分区时，如果未显式定义二级分区，数据库会自动为该一级分区创建一个默认的二级分区。\n\n具体行为如下：\n- 当执行 `ALTER TABLE ... ADD PARTITION` 语句新增一级分区时，如果该语句中**没有**包含 `SUBPARTITION` 子句来定义二级分区，数据库系统会自动生成一个默认的二级分区。\n- 这个默认的二级分区会涵盖所有未明确分配给其他二级分区的值（即使用 `VALUES (DEFAULT)`）。\n\n示例说明：\n在上下文的第二个示例中，SQL语句：\n```sql\nALTER TABLE range_list_sales ADD PARTITION date_202005 VALUES LESS THAN ('2020-06-01') TABLESPACE tb1;\n```\n这条语句仅新增了一个名为 `date_202005` 的一级分区，但没有指定任何二级分区定义。因此，数据库会自动创建一个默认的二级分区。此行为与以下显式定义默认二级分区的SQL语句完全等价：\n```sql\nALTER TABLE range_list_sales ADD PARTITION date_202005 VALUES LESS THAN ('2020-06-01') TABLESPACE tb1\n(SUBPARTITION date_202005_channel1 VALUES (DEFAULT));\n```\n这意味着，新增的 `date_202005` 一级分区下会包含一个默认的二级分区（例如名为 `date_202005_channel1`），该分区接收所有未在显式定义的二级分区中指定的值。\n\n总结：在未显式定义二级分区的情况下，数据库的默认行为是自动创建一个使用 `DEFAULT` 值的二级分区，以确保分区表的完整性和数据可存储性。",
                "score": "1",
                "explain": "包含元信息引用"
            },
            {
                "text": "在向使用RANGE策略的一级分区表中新增一级分区时，如果未显式定义二级分区，数据库将**自动为该新增的一级分区创建一个默认的二级分区**。\n\n具体来说，当执行 `ALTER TABLE ADD PARTITION` 语句向二级分区表添加一个新的RANGE一级分区，但未在该一级分区内声明具体的二级分区（即SUBPARTITION）定义时，系统会自动生成一个包含默认子分区（DEFAULT subpartition）的二级分区。这个默认二级分区通常用于接收该一级分区下所有未被其他明确规则覆盖的数据。\n\n### 示例说明：\n\n假设存在一个名为 `range_list_sales` 的二级分区表，其一级分区策略为 RANGE（按时间范围划分），二级分区策略为 LIST（按渠道值划分）。现在执行以下SQL语句：\n\n```sql\nALTER TABLE range_list_sales ADD PARTITION date_202005 \nVALUES LESS THAN ('2020-06-01') TABLESPACE tb1;\n```\n\n此语句仅指定了新增一个名为 `date_202005` 的一级分区，用于存储小于 `'2020-06-01'` 的数据，并将其存放在表空间 `tb1` 中，**但未显式定义任何二级分区**。\n\n在这种情况下，数据库会自动执行等效于如下语句的操作：\n\n```sql\nALTER TABLE range_list_sales ADD PARTITION date_202005 \nVALUES LESS THAN ('2020-06-01') TABLESPACE tb1  \n(SUBPARTITION date_202005_channel1 VALUES (DEFAULT));\n```\n\n也就是说，数据库会为这个新加入的一级分区 `date_202005` 创建一个名称可能由系统生成或遵循命名惯例（如示例中的 `date_202005_channel1`）的二级分区，且其值为 `(DEFAULT)`，表示它可以接受该一级分区内所有符合LIST列定义但未匹配其他具体值的记录。\n\n### 总结：\n\n- **处理方式**：若未显式定义二级分区，数据库会自动创建一个带有 `VALUES (DEFAULT)` 的默认二级分区。\n- **目的**：确保即使没有详细子分区定义，新增的一级分区仍具备完整的二级分区结构，避免数据无法写入的问题。\n- **适用场景**：适用于需要快速扩展分区结构、暂时无需精细控制二级分区的场合。\n- **注意事项**：依赖默认分区可能导致后续管理不够灵活，建议在生产环境中根据实际业务需求明确定义二级分区结构。",
                "score": "5",
                "explain": "答案详细且无元信息引用"
            }
        ]
    },
    
    # ========== 补充的通用维度（可根据需要调整）==========
    {
        "dimension_name": "Accuracy",
        "description": "评估答案内容是否正确，无事实性错误",
        "example_list": [
        ]
    },
    {
        "dimension_name": "Relevance",
        "description": "评估答案是否直接回答问题，与问题焦点一致",
        "example_list": [
        ]
    },
    {
        "dimension_name": "Source Fidelity",
        "description": "评估答案是否基于原文，非编造内容",
        "example_list": [
        ]
    },
    {
        "dimension_name": "Structure",
        "description": "评估答案是否分点分段，结构清晰易读",
        "example_list": [
        ]
    }
]


# 技术文档 V2：单次 MetaFilter 用（须 6 个维度，与 MetaSampleEvaluator 解析一致）
# 定位：Rubric 之后的最后质量门禁，评估的是「问答对作为微调训练样本」的综合质量。
# Rubric 已保证答案基本溯源正确；Meta 在此基础上补充检测：
#   - 问答语义是否真正对齐（Rubric 不检查问答话题偏移）
#   - 问题是否精准到能唯一定位答案（Rubric 不检查问题端）
#   - 是否有训练污染风险（元信息残留、托词、题干答案泄漏）
#   - 答案完整性（Rubric SC-4 粗粒度，这里专项检查遗漏关键信息）
#   - 技术准确性（Rubric HC 兜底后仍可能有细节偏差）
#   - 运维训练价值（对微调收益的整体判断，Rubric 无此维度）
tech_doc_qa_meta_dimensions = [
    {
        "dimension_name": "问答语义一致性",
        "description": (
            "答案是否直接、完整地回应了问题所询问的内容，"
            "没有话题漂移（问A答B）、范围错位（问原因答步骤）或视角反转（问应避免什么却答应优先做什么）"
        ),
        "example_list": [
            {
                "text": "问题询问排查连接超时的检查顺序，答案按优先级给出了具体检查步骤和每步的判断标准。",
                "score": "5",
            },
            {
                "text": "问题询问某参数超阈值时的告警处理路径，答案却介绍了该参数的定义和默认值历史。",
                "score": "1",
            },
        ],
    },
    {
        "dimension_name": "问题精准度与唯一性",
        "description": (
            "问题的约束条件（现象描述、版本、操作对象、场景）是否足够精确，"
            "能在给定上下文中唯一定位到该答案；"
            "过于宽泛（多种答案均成立）或过度细化（问到单个字段/行号）均不合格"
        ),
        "example_list": [
            {
                "text": "问题包含具体的错误码、触发条件和操作对象约束，对应答案在上下文中唯一可定位。",
                "score": "5",
            },
            {
                "text": "问题仅说『集群出现异常该怎么办』，无现象约束，任何排障步骤都能成为答案。",
                "score": "1",
            },
        ],
    },
    {
        "dimension_name": "答案完整性",
        "description": (
            "答案是否覆盖了回答该问题所必须的全部关键信息，"
            "没有遗漏关键步骤、前置条件或验证方法；"
            "同时不应包含与问题无关的额外推断或场景扩写"
        ),
        "example_list": [
            {
                "text": "问题要求列出三个 RPO=0 的前置条件，答案完整给出三条并各附验证命令。",
                "score": "5",
            },
            {
                "text": "问题要求说明完整的故障处理路径，答案只给出了第一步日志检查，后续步骤缺失。",
                "score": "1",
            },
        ],
    },
    {
        "dimension_name": "训练污染风险",
        "description": (
            "问答对中是否存在会污染模型微调的元素，包括："
            "（1）元信息引用：如『根据参考内容』『原文指出』『在步骤X』等；"
            "（2）模糊托词：如『无法根据文档回答』『参考内容未提供』；"
            "（3）答案泄漏：问题题干已包含核心答案的参数值/命令/结论；"
            "（4）过度推断：答案末尾追加了原文不支持的因果推断或风险分析段"
        ),
        "example_list": [
            {
                "text": "问题和答案均直接陈述事实，无元信息引用、无托词、题干未泄漏答案值。",
                "score": "5",
            },
            {
                "text": "答案以『根据参考内容，...』开头，且问题题干中已写明『把参数设为70%』这一结论。",
                "score": "1",
            },
        ],
    },
    {
        "dimension_name": "技术准确性",
        "description": (
            "问答中的命令语法、参数名称、数值/单位、操作顺序、术语"
            "是否与原文上下文完全一致，无拼写错误、数量级偏差或版本范围错误"
        ),
        "example_list": [
            {
                "text": "命令字符串、参数值与原文逐字一致，术语使用规范，版本适用范围描述准确。",
                "score": "5",
            },
            {
                "text": "答案中的参数名与原文有大小写差异，或引用了原文不存在的命令选项。",
                "score": "1",
            },
        ],
    },
    {
        "dimension_name": "运维训练价值",
        "description": (
            "从 SRE/DBA 视角评估，这条问答对作为微调训练样本的收益："
            "能否教会模型在真实运维场景（排障/变更/配置/容量/安全）中给出可落地的判断或操作；"
            "纯文档背诵、无决策意义的定义题、或答案对实际排障决策无帮助的题目评分低"
        ),
        "example_list": [
            {
                "text": "问答对直接对应一线告警处理，答案给出了带可验证结果的操作步骤，值班人员可直接照做。",
                "score": "5",
            },
            {
                "text": "问题是『界面上哪种颜色代表告警状态』，答案对运维决策毫无帮助。",
                "score": "1",
            },
        ],
    },
]


TECH_DOC_V2_ANSWER_PROMPT = """
严格要求：
1. 只使用参考内容中明确存在的信息作答
2. 不得添加参考内容中没有的任何具体信息
3. 不得出现"参考内容未提供"/"无法回答"/"文档未提及"等声明
4. 如果问题超出参考内容范围，仅输出 OUTOFSCOPE 四个大写字母（不要其它文字）
"""


def storage_apply_row_mask(storage, row_predicate):
    """读取当前 step 的 DataFrame，按行谓词过滤后写回（需先 storage.step()）。"""
    st = storage
    df = st.read("dataframe")
    mask = df.apply(row_predicate, axis=1)
    st.write(df.loc[mask].reset_index(drop=True))


def storage_strip_llm_answer_columns(storage, columns: list[str]) -> None:
    """去掉 reasoning 模型输出中的 <think>，只保留 <answer> 正文。"""
    df = storage.read("dataframe").copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(extract_llm_answer_text)
    storage.write(df)


# === V2 pipeline 行级过滤谓词（集中管理，避免散落 lambda） ===
def _initial_answer_valid(row) -> bool:
    """OUTOFSCOPE token 或元信息托词答案 → False。复用 is_outofscope_or_meta_tox 共享逻辑。"""
    drop, _ = is_outofscope_or_meta_tox(extract_llm_answer_text(row.get("initial_answer")))
    return not drop


def _refined_answer_valid(row) -> bool:
    """Step 6 精炼后答案仍不得含 OUTOFSCOPE / 元信息托词。force_rewrite 后的廉价兜底。"""
    drop, _ = is_outofscope_or_meta_tox(extract_llm_answer_text(row.get("refined_answer")))
    return not drop


def _grounding_not_drop(row) -> bool:
    return str(row.get("grounding_verdict", "")).strip().upper() != "DROP"


def _has_clean_answer(row) -> bool:
    ca = row.get("clean_answer")
    if ca is None:
        return False
    return str(ca).strip() != ""



def _rubric_pass(row) -> bool:
    v = row.get("rubric_verdict")
    return isinstance(v, str) and v.strip().upper() == "PASS"


# ── 轴B题型 → Rubric question_type 映射表 ──────────────────────────────────────
# DistillQuestionGenerator 的 distill_question_type 使用轴B分类（13种），
# RubricScorer 的 HARD_CONSTRAINTS_BY_TYPE 使用另一套 5+1 分类。
# 此映射在 Rubric 之前执行，将 distill_question_type 翻译到 Rubric 能识别的 question_type。
_DISTILL_TO_RUBRIC_QUESTION_TYPE: dict[str, str] = {
    # 命令/脚本类 → 命令查询
    "ScriptCommand":        "命令查询",
    # 配置/参数类 → 参数确认
    "ConfigExample":        "参数确认",
    "Factoid":              "参数确认",
    "FillBlank":            "参数确认",
    # 故障/排障/操作类 → 故障处理
    "Diagnostic":           "故障处理",
    "Procedural":           "故障处理",
    "FaultReproFix":        "故障处理",
    "RootCause":            "故障处理",
    "BestPractice":         "故障处理",
    # 差异对比 → 差异对比（P3-C 新增类型；DistillQuestionGenerator 输出此值时激活专项 HC）
    "DifferenceComparison": "差异对比",
    # 版本特性 → 版本特性（P3-C 新增类型；DistillQuestionGenerator 输出此值时激活专项 HC）
    "VersionFeature":       "版本特性",
    # 合规/安全 → 暂无专属 Rubric 类型，走 default
    "ComplianceSecurity":   "default",
    # 客观题 → default（题型差异大，专项 HC 无法适用）
    "MultipleChoice":       "default",
}


def storage_derive_question_type(
    storage,
    distill_type_key: str = "distill_question_type",
    output_type_key: str = "question_type",
) -> None:
    """从 distill_question_type 派生 Rubric 兼容的 question_type 字段。

    必须在 QuestionRefiner 之后、RubricScorer 之前调用。
    若 distill_question_type 不存在或映射不到已知类型，写入 "default"。
    """
    df = storage.read("dataframe").copy()
    if distill_type_key not in df.columns:
        df[output_type_key] = "default"
    else:
        df[output_type_key] = df[distill_type_key].apply(
            lambda v: _DISTILL_TO_RUBRIC_QUESTION_TYPE.get(
                str(v).strip() if v else "", "default"
            )
        )
    storage.write(df)


_TABLE_ROW_REGEX = re.compile(r"[\w\\/._\-]+\s*\t\s*[^\n]+")
# Markdown pipe-table 行（含分隔行）
_MD_PIPE_ROW_RE = re.compile(r"^\s*\|.+\|")
_MD_PIPE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|")


def _is_table_only_chunk(text: str) -> bool:
    """判断 chunk 是否属于"纯表格 / 路径对照 chunk"。

    命中规则（任一即返回 True，跳过不进问题生成）：
    - 短文本 + 高 \\t / </td> 行密度：含 \\t 或 </td> 的行占比 > 60%
      且整段字符总数 < 600
    - 路径+描述对照行占比 > 70%（regex `[\\w\\\\/._\\-]+\\s*\\t\\s*[^\\n]+`）
    - Markdown pipe 表格：含 | 行占比 > 60% 且有 ≥1 条分隔行且字符总数 < 1200
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    total_chars = len(s)
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if not lines:
        return False

    table_like_lines = sum(
        1 for ln in lines if "\t" in ln or "</td>" in ln.lower()
    )
    table_ratio = table_like_lines / max(len(lines), 1)
    if table_ratio > 0.6 and total_chars < 600:
        return True

    path_table_lines = sum(1 for ln in lines if _TABLE_ROW_REGEX.search(ln))
    path_ratio = path_table_lines / max(len(lines), 1)
    if path_ratio > 0.7:
        return True

    # Markdown pipe 表格检测
    pipe_rows = sum(1 for ln in lines if _MD_PIPE_ROW_RE.match(ln))
    sep_rows = sum(1 for ln in lines if _MD_PIPE_SEP_RE.match(ln))
    pipe_ratio = pipe_rows / max(len(lines), 1)
    if pipe_ratio > 0.6 and sep_rows >= 1 and total_chars < 1200:
        return True

    return False


def storage_prepare_meta_triplet_view(
    storage,
    refined_question_key: str = "refined_reverse_question",
    anchor_key: str = "anchor_sentences",
    output_key: str = "meta_triplet_instruction",
):
    """构造 MetaFilter 的质量评估输入视图。

    只将「精炼后的问题」和「原文支撑句」放入 instruction 字段；
    raw_content（context）和 refined_answer 已通过 input_input_key /
    input_output_key 单独传入 MetaFilter，无需在此重复拼接。
    MetaFilter._combine_fields 最终会拼成：
        Question: {meta_triplet_instruction}
        Context: {raw_content}
        Answer: {refined_answer}
    """
    df = storage.read("dataframe").copy()

    def _build(row) -> str:
        question = str(row.get(refined_question_key, "") or "").strip()
        anchors = row.get(anchor_key) or []
        if isinstance(anchors, list):
            anchor_lines = "\n".join(
                f"  -「{s}」"
                for s in anchors
                if isinstance(s, str) and s.strip()
            )
        else:
            anchor_lines = str(anchors).strip()

        parts = [question]
        if anchor_lines:
            parts.append(f"原文支撑句（答案所依据的原文片段）：\n{anchor_lines}")
        return "\n\n".join(parts)

    df[output_key] = df.apply(_build, axis=1)
    storage.write(df)


def analyze_rubric_results(df):
    """打印 Rubric 结果分布（需存在 rubric_result 列）。"""
    if "rubric_result" not in df.columns:
        print("analyze_rubric_results: 缺少列 rubric_result，跳过。")
        return
    rubric_col = df["rubric_result"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    hc_failures = {"HC-1": 0, "HC-2": 0, "HC-3": 0}
    for r in rubric_col:
        for hc_id, hc_val in (r.get("hard_constraints") or {}).items():
            if (
                isinstance(hc_val, dict)
                and hc_val.get("applicable", True) is not False
                and hc_val.get("pass") is False
            ):
                hc_failures[hc_id] = hc_failures.get(hc_id, 0) + 1
    sc_lists = {"SC-1": [], "SC-2": [], "SC-3": [], "SC-4": []}
    for r in rubric_col:
        for sc_id, sc_val in (r.get("soft_scores") or {}).items():
            if sc_id in sc_lists and isinstance(sc_val, dict):
                try:
                    sc_lists[sc_id].append(float(sc_val.get("score", 0)))
                except (TypeError, ValueError):
                    pass
    print("=== HC 失败计数（若有 hard_constraints）===")
    for hc, c in sorted(hc_failures.items(), key=lambda x: -x[1]):
        print(f"  {hc}: {c}")
    print("\n=== SC 平均分 ===")
    for sc, scores in sc_lists.items():
        if scores:
            print(f"  {sc}: {sum(scores) / len(scores):.2f}")
    if "question_type" in df.columns and "rubric_verdict" in df.columns:
        print("\n=== 题型 PASS 比例 ===")
        g = df.groupby("question_type")["rubric_verdict"].apply(
            lambda x: (x == "PASS").mean()
        )
        print(g.sort_values())
    if "rubric_score" in df.columns:
        print("\n=== rubric_score 描述统计 ===")
        print(df["rubric_score"].describe())


class TechDocQAPipelineV2:
    """
    技术文档逆向 QA：蒸馏出题（内置溯源锚定约束）→ 初答(OUTOFSCOPE 过滤) → 溯源 → 改写
    → 答案精炼 → 问题精炼（闭包失败回退 rough_question）→ Rubric(clean vs refined) → Meta 托底。
    """

    def __init__(
        self,
        input_file: str,
        num_questions: int = 4,
        distill_tag: str = "运维",
        min_score: float = 3.5,
        max_score: float = 5.0,
        min_soft_score: float = 12.0,
        rubric_filter: bool = False,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        max_workers: int = DEFAULT_MAX_WORKERS,
        cache_path: str = "./cache_local_qa",
        file_name_prefix: str = "qa_pipeline_v2_step",
        resume: bool = True,
    ):
        self.resume = resume
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="json",
        )
        llm_serving = APILLMServing_request(
            api_url=api_url,
            model_name=model_name,
            max_workers=max_workers,
        )
        self.distill_question_generator = DistillQuestionGenerator(
            llm_serving=llm_serving,
            current_tag=distill_tag,
            count=num_questions,
            two_step_type_selection=True,
            ensure_objective_questions=True,
        )
        self.answer_generator = TextAnswerGenerator(
            llm_serving=llm_serving,
            custom_prompt=TECH_DOC_V2_ANSWER_PROMPT,
        )
        self.answer_grounding_filter = AnswerGroundingFilter(llm_serving=llm_serving)
        self.answer_rewriter = AnswerRewriter(llm_serving=llm_serving)
        self.answer_critique_evaluator = AnswerCritiqueEvaluator(llm_serving=llm_serving)
        self.answer_refiner = AnswerRefiner(llm_serving=llm_serving)
        self.question_refiner = QuestionRefiner(llm_serving=llm_serving)
        self.rubric_scorer = RubricScorer(
            llm_serving=llm_serving,
            min_soft_score=min_soft_score,
            filter_on_run=rubric_filter,
        )
        self.meta_filter = MetaFilter(
            min_score=min_score,
            max_score=max_score,
            llm_serving=llm_serving,
            dimensions=tech_doc_qa_meta_dimensions,
        )

    def _step_output_path(self, output_step: int) -> str:
        return os.path.join(
            self.storage.cache_path,
            f"{self.storage.file_name_prefix}_step{output_step}.json",
        )

    def _step_output_ready(self, output_step: int, min_bytes: int = 8) -> bool:
        if not self.resume:
            return False
        path = self._step_output_path(output_step)
        if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
            return False
        try:
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == 0:
                return False
        except Exception:
            return False
        return True

    def _find_latest_cached_step(self) -> int:
        cache_dir = self.storage.cache_path
        prefix = self.storage.file_name_prefix
        latest = 0
        if not os.path.isdir(cache_dir):
            return 0
        pat = re.compile(rf"^{re.escape(prefix)}_step(\d+)\.json$")
        for name in os.listdir(cache_dir):
            m = pat.match(name)
            if m:
                latest = max(latest, int(m.group(1)))
        return latest

    def _log_resume_skip(self, label: str, output_step: int) -> None:
        path = self._step_output_path(output_step)
        print(
            f"[resume] skip {label} "
            f"(已有 {os.path.basename(path)}, {os.path.getsize(path)} bytes)"
        )

    def forward(self):
        s = self.storage
        if self.resume:
            latest = self._find_latest_cached_step()
            if latest > 0:
                print("=" * 50)
                print(f"[resume] 检测到 step 缓存 step{latest}.json，已完成的步骤将跳过")
                print("=" * 50)

        print("=" * 50)
        print("V2 Step 0: Filter table-only chunks (pre-filter)")
        print("=" * 50)
        pre_step = s.step()
        if self._step_output_ready(1):
            self._log_resume_skip("Step 0 pre-filter", 1)
        else:
            df_in = pre_step.read("dataframe")
            n_in = len(df_in)
            if "text" in df_in.columns:
                mask = df_in["text"].apply(lambda t: not _is_table_only_chunk(t))
                df_kept = df_in.loc[mask].reset_index(drop=True)
                n_skipped = n_in - len(df_kept)
                pre_step.write(df_kept)
                print(
                    f"[pre-filter] table-only chunks skipped: {n_skipped}/{n_in} "
                    f"(kept {len(df_kept)})"
                )
            else:
                print("[pre-filter] no 'text' column found, skip pre-filter.")

        print("=" * 50)
        print("V2 Step 1: Distill questions")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(2):
            self._log_resume_skip("Step 1 Distill questions", 2)
        else:
            out_keys = self.distill_question_generator.run(
                storage=st,
                input_context_key="text",
                output_question_key="rough_question",
                output_context_key="raw_content",
            )
            print(f"Distill generator output columns: {out_keys}")

        print("=" * 50)
        print("V2 Step 2: Initial answers")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(3):
            self._log_resume_skip("Step 2 Initial answers", 3)
        else:
            self.answer_generator.run(
                storage=st,
                input_question_key="rough_question",
                input_content_key="raw_content",
                output_instruction_key="instruction",
                output_answer_key="initial_answer",
                output_content_key="raw_content",
                preserve_input_columns=True,
            )

        print("V2 Step 2.5: Strip thinking tags from initial_answer")
        st = s.step()
        if self._step_output_ready(4):
            self._log_resume_skip("Step 2.5 strip initial_answer", 4)
        else:
            storage_strip_llm_answer_columns(st, ["initial_answer"])

        # 行级过滤：OUTOFSCOPE token + 元信息托词答案，统一用 is_outofscope_or_meta_tox 判断。
        # 提前在 grounding 之前过掉，可省 LLM 调用 token。
        st = s.step()
        if self._step_output_ready(5):
            self._log_resume_skip("Step 2.6 filter initial_answer", 5)
        else:
            storage_apply_row_mask(st, _initial_answer_valid)

        print("=" * 50)
        print("V2 Step 3: Answer grounding")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(6):
            self._log_resume_skip("Step 3 Answer grounding", 6)
        else:
            self.answer_grounding_filter.run(
                storage=st,
                input_context_key="raw_content",
                input_question_key="rough_question",
                input_answer_key="initial_answer",
            )

        st = s.step()
        if self._step_output_ready(7):
            self._log_resume_skip("Step 3.5 filter grounding DROP", 7)
        else:
            storage_apply_row_mask(st, _grounding_not_drop)

        print("=" * 50)
        print("V2 Step 4: Answer rewrite (clean_answer)")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(8):
            self._log_resume_skip("Step 4 Answer rewrite", 8)
        else:
            # 注意：AnswerRewriter 默认 output_stage_key='rewrite_stage'（避免覆盖
            # 上游 grounding_verdict），并在内部对 OUTOFSCOPE / 元信息托词答案做兜底 DROP。
            self.answer_rewriter.run(storage=st)

        st = s.step()
        if self._step_output_ready(9):
            self._log_resume_skip("Step 4.5 strip clean_answer", 9)
        else:
            storage_strip_llm_answer_columns(st, ["clean_answer"])

        st = s.step()
        if self._step_output_ready(10):
            self._log_resume_skip("Step 4.6 filter clean_answer", 10)
        else:
            storage_apply_row_mask(st, _has_clean_answer)

        print("=" * 50)
        print("V2 Step 5: Answer critique")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(11):
            self._log_resume_skip("Step 5 Answer critique", 11)
        else:
            self.answer_critique_evaluator.run(
                storage=st,
                input_context_key="raw_content",
                input_question_key="rough_question",
                input_answer_key="clean_answer",
            )

        print("=" * 50)
        print("V2 Step 6: Answer refiner -> refined_answer")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(12):
            self._log_resume_skip("Step 6 Answer refiner", 12)
        else:
            self.answer_refiner.run(
                storage=st,
                input_context_key="raw_content",
                input_question_key="rough_question",
                input_answer_key="clean_answer",
                output_answer_key="refined_answer",
                reuse_existing_critique=True,
            )

        st = s.step()
        if self._step_output_ready(13):
            self._log_resume_skip("Step 6.5 strip refined_answer", 13)
        else:
            storage_strip_llm_answer_columns(st, ["refined_answer"])

        st = s.step()
        if self._step_output_ready(14):
            self._log_resume_skip("Step 6.6 filter refined_answer", 14)
        else:
            # force_rewrite 廉价兜底：精炼后仍含 OUTOFSCOPE / 元信息托词则丢弃
            storage_apply_row_mask(st, _refined_answer_valid)

        print("=" * 50)
        print("V2 Step 7: Question refiner -> refined_reverse_question")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(15):
            self._log_resume_skip("Step 7 Question refiner", 15)
        else:
            # reverseQuestionBuilder 已移除，改用 rough_question（distillQuestionGen 产出）
            # 作为 QuestionRefiner 的输入；事实闭包失败时自动回退到 rough_question。
            # input_answer_key 传入 refined_answer，使提示词中的事实闭包校验能真正对照答案内容。
            self.question_refiner.run(
                storage=st,
                input_context_key="raw_content",
                input_question_key="rough_question",
                input_answer_key="refined_answer",
                output_question_key="refined_reverse_question",
            )

        st = s.step()
        if self._step_output_ready(16):
            self._log_resume_skip("Step 7.5 derive question_type", 16)
        else:
            # Step 7.5: 将 distill_question_type（轴B）映射为 Rubric 兼容的 question_type
            # 使 RubricScorer 能走题型专属 HC/OC，而非全部 fallback 到 "default"
            storage_derive_question_type(st)

        st = s.step()
        if self._step_output_ready(17):
            self._log_resume_skip("Step 7.6 strip clean/refined answers", 17)
        else:
            storage_strip_llm_answer_columns(st, ["clean_answer", "refined_answer"])

        print("=" * 50)
        print("V2 Step 8: Rubric (standard=clean_answer, candidate=refined_answer)")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(18):
            self._log_resume_skip("Step 8 Rubric scorer", 18)
        else:
            self.rubric_scorer.run(
                storage=st,
                input_context_key="raw_content",
                input_question_key="refined_reverse_question",
                input_standard_answer_key="clean_answer",
                input_answer_key="refined_answer",
                input_type_key="question_type",
                input_anchors_key="anchor_sentences",
            )

        st = s.step()
        if self._step_output_ready(19):
            self._log_resume_skip("Step 8.5 Rubric PASS filter", 19)
        else:
            # Rubric 本身不过滤（保留 FAIL 行及 rubric_result 供审计），这里单独做一次 PASS 过滤
            storage_apply_row_mask(st, _rubric_pass)

        print("=" * 50)
        print("V2 Step 9: Meta filter (triple consistency)")
        print("=" * 50)
        st = s.step()
        if self._step_output_ready(20):
            self._log_resume_skip("Step 9.0 meta triplet view", 20)
        else:
            storage_prepare_meta_triplet_view(st)

        st = s.step()
        if self._step_output_ready(21):
            self._log_resume_skip("Step 9 Meta filter", 21)
        else:
            self.meta_filter.run(
                storage=st,
                input_instruction_key="meta_triplet_instruction",
                input_input_key="raw_content",
                input_output_key="refined_answer",
                output_key="MetaScore",
            )

        final_st = s.step()
        df_final = final_st.read("dataframe")
        analyze_rubric_results(df_final)
        print("=" * 50)
        print(f"V2 Pipeline completed. Final rows: {len(df_final)}")
        print("=" * 50)


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
        num_questions: int = 4,
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
            api_url=DEFAULT_API_URL,
            model_name=DEFAULT_MODEL_NAME,
            max_workers=DEFAULT_MAX_WORKERS,
        )
        
        # 第一阶段：问题生成器
        self.distill_question_generator = DistillQuestionGenerator(
            llm_serving=llm_serving,
            current_tag= "运维",
            count = num_questions
        )
        # self.question_generator = TextQuestionGenerator(
        #     llm_serving=llm_serving,
        #     number=num_questions,
        #     custom_prompt=question_custom_prompt
        # )
        self.question_refiner = QuestionRefiner(
            llm_serving=llm_serving
        )
        self.answer_refiner = AnswerRefiner(
            llm_serving=llm_serving
        )
        # self.evol_instruct = EvolInstructGenerator(
        #     llm_serving=llm_serving,
        #     evolution_type='all',           # 随机使用深度或广度进化
        #     generate_answer=False,           # 同时生成答案
        #     num_evolutions=1                # 每条指令进化 1 次
        # )
        # 第二阶段：答案生成器
        self.answer_generator = TextAnswerGenerator(
            llm_serving=llm_serving,
            custom_prompt=answer_custom_prompt
        )
        # 第三阶段：质量过滤
        self.question_filter = MetaFilter(
            min_score=min_score,           # 均值最低分
            max_score=max_score,           # 均值最高分
            llm_serving=llm_serving,
            dimensions=question_dimensions  # 可选，不传则使用默认6维度
        )
        self.answer_filter = MetaFilter(
            min_score=min_score,           # 均值最低分
            max_score=max_score,           # 均值最高分
            llm_serving=llm_serving,
            dimensions=answer_dimensions  # 可选，不传则使用默认6维度
        )
    def forward(self):
        """执行 Pipeline"""
        
        # 步骤 1: 从文本生成多个问题
        print("=" * 50)
        print("Step 1: Generating questions from text...")
        print("=" * 50)
        # self.evol_instruct.run(
        #     storage=self.storage.step(),
        #     input_instruction_key="question",
        #     output_instruction_key="evolved_quetsion",
        #     output_original_key="original_question"
        # )
        self.distill_question_generator.run(
            storage=self.storage.step(),
            input_context_key='text',
            output_question_key='question',
            output_context_key='raw_content',
        )
        self.question_refiner.run(
            storage=self.storage.step()
        )
        self.question_filter.run(
            storage=self.storage.step(),
            input_instruction_key='question',      # 原始问题（供对比参照）
            input_input_key='raw_content',         # 上下文字段
            input_output_key='refined_question',   # 被评估的精炼问题
            output_key='MetaScore'                 # 输出均值分数字段
        )
        # 步骤 2: 为每个问题生成答案
        print("=" * 50)
        print("Step 2: Generating answers for questions...")
        print("=" * 50)
        self.answer_generator.run(
            storage=self.storage.step(),
            input_question_key='refined_question',
            input_content_key='raw_content',
            output_instruction_key='instruction',
            output_answer_key='output',
            output_content_key='raw_content',
            preserve_input_columns=False,
        )
        self.answer_refiner.run(
            storage=self.storage.step(),
            input_answer_key='output',
            input_question_key='instruction'
        )
        self.answer_filter.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',  # 问题字段
            input_input_key='raw_content',         # 上下文字段
            input_output_key='refined_answer',             # 答案字段
            output_key='MetaScore'                 # 输出均值分数字段
        )
        
        print("=" * 50)
        print("Pipeline completed!")
        print("=" * 50)


if __name__ == "__main__":
    # 用法：
    #   export DF_API_KEY=...   # 必填
    #   export DF_API_URL=http://ai.wenmodel.com/v1/chat/completions
    #   export DF_MODEL_NAME=qwen-plus
    #   python test_filter.py              # 旧版 TextQAPipeline
    #   python test_filter.py v2           # 技术文档逆向链路 TechDocQAPipelineV2
    #   TECHDOC_INPUT=/path/to.json python test_filter.py v2
    default_input = os.environ.get(
        "TECHDOC_INPUT",
        "/home/wugk/finetune/finetune/dataflow.json",
    )
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("v2", "--v2"):
        TechDocQAPipelineV2(
            input_file=default_input,
            num_questions=4,
            min_score=3.5,
            max_score=5.0,
            min_soft_score=12.0,
            max_workers=DEFAULT_MAX_WORKERS,
        ).forward()
    else:
        pipeline = TextQAPipeline(
            input_file=default_input,
            num_questions=4,
            question_custom_prompt="你是一名运维领域的专家，生成的问题必须是运维领域中有价值的技术问题",
            answer_custom_prompt="答案必须严格按照上下文生成, 操作步骤要详细不能进行总结概要",
            min_score=4.5,
            max_score=5,
        )
        pipeline.forward()

