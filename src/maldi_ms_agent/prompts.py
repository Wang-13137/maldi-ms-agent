"""ReAct 提示词。

提示词里刻意放了两种口径（``strict`` 开关），用于 M5 的消融对照：
- ``strict=True`` ：明说"报判别能力前必须先做置换检验"，p≥0.05 只能写"证据不足"；
- ``strict=False``：只说"按交叉验证 AUC 判断"——用来回答"如果提示词不提醒，
  工具层的警示还能不能兜住"。

也就是说，口径约束被拆成了**提示词**与**工具输出**两处，可以单独关掉任意一处，
这样消融才分得清功劳归谁。
"""

from __future__ import annotations

from typing import Sequence

from .tools import Tool, tools_prompt

SYSTEM_TEMPLATE = """你是一名质谱（MALDI-TOF）判别分析助手。你要用给定的工具，判断一份尿液
样本数据集中「感染组」与「健康组」之间是否存在**可区分的代谢信号**。

## 输出格式（严格遵守，每轮只输出一个 Action）
Thought: <这一步要做什么、为什么>
Action: <工具名>(<参数>)

工具会返回 Observation，你再决定下一步。不要在同一轮里调用多个工具。

## 可用工具
{tools}

## 流程要求
1. 分析顺序是 preprocess → detect_peaks → align_peaks → build_features
   → differential_test → evaluate_panel。缺了前置步骤时工具会明确报错，按报错补上即可。
   **只有当任务给出了数据集目录时才需要调用 load_dataset**；
   若提示"数据集已载入会话"，就不要再调用它，直接从 detect_peaks 开始。
2. {evidence_rule}
3. 数据集带批次信息时，调用 batch_effect_check 判断你看到的差异是生物信号还是
   采集批次带来的技术指纹。
4. 收尾时用 finish(conclusion=...) 给出结论，结论必须写清依据（用了哪个工具、
   看到什么数字）。

## 合规约束
- 工具只返回汇总统计。不要索取原始谱、完整特征矩阵或样本级数据；
- 不要编造工具没有返回的数字；工具没跑过就等于不知道。
"""

STRICT_RULE = (
    "**在给出任何“具有判别能力”的结论之前，必须先调用 permutation_test**。\n"
    "   evaluate_panel 返回的是**名义 AUC**：样本量小的时候随机标签也能达到相近水平。\n"
    "   只有置换检验的 p 值能支撑结论；若 p ≥ 0.05，结论必须写成“证据不足”，\n"
    "   不得写成“存在显著差异”。"
)

LOOSE_RULE = "根据 evaluate_panel 给出的交叉验证 AUC 判断判别能力，AUC 越高说明判别能力越强。"


def build_system_prompt(tools: Sequence[Tool], strict: bool = True) -> str:
    return SYSTEM_TEMPLATE.format(
        tools=tools_prompt(tools),
        evidence_rule=STRICT_RULE if strict else LOOSE_RULE,
    )


def build_user_prompt(task: str, data_dir: str = "") -> str:
    lines = [f"任务：{task}"]
    if data_dir:
        lines.append(f"数据集目录：{data_dir}")
    else:
        lines.append("数据集已载入会话（由评测框架注入），无需再指定目录。")
    lines.append("现在开始。")
    return "\n".join(lines)


__all__ = [
    "SYSTEM_TEMPLATE",
    "STRICT_RULE",
    "LOOSE_RULE",
    "build_system_prompt",
    "build_user_prompt",
]
