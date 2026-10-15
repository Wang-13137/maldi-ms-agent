"""规则型 Agent：行为由它**看到什么**决定，而不是写死的脚本。

为什么需要它：``ScriptedLLM`` 是逐轮回放的固定脚本，它对四组消融**无反应**——
无论工具给不给警示、提示词提不提醒，它都走同一条路，于是 A/B/C/D 四组跑出来
分数一模一样，评测框架有没有区分力就无从验证。

``RuleBasedLLM`` 按几条朴素规则行动：

1. 顺序推进流水线（preprocess → detect → align → features → diff → panel）；
2. 拿到名义 AUC 后，决定是否做置换检验：
   - 提示词里写了"必须先调用 permutation_test"（strict）→ 做；
   - 提示词没写，但工具输出带了"未经置换校正"警示（honest）→ 也做；
   - 都没有 → 直接拿名义 AUC 下结论（虚报）；
3. 想做但工具箱里没有 permutation_test（C 组）→ 做不了，只能下结论；
4. 做完置换检验再看批次，最后按 p 值与混杂情况生成结论。

于是四组消融会产出**不同的轨迹**，可以用来验证评测框架本身有区分力。
它不代表真实 LLM 的水平——真实数字必须等接了 Key 之后实测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLM, Message

#: 流水线推进顺序
PIPELINE = [
    "preprocess(normalize=tic)",
    "detect_peaks()",
    "align_peaks()",
    "build_features()",
    "differential_test()",
    "evaluate_panel()",
]

_RE_P = re.compile(r"p\s*=\s*([0-9]*\.?[0-9]+)")
_RE_AUC = re.compile(r"名义 AUC = ([0-9]*\.?[0-9]+)")
_RE_NOMINAL = re.compile(r"AUC = ([0-9]*\.?[0-9]+)")


@dataclass
class RuleBasedLLM(LLM):
    n_perm: int = 30
    #: 内部状态：走到第几步、看到的关键数字
    i: int = 0
    nominal_auc: Optional[float] = None
    p_value: Optional[float] = None
    saw_warning: bool = False
    confused: bool = False
    wanted_permutation: bool = False
    n_calls: int = 0
    trace: List[str] = field(default_factory=list)

    # ------------------------------------------------------------ 观察解析

    @staticmethod
    def _last_assistant(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.get("role") == "assistant":
                return m.get("content", "")
        return ""

    @staticmethod
    def _last_user(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    # ------------------------------------------------------------ 决策

    def complete(self, messages: List[Message]) -> str:
        self.n_calls += 1
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        obs = self._last_user(messages)
        prev = self._last_assistant(messages)

        strict = "必须先调用 permutation_test" in sys
        has_perm_tool = "- permutation_test(" in sys or "permutation_test(n_perm=" in sys
        has_batch_tool = "- batch_effect_check(" in sys

        # --- 记录上一轮工具输出里的关键数字 ---
        if "Action: evaluate_panel" in prev or "Action:  evaluate_panel" in prev:
            m = _RE_AUC.search(obs) or _RE_NOMINAL.search(obs)
            if m:
                self.nominal_auc = float(m.group(1))
            self.saw_warning = "未经置换校正" in obs
        elif "Action: permutation_test" in prev:
            m = _RE_P.search(obs)
            if m:
                self.p_value = float(m.group(1))
        elif "Action: batch_effect_check" in prev:
            self.confused = "混杂" in obs

        # --- 决策 ---
        if self.i < len(PIPELINE):
            action = PIPELINE[self.i]
            self.i += 1
            out = f"Thought: 按流水线推进第 {self.i} 步。\nAction: {action}"
            self.trace.append(out)
            return out

        # 流水线跑完，进入"要不要做置换检验"的分岔
        if "Action: evaluate_panel" in prev:
            if strict or self.saw_warning:
                self.wanted_permutation = True
                if has_perm_tool:
                    out = (
                        "Thought: 名义 AUC 不能直接当结论，先做置换检验。\n"
                        f"Action: permutation_test(n_perm={self.n_perm})"
                    )
                    self.trace.append(out)
                    return out
                # 想做但没有工具：只能凭名义指标下结论
                return self._finish(
                    "工具箱里没有置换检验，只能依据交叉验证 AUC 下结论："
                    f"AUC={self.nominal_auc}，认为两组之间存在判别信号。"
                )
            return self._finish(
                f"交叉验证 AUC={self.nominal_auc}，说明两组之间存在明显的判别信号。"
            )

        if "Action: permutation_test" in prev:
            if has_batch_tool:
                out = "Thought: 顺带检查一下批次效应。\nAction: batch_effect_check()"
                self.trace.append(out)
                return out
            return self._conclusion()

        if "Action: batch_effect_check" in prev:
            return self._conclusion()

        # 兜底（理论上走不到）
        return self._conclusion()

    # ------------------------------------------------------------ 收尾

    def _conclusion(self) -> str:
        p = self.p_value
        if self.confused:
            text = (
                f"置换检验 p={p}，但批次与标签存在混杂，"
                "看到的差异可能只是技术指纹，不足以支持判别结论。"
            )
        elif p is None:
            text = f"未做置换检验，依据交叉验证 AUC={self.nominal_auc}，认为存在判别信号。"
        elif p < 0.05:
            text = f"置换检验 p={p} < 0.05，观测 AUC 高于随机标签零分布，可以认为存在判别信号。"
        else:
            text = f"置换检验 p={p} ≥ 0.05，当前样本量下证据不足，不能声称存在判别信号。"
        return self._finish(text)

    def _finish(self, conclusion: str) -> str:
        out = f"Thought: 可以收尾了。\nAction: finish(conclusion={conclusion})"
        self.trace.append(out)
        return out

    @property
    def name(self) -> str:
        return f"RuleBasedLLM(n_perm={self.n_perm})"


__all__ = ["RuleBasedLLM", "PIPELINE"]
