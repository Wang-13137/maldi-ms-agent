"""运行时预算硬熔断（M8）。

M7 之前只有 ``max_steps`` 步数上限，挡不住三类真实生产故障：

- 模型在长上下文里反复重试，token 烧穿配额；
- 某次工具调用卡在重计算（置换检验尤其贵），墙钟时间失控；
- Agent 陷入工具循环，短时间内打爆下游。

这里把预算做成**可注入时钟、可解释、可观测**的一等对象：

- 每轮循环开始查一次墙钟/步数，每次 LLM 返回后累加 token，每次工具调用后累加次数；
- 任何一项超限立即停止，``stopped_reason="budget_exceeded"``，
  ``error`` 写明是哪一类、用了多少、上限多少——不是笼统的"失败"；
- 用量与限额写进 ``app.budget.*`` span 属性，随 OTLP 一起导出；
- 默认 ``budget=None``，行为与 M5/M7 完全一致，四组消融的评测口径不受影响。

刻意不做的事：不在超预算时"让模型再总结一句"。熔断的价值在于**确定地停**，
给模型最后一次发言机会等于给了它最后一次虚报结论的机会。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class Budget:
    """硬预算。``None`` 表示该维度不限制。"""

    max_total_tokens: Optional[int] = None
    max_wall_seconds: Optional[float] = None
    max_tool_calls: Optional[int] = None
    max_steps: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_steps": self.max_steps,
        }


@dataclass
class BudgetState:
    """一次 run 的用量账本。``clock`` 可注入，便于确定性测试。"""

    budget: Budget
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(default_factory=time.monotonic)
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    steps: int = 0
    exceeded_reason: str = ""

    def __post_init__(self) -> None:
        # 用注入的时钟取起点，测试里才能确定性地推进"墙钟时间"。
        self.started_at = self.clock()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def add_usage(self, usage: Any) -> None:
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def tick_tool(self) -> None:
        self.tool_calls += 1

    def tick_step(self) -> None:
        self.steps += 1

    def check(self) -> str:
        """返回首个超限原因；未超限返回空串。超限值一旦触发即锁定，保证幂等。"""
        if self.exceeded_reason:
            return self.exceeded_reason
        b, reason = self.budget, ""
        if b.max_total_tokens is not None and self.total_tokens > b.max_total_tokens:
            reason = f"token 预算超出：{self.total_tokens} > {b.max_total_tokens}"
        elif b.max_wall_seconds is not None and self.elapsed_seconds > b.max_wall_seconds:
            reason = (
                f"墙钟时间预算超出：{self.elapsed_seconds:.3f}s > "
                f"{b.max_wall_seconds:.3f}s"
            )
        elif b.max_tool_calls is not None and self.tool_calls > b.max_tool_calls:
            reason = f"工具调用预算超出：{self.tool_calls} > {b.max_tool_calls}"
        elif b.max_steps is not None and self.steps > b.max_steps:
            reason = f"步数预算超出：{self.steps} > {b.max_steps}"
        self.exceeded_reason = reason
        return reason

    def snapshot(self) -> Dict[str, Any]:
        """给 span 属性 / OTLP 导出用的扁平快照。"""
        return {
            "limit_total_tokens": self.budget.max_total_tokens,
            "limit_wall_seconds": self.budget.max_wall_seconds,
            "limit_tool_calls": self.budget.max_tool_calls,
            "limit_steps": self.budget.max_steps,
            "used_total_tokens": self.total_tokens,
            "used_wall_seconds": round(self.elapsed_seconds, 4),
            "used_tool_calls": self.tool_calls,
            "used_steps": self.steps,
            "exceeded": bool(self.exceeded_reason),
            "exceeded_reason": self.exceeded_reason,
        }

    @classmethod
    def from_snapshot(
        cls,
        budget: "Budget",
        snap: Dict[str, Any],
        clock: Optional[Callable[[], float]] = None,
    ) -> "BudgetState":
        """从检查点快照恢复账本（M9 续跑用）。

        只恢复**计数类**用量（token / 工具调用 / 步数），墙钟时间重置为当前时刻——
        续跑视为一次新的墙钟窗口，这是合理且可解释的。token 只记总和，这里把
        ``used_total_tokens`` 整体记到 ``input_tokens``（``total = input + output``
        不变，预算判定等价）。``exceeded_reason`` 清零，让续跑重新走完整判定。
        """
        bs = cls(budget, clock=clock if clock is not None else time.monotonic)
        bs.input_tokens = int(snap.get("used_total_tokens", 0))
        bs.output_tokens = 0
        bs.tool_calls = int(snap.get("used_tool_calls", 0))
        bs.steps = int(snap.get("used_steps", 0))
        bs.exceeded_reason = ""
        return bs


__all__ = ["Budget", "BudgetState"]
