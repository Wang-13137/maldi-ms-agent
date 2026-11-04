"""断点续跑检查点（M9）。

M8 给 ReAct 循环加了硬预算熔断：预算耗尽时 ``stopped_reason="budget_exceeded"``，
循环**确定地停**。但"停"本身还不够——生产里一次分析可能要跑很多步、耗时较长，
中途因为预算 / 墙钟被打断后，应当能从断点**续跑**而不是从头再来。

``Checkpoint`` 把一次 run 的可恢复状态（对话历史、已完成步骤、累计工具调用、
预算账本快照、卡壳检测状态）序列化成 JSON：

- 纯数据、无对象引用，可直接 ``save()`` / ``load()``；
- ``from_run()`` 从 ``RunResult`` 抽取；``run_react(checkpoint=...)`` 反序列化后
  从下一个步号接着跑，**不会重放**已完成的步骤；
- 已 ``finish`` 的检查点续跑是幂等 no-op（直接返回原结果，不发起任何 LLM 调用）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = dict


@dataclass
class Checkpoint:
    task: str
    data_dir: str
    max_steps: int
    strict: bool
    honest: bool
    with_permutation: bool
    backend: str
    messages: List[Message] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    budget_snapshot: Optional[Dict[str, Any]] = None
    last_sig: str = ""
    repeat_streak: int = 0
    stopped_reason: str = ""
    conclusion: str = ""

    # ---------------------------------------------------------- 序列化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "maldi-ms-agent/checkpoint@1",
            "task": self.task,
            "data_dir": self.data_dir,
            "max_steps": self.max_steps,
            "strict": self.strict,
            "honest": self.honest,
            "with_permutation": self.with_permutation,
            "backend": self.backend,
            "messages": self.messages,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "budget_snapshot": self.budget_snapshot,
            "last_sig": self.last_sig,
            "repeat_streak": self.repeat_streak,
            "stopped_reason": self.stopped_reason,
            "conclusion": self.conclusion,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            task=d["task"],
            data_dir=d["data_dir"],
            max_steps=d["max_steps"],
            strict=d["strict"],
            honest=d["honest"],
            with_permutation=d["with_permutation"],
            backend=d["backend"],
            messages=list(d.get("messages", [])),
            steps=list(d.get("steps", [])),
            tool_calls=list(d.get("tool_calls", [])),
            budget_snapshot=d.get("budget_snapshot"),
            last_sig=d.get("last_sig", ""),
            repeat_streak=int(d.get("repeat_streak", 0)),
            stopped_reason=d.get("stopped_reason", ""),
            conclusion=d.get("conclusion", ""),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @property
    def finished(self) -> bool:
        return self.stopped_reason == "finish"

    @classmethod
    def from_run(cls, run, messages, last_sig, repeat_streak) -> "Checkpoint":
        """从一次 ``RunResult`` 抽取可续跑状态。``messages`` 为最终对话历史。"""
        bsnap = run.budget_state.snapshot() if run.budget_state is not None else None
        return cls(
            task=run.task,
            data_dir=run.data_dir,
            max_steps=run.max_steps,
            strict=run.strict,
            honest=run.honest,
            with_permutation=run.with_permutation,
            backend=run.backend,
            messages=list(messages),
            steps=[s.to_dict() for s in run.steps],
            tool_calls=list(run.tool_calls),
            budget_snapshot=bsnap,
            last_sig=last_sig,
            repeat_streak=repeat_streak,
            stopped_reason=run.stopped_reason,
            conclusion=run.conclusion,
        )
