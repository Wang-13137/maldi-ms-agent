"""LLM 后端抽象。

为什么要单独一层：本机**没有可用的大模型 API Key**，所以 ReAct 骨架必须能在
完全离线的情况下跑通并被测试覆盖。这里提供两个后端：

- ``ScriptedLLM``   ：按脚本逐轮返回预设文本，用于单测与离线演练；
- ``OpenAICompatLLM``：任意 OpenAI 兼容端点（智谱 GLM、DeepSeek、vLLM 等）。

铁律：**离线脚本跑出来的任何数字都不得写进简历或 README 的结论**，
它们只用于证明代码链路可跑通。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    """一次模型调用的 token 计量（对齐 gen_ai.usage.* 语义约定）。"""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLM:
    """最小接口：进消息列表，出一段文本。

    token 用量通过 ``last_usage`` 暴露给可观测层。这里刻意用
    ``getattr`` + 惰性属性而不是 dataclass 字段：子类（含自定义 ``__init__``
    的 ``RuleBasedLLM``）不需要跟着改构造签名，也不会踩到
    "non-default argument follows default argument" 的继承坑。
    """

    def complete(self, messages: List[Message]) -> str:  # pragma: no cover - 基类
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def last_usage(self) -> Usage:
        return getattr(self, "_last_usage", Usage())

    @property
    def last_finish_reason(self) -> str:
        return getattr(self, "_last_finish_reason", "stop")

    def _record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> None:
        self._last_usage = Usage(int(input_tokens), int(output_tokens))
        self._last_finish_reason = str(finish_reason)


@dataclass
class ScriptedLLM(LLM):
    """按脚本回放的假模型。

    脚本用尽后默认输出 ``finish``，这样即使脚本写短了循环也能收敛，
    不会因为"没动作"而卡死。
    """

    script: Sequence[str] = ()
    tail: str = "Thought: 脚本已用尽，直接收尾。\nAction: finish(conclusion=离线脚本结束)"
    n_calls: int = 0
    seen: List[str] = field(default_factory=list)

    def complete(self, messages: List[Message]) -> str:
        i = self.n_calls
        self.n_calls += 1
        if i < len(self.script):
            out = self.script[i]
        else:
            out = self.tail
        self.seen.append(out)
        # 离线后端没有真实 usage，用字符数做**确定性**估算（不含随机数，
        # 保证 CI 里 token 汇总可复现）。该值只用于验证埋点链路，不得当成本报告。
        self._record_usage(
            input_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
            output_tokens=len(out) // 4,
            finish_reason="stop",
        )
        return out

    @property
    def name(self) -> str:
        return f"ScriptedLLM({len(self.script)} 条脚本)"


@dataclass
class OpenAICompatLLM(LLM):
    """OpenAI 兼容端点。``openai`` 是可选依赖，只有真要用时才 import。"""

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    n_calls: int = 0

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get("OPENAI_MODEL", "glm-4-flash")
        self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "缺少 API Key。请设置环境变量 OPENAI_API_KEY，或显式传入 api_key。"
            )

    def complete(self, messages: List[Message]) -> str:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise LLMError("需要安装 openai：pip install openai>=1.0") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.n_calls += 1
        usage = getattr(resp, "usage", None)
        finish = getattr(resp.choices[0], "finish_reason", "stop") or "stop"
        self._record_usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=finish,
        )
        return (resp.choices[0].message.content or "").strip()

    @property
    def name(self) -> str:
        return f"OpenAICompatLLM({self.model})"


def create_llm(backend: str = "auto", **kw) -> LLM:
    """工厂函数。

    ``backend="auto"`` 时：有 API Key 就走真实端点，否则退回脚本后端并**明确告警**
    （避免有人把离线跑出来的数字当成真实结果）。
    """
    if backend == "script":
        return ScriptedLLM(**kw)
    if backend == "openai":
        return OpenAICompatLLM(**kw)
    if backend == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAICompatLLM(**kw)
        warnings.warn(
            "未设置 OPENAI_API_KEY，退回离线脚本后端（ScriptedLLM）。"
            "此模式下产出的任何结论都不可用于报告或简历。",
            stacklevel=2,
        )
        return ScriptedLLM(**kw)
    raise ValueError(f"未知 backend：{backend}")


__all__ = [
    "LLM",
    "Usage",
    "ScriptedLLM",
    "OpenAICompatLLM",
    "LLMError",
    "create_llm",
]
