"""结构化可观测层：对齐 OpenTelemetry GenAI 语义约定。

为什么自己写而不是直接引入 ``opentelemetry-sdk``：

- 本项目的 CI 要在 py3.10/3.11/3.12 上、没有 collector、没有网络凭据的情况下跑通；
  引入 SDK + OTLP exporter 会把「行为门禁」变成「网络可用性测试」。
- 但**输出格式必须对**：span 名（``invoke_agent`` / ``chat`` / ``execute_tool``）、
  属性键（``gen_ai.*``）、以及 OTLP/JSON 结构都按 OTel GenAI semantic conventions
  来写。将来接 Langfuse / Arize Phoenix / 任意 OTLP 后端，只需换一个 exporter，
  **不用重新埋点**。同时补 ``openinference.span.kind`` 便于按 span 类型检索。

合规默认（企业级要求，也是本项目的硬约束）：``capture_content=False``。
默认只记录元数据——模型名、token 数、耗时、工具名、调用状态、run_id；
**不记录 prompt、工具参数、工具返回文本**。本项目处理的质谱数据可能含未发表的
真实样本，内容采集必须显式打开，且打开后仍需自行确认数据出域合规。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: 语义约定仍在 Development 状态，这里显式标注版本口径，避免属性名漂移时静默失配。
SEMCONV_NOTE = "gen_ai.* (OTel GenAI semconv, Development)"
SCOPE_NAME = "maldi-ms-agent"

#: 一旦出现在 span 属性里，就意味着采集了内容而非元数据。
CONTENT_ATTRIBUTE_PREFIXES = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "app.content.",
)

_STATUS_CODE = {"UNSET": 0, "OK": 1, "ERROR": 2}


def _new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


def _attr_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_attr_value(v) for v in value]}}
    return {"stringValue": str(value)}


@dataclass
class Span:
    """一条 span。字段命名对齐 OTLP/JSON，便于零转换导出。"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str                      # AGENT / LLM / TOOL（OpenInference span kind）
    start_ns: int
    end_ns: int = 0
    status: str = "UNSET"          # UNSET / OK / ERROR
    status_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.end_ns or time.time_ns()
        return max(0.0, (end - self.start_ns) / 1_000_000)

    def set_status(self, status: str, message: str = "") -> None:
        if status not in _STATUS_CODE:
            raise ValueError(f"未知 span 状态：{status}")
        self.status = status
        self.status_message = message

    def finish(self, status: str = "OK", message: str = "") -> None:
        if self.status == "UNSET":
            self.set_status(status, message)
        self.end_ns = time.time_ns()

    def to_otlp(self) -> Dict[str, Any]:
        span: Dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": 1,             # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(self.end_ns or time.time_ns()),
            "attributes": [
                {"key": k, "value": _attr_value(v)}
                for k, v in sorted(self.attributes.items())
            ],
            "status": {
                "code": _STATUS_CODE.get(self.status, 0),
                "message": self.status_message,
            },
        }
        if self.parent_span_id:
            span["parentSpanId"] = self.parent_span_id
        return span


@dataclass
class TraceSummary:
    """一次 run 的聚合指标：成本、延迟、调用量、失败率。

    这是「dollar-per-decision」视角的原始数据，不含任何样本内容。
    """

    run_id: str
    trace_id: str
    duration_ms: float
    n_steps: int
    llm_calls: int
    tool_calls: int
    tool_errors: int
    input_tokens: int
    output_tokens: int
    stopped_reason: str
    status: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tool_error_rate(self) -> float:
        return self.tool_errors / self.tool_calls if self.tool_calls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "duration_ms": round(self.duration_ms, 3),
            "n_steps": self.n_steps,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_error_rate": round(self.tool_error_rate, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "stopped_reason": self.stopped_reason,
            "status": self.status,
        }


class TraceRecorder:
    """一次 Agent run 的 span 树记录器。

    典型用法::

        rec = TraceRecorder(agent_name="maldi-ms-agent")
        root = rec.begin_root(gen_ai_request_model=...)
        with rec.child("chat glm-4-flash", "LLM") as sp:
            ...
        rec.finish_root("OK", **{"app.run.n_steps": 3})
        rec.write_jsonl(Path("reports/trace.jsonl"))
    """

    def __init__(
        self,
        agent_name: str = "maldi-ms-agent",
        capture_content: bool = False,
        run_id: Optional[str] = None,
        service_name: str = SCOPE_NAME,
    ) -> None:
        self.trace_id = _new_id(32)
        self.run_id = run_id or _new_id()
        self.agent_name = agent_name
        self.service_name = service_name
        self.capture_content = bool(capture_content)
        self.spans: List[Span] = []
        self._stack: List[Span] = []
        self._root: Optional[Span] = None

    # ------------------------------------------------------------ 埋点 API

    def begin(self, name: str, kind: str, **attributes: Any) -> Span:
        parent = self._stack[-1] if self._stack else None
        span = Span(
            trace_id=self.trace_id,
            span_id=_new_id(),
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_ns=time.time_ns(),
            attributes=dict(attributes),
        )
        span.attributes["openinference.span.kind"] = kind
        span.attributes["app.run_id"] = self.run_id
        self.spans.append(span)
        self._stack.append(span)
        return span

    def begin_root(self, **attributes: Any) -> Span:
        """开始顶层 ``invoke_agent`` span。一次 run 只应调用一次。"""
        if self._root is not None:
            raise RuntimeError("begin_root 只能调用一次")
        root = self.begin("invoke_agent", "AGENT", **attributes)
        root.attributes.setdefault("gen_ai.operation.name", "invoke_agent")
        root.attributes.setdefault("gen_ai.agent.name", self.agent_name)
        self._root = root
        return root

    @contextmanager
    def child(self, name: str, kind: str, **attributes: Any) -> Iterator[Span]:
        """子 span 上下文；异常自动标 ERROR 并重新抛出，不会吞掉故障。"""
        span = self.begin(name, kind, **attributes)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            span.finish("ERROR", f"{type(exc).__name__}: {exc}")
            raise
        else:
            # finish() 只在状态仍为 UNSET 时才写状态，但**总会**写 end_ns；
            # 因此显式标过 ERROR 的 span 也必须走这里，否则会留下没有结束时间的 span。
            span.finish("OK")
        finally:
            if self._stack and self._stack[-1] is span:
                self._stack.pop()

    def content(self, span: Span, key: str, value: Any) -> None:
        """显式内容采集。**默认关闭**：关掉时调用是 no-op。

        这是刻意设计——内容里可能有未发表的谱数据/患者相关信息，
        可观测性不能成为数据出域的旁路。
        """
        if self.capture_content:
            span.attributes[key] = value

    def finish_root(self, status: str = "OK", **attributes: Any) -> None:
        if self._root is None:
            return
        self._root.attributes.update(attributes)
        self._root.finish(status)
        if self._stack and self._stack[-1] is self._root:
            self._stack.pop()

    # ------------------------------------------------------------ 汇总与导出

    def summary(self, n_steps: int = 0, stopped_reason: str = "") -> TraceSummary:
        llm_spans = [s for s in self.spans if s.kind == "LLM"]
        tool_spans = [s for s in self.spans if s.kind == "TOOL"]
        root = self._root
        duration = root.duration_ms if root else 0.0
        return TraceSummary(
            run_id=self.run_id,
            trace_id=self.trace_id,
            duration_ms=duration,
            n_steps=n_steps,
            llm_calls=len(llm_spans),
            tool_calls=len(tool_spans),
            tool_errors=sum(s.status == "ERROR" for s in tool_spans),
            input_tokens=sum(
                int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in llm_spans
            ),
            output_tokens=sum(
                int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in llm_spans
            ),
            stopped_reason=stopped_reason,
            status=root.status if root else "UNSET",
        )

    def to_otlp_json(self) -> Dict[str, Any]:
        """完整 OTLP/JSON 结构，可直接 POST 到任意 OTLP/HTTP collector。"""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "app.semconv", "value": {"stringValue": SEMCONV_NOTE}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": SCOPE_NAME, "version": "0.3.0"},
                            "spans": [s.to_otlp() for s in self.spans],
                        }
                    ],
                }
            ]
        }

    def write_jsonl(self, path: str | Path) -> Path:
        """一行一个 run，便于按 run_id 聚合、按天切分、接入日志管道。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "agent-trace/v1",
            "summary": self.summary().to_dict(),
            "otlp": self.to_otlp_json(),
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return p

    @property
    def content_attributes(self) -> List[str]:
        """当前 trace 里所有「内容型」属性键，供合规自检使用。"""
        return sorted(
            {
                key
                for s in self.spans
                for key in s.attributes
                if any(key.startswith(p) for p in CONTENT_ATTRIBUTE_PREFIXES)
            }
        )


__all__ = [
    "Span",
    "TraceRecorder",
    "TraceSummary",
    "SEMCONV_NOTE",
    "CONTENT_ATTRIBUTE_PREFIXES",
]
