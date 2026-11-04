"""MCP 工具服务（M8）：把质谱判别流水线暴露成标准工具协议。

与 ``medical-rag`` W6 的无状态 MCP 服务不同，这里的工具是**有状态流水线**：
``preprocess → detect_peaks → align_peaks → build_features → differential_test
→ evaluate_panel → permutation_test → finish`` 共享一份 ``AnalysisSession``。
因此服务化时补了三件企业级必需的事：

1. **会话隔离**：每个 ``sessionId`` 一份独立 ``AnalysisSession`` 与工具集，
   不同客户端不会串数据；用 ``OrderedDict`` 做 LRU，超过 ``max_sessions`` 淘汰最旧会话。
2. **最小权限**：默认只暴露白名单内的 9 个分析工具；``load_dataset`` 不在默认白名单，
   服务启动时绑定合成数据集，客户端无法借协议让服务读取任意本地目录（路径越权）。
3. **默认脱敏**：工具本来就只返回汇总统计文本；``structuredContent`` 在出进程前
   再过一遍 ``sanitize_payload``，长数组 / 矩阵裁成摘要，原始谱与特征矩阵绝不出域。

实现的是 MCP 的 stdio 传输 + JSON-RPC 2.0 帧，支持
``initialize / notifications/initialized / ping / tools/list / tools/call``。
手写而不引 SDK：依赖面小、协议帧可解释、面试能讲清每一帧，且与本项目
「能手写就手写，再用框架做对照」的做法一致。

用法::

    python scripts/run_mcp_server.py                       # stdio 常驻服务
    python scripts/run_mcp_server.py --once '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

合规：默认绑定 ``synth.py`` 合成数据；真实 37 例谱与特征矩阵不随仓分发、
不从本服务外发，也不进入任何外部 LLM 请求体。
"""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .react import call_tool
from .session import AnalysisSession
from .tools import Tool, build_tools

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS = (PROTOCOL_VERSION, "2024-11-05")
SERVER_INFO = {"name": "maldi-ms-agent", "version": "0.4.0"}

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: 默认暴露的分析工具。刻意不含 ``load_dataset``——服务已绑定数据集，
#: 客户端不该有能力指定任意本地路径。
DEFAULT_EXPOSED_TOOLS = (
    "preprocess",
    "detect_peaks",
    "align_peaks",
    "build_features",
    "differential_test",
    "evaluate_panel",
    "permutation_test",
    "batch_effect_check",
    "finish",
)

#: 工具入参 schema。参数统一按字符串收，交给 ``call_tool`` 做类型强制，
#: 这样「协议层宽松、工具边界严格」的分工与进程内调用完全一致。
_TOOL_PARAMS: Dict[str, Dict[str, str]] = {
    "load_dataset": {"data_dir": "数据目录路径"},
    "preprocess": {"normalize": "none|tic", "baseline": "true|false"},
    "detect_peaks": {"snr": "信噪比阈值", "min_distance": "最小峰间距(Da)"},
    "align_peaks": {"ppm_tolerance": "对齐容差(ppm)"},
    "build_features": {"detection_rate": "检出率阈值(0~1)"},
    "differential_test": {"alpha": "显著性水平"},
    "evaluate_panel": {"sizes": "候选特征数，如 3,5,10,20"},
    "permutation_test": {"n_perm": "置换次数", "seed": "随机种子"},
    "batch_effect_check": {"lo_mz": "波段下界", "hi_mz": "波段上界"},
    "finish": {"conclusion": "最终结论文本"},
}

REDACTION_NOTE = (
    "只导出汇总标量；长数组/矩阵裁为摘要，prompt、原始谱与特征矩阵不出域"
)


# ---------------------------------------------------------------- 脱敏


def sanitize_payload(value: Any, *, max_list: int = 16, max_str: int = 500,
                     max_keys: int = 50, _depth: int = 0) -> Any:
    """把工具 payload 裁成可安全外发的汇总数据。

    - 标量原样保留（float 的 nan/inf 转字符串，避免产出非法 JSON）；
    - list / tuple / ndarray 只保留前 ``max_list`` 项并标注总长度；
    - dict 最多 ``max_keys`` 个键；
    - 其它对象一律降级成类型名，绝不序列化内部状态（如 DataFrame）。
    """
    if _depth > 6:
        return "<depth-limit>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else str(f)
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + "…"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        n = len(value)
        out = [sanitize_payload(v, max_list=max_list, max_str=max_str,
                                max_keys=max_keys, _depth=_depth + 1)
               for v in list(value)[:max_list]]
        if n > max_list:
            out.append(f"…<truncated {n - max_list} items, total={n}>")
        return out
    if isinstance(value, dict):
        items = list(value.items())[:max_keys]
        out = {
            str(k)[:120]: sanitize_payload(v, max_list=max_list, max_str=max_str,
                                           max_keys=max_keys, _depth=_depth + 1)
            for k, v in items
        }
        if len(value) > max_keys:
            out["__truncated_keys__"] = len(value) - max_keys
        return out
    return f"<{type(value).__name__}>"


def _stringify_arguments(arguments: Dict[str, Any]) -> Dict[str, str]:
    """协议层传来的 JSON 值统一转字符串，交给 ``call_tool`` 按注解强制类型。"""
    out: Dict[str, str] = {}
    for k, v in arguments.items():
        if v is None:
            out[str(k)] = ""
        elif isinstance(v, bool):
            out[str(k)] = "true" if v else "false"
        elif isinstance(v, (dict, list)):
            out[str(k)] = json.dumps(v, ensure_ascii=False)
        else:
            out[str(k)] = str(v)
    return out


# ---------------------------------------------------------------- 数据结构


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class _SessionState:
    session: AnalysisSession
    tools: List[Tool] = field(default_factory=list)
    by_name: Dict[str, Tool] = field(default_factory=dict)


def _schema_for(name: str) -> Dict[str, Any]:
    props = {
        k: {"type": "string", "description": v}
        for k, v in _TOOL_PARAMS.get(name, {}).items()
    }
    return {"type": "object", "properties": props, "required": []}


# ---------------------------------------------------------------- 服务


class MCPToolServer:
    """有状态、会话隔离、默认脱敏的 MCP 工具服务。

    ``session_factory`` 每次调用必须返回一份**全新的** ``AnalysisSession``，
    服务不共享会话对象，这是隔离的前提。
    """

    def __init__(
        self,
        session_factory: Callable[[], AnalysisSession],
        allowed_tools: Optional[Sequence[str]] = None,
        server_info: Optional[Dict[str, str]] = None,
        max_sessions: int = 8,
        redact: bool = True,
    ) -> None:
        self.session_factory = session_factory
        self.allowed = tuple(allowed_tools) if allowed_tools is not None else DEFAULT_EXPOSED_TOOLS
        self.server_info = server_info or SERVER_INFO
        self.max_sessions = max(1, int(max_sessions))
        self.redact = bool(redact)
        self._sessions: "OrderedDict[str, _SessionState]" = OrderedDict()
        self.n_evicted = 0

    # ------------------------------------------------------------ 会话

    def _state(self, session_id: str) -> _SessionState:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        state = _SessionState(session=self.session_factory())
        tools = build_tools(state.session)
        state.by_name = {t.name: t for t in tools}
        state.tools = [t for t in tools if t.name in self.allowed]
        self._sessions[session_id] = state
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
            self.n_evicted += 1
        return state

    def sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------ 工具表

    def tools(self, session_id: str = "default") -> List[MCPTool]:
        state = self._state(session_id)

        def make_invoker(tool: Tool):
            def invoke(**kwargs: Any) -> Dict[str, Any]:
                try:
                    res, resolved = call_tool(tool, _stringify_arguments(kwargs))
                except Exception as exc:  # noqa: BLE001
                    return {"error": f"{type(exc).__name__}: {exc}"}
                payload = res.payload if self.redact else dict(res.payload)
                safe = sanitize_payload(payload) if self.redact else payload
                if not isinstance(safe, dict):
                    safe = {"value": safe}
                safe["__text__"] = str(res.text)[:2000]
                safe["__ok__"] = bool(res.ok)
                safe["__args__"] = sanitize_payload(resolved) if self.redact else resolved
                return safe

            return invoke

        return [
            MCPTool(
                name=t.name,
                description=t.description + f"｜{REDACTION_NOTE}",
                input_schema=_schema_for(t.name),
                func=make_invoker(t),
            )
            for t in state.tools
        ]

    # ------------------------------------------------------------ 协议帧

    def handle(self, raw: str) -> Optional[str]:
        """处理一帧请求，返回一帧响应字符串；通知类返回 ``None``。"""
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._reply_error(None, PARSE_ERROR, "JSON 解析失败")
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            return self._reply_error(None, INVALID_REQUEST, "不是合法的 JSON-RPC 2.0 请求")

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        if not isinstance(params, dict):
            return self._reply_error(req_id, INVALID_PARAMS, "params 必须是对象")
        is_notification = "id" not in req
        session_id = str(params.get("sessionId") or "default")

        try:
            if method == "initialize":
                client_version = params.get("protocolVersion") or PROTOCOL_VERSION
                version = client_version if client_version in SUPPORTED_VERSIONS else PROTOCOL_VERSION
                return self._reply(req_id, {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": self.server_info,
                    "_meta": {
                        "sessions": "isolated",
                        "redaction": REDACTION_NOTE,
                        "defaultExposedTools": list(self.allowed),
                    },
                })
            if method in ("notifications/initialized", "initialized"):
                return None
            if method == "ping":
                return self._reply(req_id, {})
            if method == "tools/list":
                return self._reply(req_id, {
                    "tools": [t.to_dict() for t in self.tools(session_id)]
                })
            if method == "tools/call":
                return self._reply(req_id, self._call_tool(params, session_id))
            return self._reply_error(req_id, METHOD_NOT_FOUND, f"不支持的方法：{method}")
        except Exception as exc:  # 工具内部异常不该把服务打死
            if is_notification:
                return None
            return self._reply_error(req_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _call_tool(self, params: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """工具级错误走 ``isError`` 结果帧，不升级成协议错误，连接不会断。"""
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        table = {t.name: t for t in self.tools(session_id)}
        if name not in table:
            return self._tool_error(
                f"未知或未授权工具：{name}；可用：{', '.join(sorted(table))}"
            )
        if not isinstance(arguments, dict):
            return self._tool_error("arguments 必须是对象")
        try:
            payload = table[name].func(**arguments)
        except TypeError as exc:
            return self._tool_error(f"参数不对：{exc}")
        except Exception as exc:  # noqa: BLE001
            return self._tool_error(f"{type(exc).__name__}: {exc}")
        is_error = bool(payload.get("error")) or not payload.get("__ok__", True)
        return {
            "content": [{"type": "text", "text": str(payload.get("__text__", ""))}],
            "structuredContent": payload,
            "isError": bool(is_error),
        }

    @staticmethod
    def _tool_error(message: str) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": message}],
            "structuredContent": {"error": message},
            "isError": True,
        }

    @staticmethod
    def _reply(req_id: Any, result: Dict[str, Any]) -> str:
        return json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "result": result},
            ensure_ascii=False, default=str,
        )

    @staticmethod
    def _reply_error(req_id: Any, code: int, message: str) -> str:
        return json.dumps(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )

    def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> None:
        """按行读 stdin、按行写 stdout；日志一律走 stderr，不污染协议流。"""
        inp = stdin if stdin is not None else sys.stdin
        out = stdout if stdout is not None else sys.stdout
        for line in inp:
            line = line.strip()
            if not line:
                continue
            reply = self.handle(line)
            if reply is None:
                continue
            out.write(reply + "\n")
            out.flush()


# ---------------------------------------------------------------- 默认装配


def _default_cfg() -> dict:
    """找 pipeline 的默认配置；找不到时退回合成配置，保证服务始终可启动。"""
    from .synth import synth_config

    here = Path(__file__).resolve()
    roots = [here.parents[3] / "maldi-ms-pipeline" / "configs"]
    import os
    if os.environ.get("MALDI_PIPELINE_SRC"):
        roots.insert(0, Path(os.environ["MALDI_PIPELINE_SRC"]).parent / "configs")
    for d in roots:
        for p in sorted(d.glob("*.yaml")):
            try:
                from maldi_ms.io import load_config
                return synth_config(load_config(str(p)))
            except Exception:  # noqa: BLE001
                continue
    return synth_config({})


def default_session_factory() -> AnalysisSession:
    """默认会话：绑定合成数据集，不触碰真实队列数据。"""
    from .synth import SynthConfig, make_synth_dataset

    s = AnalysisSession(cfg=_default_cfg())
    s.attach_dataset(make_synth_dataset(SynthConfig(
        n_case=8, n_ctrl=8, n_points=4000,
        n_signal_peaks=4, n_null_peaks=4, seed=42,
    )))
    return s


def build_mcp_server(
    session_factory: Optional[Callable[[], AnalysisSession]] = None,
    allowed_tools: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> MCPToolServer:
    return MCPToolServer(
        session_factory or default_session_factory,
        allowed_tools=allowed_tools,
        **kwargs,
    )


__all__ = [
    "MCPTool",
    "MCPToolServer",
    "PROTOCOL_VERSION",
    "DEFAULT_EXPOSED_TOOLS",
    "REDACTION_NOTE",
    "sanitize_payload",
    "build_mcp_server",
    "default_session_factory",
]
