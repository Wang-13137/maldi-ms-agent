"""M8-a：MCP 工具服务的协议帧、最小权限、会话隔离与默认脱敏。"""

from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest

from maldi_ms_agent.mcp_server import (
    DEFAULT_EXPOSED_TOOLS,
    PROTOCOL_VERSION,
    MCPToolServer,
    build_mcp_server,
    sanitize_payload,
)
from maldi_ms_agent.session import AnalysisSession
from maldi_ms_agent.synth import SynthConfig, make_synth_dataset


@pytest.fixture
def factory(base_cfg):
    """每次调用返回一份全新的、绑定合成数据的会话。"""
    def make() -> AnalysisSession:
        s = AnalysisSession(cfg=copy.deepcopy(base_cfg))
        s.attach_dataset(make_synth_dataset(SynthConfig(
            n_case=6, n_ctrl=6, n_points=4000,
            n_signal_peaks=4, n_null_peaks=4, seed=7,
        )))
        return s
    return make


@pytest.fixture
def server(factory):
    return MCPToolServer(factory, max_sessions=3)


def frame(method, *, rid=1, params=None, notify=False):
    d = {"jsonrpc": "2.0", "method": method}
    if not notify:
        d["id"] = rid
    if params is not None:
        d["params"] = params
    return json.dumps(d, ensure_ascii=False)


def reply(server, raw):
    out = server.handle(raw)
    return None if out is None else json.loads(out)


# ---------------------------------------------------------------- 协议帧


def test_initialize_version_negotiation(server):
    r = reply(server, frame("initialize", params={"protocolVersion": "2024-11-05"}))
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert r["result"]["serverInfo"]["name"] == "maldi-ms-agent"
    assert r["result"]["_meta"]["sessions"] == "isolated"

    r2 = reply(server, frame("initialize", params={"protocolVersion": "1999-01-01"}))
    assert r2["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_tools_list_default_whitelist_and_schema(server):
    r = reply(server, frame("tools/list"))
    names = [t["name"] for t in r["result"]["tools"]]
    assert set(names) == set(DEFAULT_EXPOSED_TOOLS)
    assert "load_dataset" not in names  # 最小权限：不让客户端指定任意本地路径
    for t in r["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"
        assert isinstance(t["inputSchema"]["properties"], dict)
        assert "只导出汇总标量" in t["description"]


def test_allowed_tools_can_be_overridden(factory):
    s = MCPToolServer(factory, allowed_tools=("finish", "load_dataset"))
    names = [t["name"] for t in reply(s, frame("tools/list"))["result"]["tools"]]
    assert names == ["load_dataset", "finish"] or set(names) == {"load_dataset", "finish"}


def test_bad_json_and_unknown_method(server):
    assert reply(server, "{not json")["error"]["code"] == -32700
    assert reply(server, frame("no.such.method"))["error"]["code"] == -32601
    assert reply(server, frame("tools/list", params="bad"))["error"]["code"] == -32602


def test_notification_has_no_reply_and_ping_works(server):
    assert server.handle(frame("notifications/initialized", notify=True)) is None
    assert reply(server, frame("ping"))["result"] == {}


# ---------------------------------------------------------------- 工具调用


def test_unauthorized_tool_returns_isError_not_protocol_error(server):
    r = reply(server, frame("tools/call", params={
        "name": "load_dataset", "arguments": {"data_dir": "/etc"},
    }))
    assert "result" in r and r["result"]["isError"] is True
    assert "未授权" in r["result"]["content"][0]["text"]


def test_bad_arguments_return_isError(server):
    r = reply(server, frame("tools/call", params={
        "name": "finish", "arguments": "not-an-object",
    }))
    assert r["result"]["isError"] is True


def test_finish_roundtrip(server):
    r = reply(server, frame("tools/call", params={
        "name": "finish", "arguments": {"conclusion": "证据不足，不能声称具有判别能力"},
    }))
    assert r["result"]["isError"] is False
    assert "证据不足" in r["result"]["content"][0]["text"]
    assert r["result"]["structuredContent"]["__ok__"] is True


def test_tool_internal_error_does_not_kill_connection(server):
    # 类型强制失败应由工具边界转成 isError 结果帧，而不是打断 JSON-RPC 连接。
    r = reply(server, frame("tools/call", params={
        "name": "detect_peaks", "arguments": {"snr": "not-a-number"},
    }))
    assert r["result"]["isError"] is True
    assert "error" in r["result"]["structuredContent"]
    assert reply(server, frame("ping"))["result"] == {}


# ---------------------------------------------------------------- 会话隔离


def test_sessions_are_isolated(server):
    a, b = server._state("A").session, server._state("B").session
    assert a is not b
    reply(server, frame("tools/call", rid=2, params={
        "sessionId": "A", "name": "preprocess", "arguments": {"normalize": "tic"},
    }))
    assert a.log != b.log
    assert len(b.log) == 1 and b.log[0] == "attach_dataset"


def test_lru_eviction(server):
    for sid in ("A", "B", "C", "D"):
        server._state(sid)
    assert server.n_evicted == 1
    assert len(server.sessions()) == 3
    assert "A" not in server.sessions()


# ---------------------------------------------------------------- 默认脱敏


def test_sanitize_payload_cuts_long_arrays():
    raw = {
        "spectrum": np.arange(100000, dtype=float),
        "top10": [0.1, 0.2, 0.3],
        "n": 37,
        "frame": {"values": np.zeros((37, 47))},
    }
    out = sanitize_payload(raw)
    assert out["n"] == 37
    assert out["top10"] == [0.1, 0.2, 0.3]
    assert "truncated" in out["spectrum"][-1]
    assert len(out["spectrum"]) <= 17
    assert "truncated" in out["frame"]["values"][-1]


def test_sanitize_payload_drops_unknown_objects_and_nonfinite():
    class Secret:
        pass
    out = sanitize_payload({"obj": Secret(), "nan": float("nan"), "s": "x" * 900})
    assert out["obj"] == "<Secret>"
    assert out["nan"] == "nan"
    assert out["s"].endswith("…")


def test_structured_content_is_json_safe_and_bounded(server):
    r = reply(server, frame("tools/call", params={
        "name": "finish", "arguments": {"conclusion": "证据不足"},
    }))
    sc = r["result"]["structuredContent"]
    assert "__text__" in sc and "__ok__" in sc
    # 出进程的结构化内容必须能被标准 JSON 序列化（无 ndarray / DataFrame 残留）
    json.dumps(sc, ensure_ascii=False)
    assert len(sc["__text__"]) <= 2000


# ---------------------------------------------------------------- stdio


def test_serve_stdio_line_framing(server):
    inp = io.StringIO(
        frame("initialize", rid=1) + "\n"
        + frame("notifications/initialized", notify=True) + "\n"
        + frame("tools/list", rid=2) + "\n"
        + "\n"
    )
    out = io.StringIO()
    server.serve_stdio(stdin=inp, stdout=out)
    lines = [x for x in out.getvalue().strip().splitlines() if x]
    assert len(lines) == 2  # 通知不回包
    assert json.loads(lines[0])["id"] == 1
    assert "tools" in json.loads(lines[1])["result"]


def test_default_server_starts_without_real_data():
    s = build_mcp_server()
    names = [t["name"] for t in reply(s, frame("tools/list"))["result"]["tools"]]
    assert "load_dataset" not in names
    assert "permutation_test" in names
