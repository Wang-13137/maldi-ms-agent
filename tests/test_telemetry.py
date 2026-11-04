"""可观测层测试：span 树、OTLP 导出、默认脱敏、与 ReAct 的集成。

重点验证两件容易写错的事：
1. 显式标过 ERROR 的 span 也必须写 end_ns（否则导出会出现没有结束时间的 span）；
2. 埋点绝不能改变 M5 评测 JSON 的结构（四组消融要逐格比对）。
"""

from __future__ import annotations

import json

import pytest

from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.react import run_react
from maldi_ms_agent.session import new_session
from maldi_ms_agent.telemetry import (
    CONTENT_ATTRIBUTE_PREFIXES,
    Span,
    TraceRecorder,
)


def test_span_tree_has_single_root_and_parent_links():
    rec = TraceRecorder()
    root = rec.begin_root(**{"gen_ai.agent.name": "maldi-ms-agent"})
    with rec.child("chat glm-4-flash", "LLM") as llm_span:
        llm_span.attributes["gen_ai.usage.input_tokens"] = 120
    with rec.child("execute_tool permutation_test", "TOOL") as tool_span:
        tool_span.set_status("ERROR", "tool failed")
    rec.finish_root("OK")

    kinds = [s.kind for s in rec.spans]
    assert kinds.count("AGENT") == 1
    assert kinds.count("LLM") == 1 and kinds.count("TOOL") == 1
    assert all(s.parent_span_id == root.span_id for s in rec.spans if s is not root)
    assert all(s.trace_id == rec.trace_id for s in rec.spans)
    assert all(s.attributes["app.run_id"] == rec.run_id for s in rec.spans)


def test_child_exception_marks_error_without_swallowing():
    rec = TraceRecorder()
    rec.begin_root()
    with pytest.raises(ValueError, match="boom"):
        with rec.child("execute_tool x", "TOOL"):
            raise ValueError("boom")
    rec.finish_root("OK")

    tool = [s for s in rec.spans if s.kind == "TOOL"][0]
    assert tool.status == "ERROR"
    assert "ValueError" in tool.status_message
    assert tool.end_ns > 0


def test_explicit_error_status_still_gets_end_time():
    """回归：以前 ERROR span 会漏写 end_ns，导出后出现幽灵 span。"""
    rec = TraceRecorder()
    rec.begin_root()
    with rec.child("execute_tool x", "TOOL") as sp:
        sp.set_status("ERROR", "bad args")
    rec.finish_root("OK")

    tool = [s for s in rec.spans if s.kind == "TOOL"][0]
    assert tool.status == "ERROR" and tool.status_message == "bad args"
    assert tool.end_ns >= tool.start_ns
    assert tool.duration_ms >= 0


def test_default_redaction_drops_all_content():
    rec = TraceRecorder()
    rec.begin_root()
    with rec.child("chat m", "LLM") as sp:
        rec.content(sp, "gen_ai.input.messages", "[系统提示词，可能含真实谱信息]")
        rec.content(sp, "gen_ai.output.messages", "[模型回复]")
    rec.finish_root("OK")

    assert rec.capture_content is False
    assert rec.content_attributes == []
    assert all(
        not any(k.startswith(p) for p in CONTENT_ATTRIBUTE_PREFIXES)
        for s in rec.spans
        for k in s.attributes
    )


def test_content_capture_requires_explicit_opt_in():
    rec = TraceRecorder(capture_content=True)
    rec.begin_root()
    with rec.child("chat m", "LLM") as sp:
        rec.content(sp, "gen_ai.input.messages", "hello")
    rec.finish_root("OK")

    assert "gen_ai.input.messages" in rec.content_attributes
    assert rec.capture_content is True


def test_otlp_export_shape_and_attribute_types():
    rec = TraceRecorder(service_name="test-service")
    root = rec.begin_root()
    with rec.child("chat glm-4-flash", "LLM") as sp:
        sp.attributes.update({
            "gen_ai.request.model": "glm-4-flash",
            "gen_ai.usage.input_tokens": 321,
            "gen_ai.response.finish_reasons": ["stop"],
        })
    rec.finish_root("OK")

    doc = rec.to_otlp_json()
    resource = doc["resourceSpans"][0]["resource"]["attributes"]
    assert {"key": "service.name", "value": {"stringValue": "test-service"}} in resource
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 2
    assert spans[0]["name"] == "invoke_agent"
    assert spans[1]["parentSpanId"] == root.span_id

    attrs = {a["key"]: a["value"] for a in spans[1]["attributes"]}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "321"}
    assert attrs["gen_ai.response.finish_reasons"]["arrayValue"]["values"] == [
        {"stringValue": "stop"}
    ]


def test_summary_aggregates_tokens_and_tool_errors():
    rec = TraceRecorder()
    rec.begin_root()
    with rec.child("chat m", "LLM") as sp:
        sp.attributes.update({
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 40,
        })
    with rec.child("chat m", "LLM") as sp:
        sp.attributes.update({
            "gen_ai.usage.input_tokens": 200,
            "gen_ai.usage.output_tokens": 60,
        })
    with rec.child("execute_tool a", "TOOL") as sp:
        sp.set_status("ERROR", "boom")
    with rec.child("execute_tool b", "TOOL"):
        pass
    rec.finish_root("OK")

    s = rec.summary(n_steps=2, stopped_reason="finish")
    assert (s.input_tokens, s.output_tokens, s.total_tokens) == (300, 100, 400)
    assert (s.llm_calls, s.tool_calls, s.tool_errors) == (2, 2, 1)
    assert s.tool_error_rate == 0.5
    assert s.n_steps == 2 and s.stopped_reason == "finish"
    assert s.duration_ms >= 0
    assert s.to_dict()["total_tokens"] == 400


def test_write_jsonl_is_one_json_object_per_line(tmp_path):
    rec = TraceRecorder()
    rec.begin_root()
    rec.finish_root("OK")
    out = rec.write_jsonl(tmp_path / "traces" / "run.jsonl")
    rec.write_jsonl(out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        doc = json.loads(line)
        assert doc["schema"] == "agent-trace/v1"
        assert doc["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]


def test_run_react_emits_trace_without_changing_eval_json(base_cfg):
    """集成：默认开启埋点，且 to_dict() 字段集一格都不能变。"""
    session = new_session(base_cfg)
    run = run_react(
        session,
        ScriptedLLM(["Thought: 收尾。\nAction: finish(conclusion=证据不足)"]),
        max_steps=3,
    )
    assert run.finished
    assert run.trace is not None and run.recorder is not None
    assert run.trace.llm_calls == run.n_steps == 1
    assert run.trace.total_tokens > 0
    assert [s.kind for s in run.recorder.spans] == ["AGENT", "LLM", "TOOL"]

    assert set(run.to_dict().keys()) == {
        "task", "conclusion", "stopped_reason", "backend", "ablation",
        "tool_calls", "n_steps", "finished", "error", "steps",
    }
    assert "trace" not in run.to_dict()
    assert run.recorder.content_attributes == []


def test_span_rejects_unknown_status():
    sp = Span("a" * 32, "b" * 16, None, "invoke_agent", "AGENT", 1)
    with pytest.raises(ValueError, match="未知 span 状态"):
        sp.set_status("WAT")
