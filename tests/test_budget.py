"""M8-b：运行时预算硬熔断的行为与兼容性。"""

from __future__ import annotations

import copy
import json

import pytest

from maldi_ms_agent.budget import Budget, BudgetState
from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.react import run_react
from maldi_ms_agent.session import AnalysisSession

EVAL_KEYS = {
    "task", "conclusion", "stopped_reason", "backend", "ablation",
    "tool_calls", "n_steps", "finished", "error", "steps",
}

SCRIPT = (
    "Thought: 先设置预处理。\nAction: preprocess(normalize=tic)",
    "Thought: 再做峰检测。\nAction: detect_peaks()",
    "Thought: 收尾。\nAction: finish(conclusion=证据不足，不能声称有判别能力)",
)


def _session(base_cfg, synth_ds) -> AnalysisSession:
    s = AnalysisSession(cfg=copy.deepcopy(base_cfg))
    s.attach_dataset(synth_ds)
    return s


def _run(base_cfg, synth_ds, budget=None, clock=None):
    return run_react(
        _session(base_cfg, synth_ds),
        ScriptedLLM(list(SCRIPT)),
        max_steps=8,
        budget=budget,
        clock=clock,
    )


class FakeClock:
    """每调用一次前进 1 秒，起点为 0。"""

    def __init__(self):
        self.t = -1.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


# ---------------------------------------------------------------- 账本


def test_budget_state_check_is_idempotent():
    st = BudgetState(Budget(max_total_tokens=5))
    st.add_usage(type("U", (), {"input_tokens": 6, "output_tokens": 0})())
    first = st.check()
    assert "token 预算超出：6 > 5" == first
    assert st.check() == first
    assert st.snapshot()["exceeded"] is True


def test_budget_state_snapshot_has_limits_and_usage():
    st = BudgetState(Budget(max_tool_calls=2, max_wall_seconds=9.0))
    st.tick_step()
    st.tick_tool()
    snap = st.snapshot()
    assert snap["limit_tool_calls"] == 2
    assert snap["used_tool_calls"] == 1
    assert snap["used_steps"] == 1
    assert snap["exceeded"] is False
    assert st.check() == ""


# ---------------------------------------------------------------- 熔断行为


def test_token_budget_fuses_without_claim(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds, budget=Budget(max_total_tokens=1))
    assert run.stopped_reason == "budget_exceeded"
    assert run.conclusion == ""
    assert "token 预算超出" in run.error
    assert run.budget_state.total_tokens > 1
    root = run.recorder.spans[0]
    assert root.attributes["app.budget.exceeded"] is True
    assert root.attributes["app.budget.limit.max_total_tokens"] == 1


def test_tool_call_budget_fuses(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds, budget=Budget(max_tool_calls=1))
    assert run.stopped_reason == "budget_exceeded"
    assert "工具调用预算超出" in run.error
    assert run.budget_state.tool_calls == 2


def test_wall_clock_budget_fuses_with_injected_clock(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds,
               budget=Budget(max_wall_seconds=2), clock=FakeClock())
    assert run.stopped_reason == "budget_exceeded"
    assert "墙钟时间预算超出" in run.error
    assert run.n_steps == 1  # 第一步跑完后时间超支，未发起第二次 LLM 调用


def test_step_budget_fuses_before_first_call(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds, budget=Budget(max_steps=0))
    assert run.stopped_reason == "budget_exceeded"
    assert "步数预算超出" in run.error
    assert run.n_steps == 0
    assert run.trace.llm_calls == 0


# ---------------------------------------------------------------- 兼容性


def test_budget_none_keeps_legacy_behavior(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds)
    assert run.stopped_reason == "finish"
    assert "证据不足" in run.conclusion
    assert run.budget_state is None
    assert run.recorder.spans[0].attributes["app.budget.hard_enabled"] is False


def test_to_dict_schema_frozen_with_budget(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds, budget=Budget(max_total_tokens=1))
    assert set(run.to_dict().keys()) == EVAL_KEYS


def test_budget_attributes_are_otlp_exportable(base_cfg, synth_ds):
    run = _run(base_cfg, synth_ds, budget=Budget(max_total_tokens=1))
    doc = run.recorder.to_otlp_json()
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    keys = {a["key"] for s in spans for a in s["attributes"]}
    assert "app.budget.exceeded" in keys
    assert "app.budget.used_total_tokens" in keys
    json.dumps(doc, ensure_ascii=False, default=str)
