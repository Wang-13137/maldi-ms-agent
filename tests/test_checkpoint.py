"""M9：断点续跑检查点。

核心断言：
- 每次 run 都附带可序列化的 Checkpoint；
- 续跑从断点恢复、不重放已完成步骤；
- 预算熔断后的 run 能用更大预算续跑至完成；
- 已 finish 的检查点续跑是幂等 no-op。
"""

from __future__ import annotations

import copy

from maldi_ms_agent.budget import Budget
from maldi_ms_agent.checkpoint import Checkpoint
from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.react import run_react
from maldi_ms_agent.session import AnalysisSession

SCRIPT = (
    "Thought: 先设置预处理。\nAction: preprocess(normalize=tic)",
    "Thought: 再做峰检测。\nAction: detect_peaks()",
    "Thought: 收尾。\nAction: finish(conclusion=证据不足，不能声称有判别能力)",
)


def _session(base_cfg, synth_ds) -> AnalysisSession:
    s = AnalysisSession(cfg=copy.deepcopy(base_cfg))
    s.attach_dataset(synth_ds)
    return s


# ---------------------------------------------------------------- 附着与序列化


def test_run_attaches_checkpoint(base_cfg, synth_ds):
    run = run_react(_session(base_cfg, synth_ds), ScriptedLLM(list(SCRIPT)), max_steps=8)
    assert run.stopped_reason == "finish"
    assert isinstance(run.checkpoint, Checkpoint)
    assert run.checkpoint.to_dict()["schema"] == "maldi-ms-agent/checkpoint@1"
    assert len(run.checkpoint.steps) == run.n_steps == 3
    assert len(run.checkpoint.messages) >= 2  # system + user 至少两条


def test_checkpoint_json_roundtrip(tmp_path, base_cfg, synth_ds):
    run = run_react(_session(base_cfg, synth_ds), ScriptedLLM(list(SCRIPT)), max_steps=8)
    path = tmp_path / "ckpt.json"
    run.checkpoint.save(str(path))
    loaded = Checkpoint.load(str(path))
    assert loaded.to_dict() == run.checkpoint.to_dict()
    # 序列化后仍能驱动一次续跑
    assert loaded.finished is True
    assert loaded.steps[0]["action"] == "preprocess"


# ---------------------------------------------------------------- 续跑语义


def test_resume_continues_after_max_steps(base_cfg, synth_ds):
    session = _session(base_cfg, synth_ds)
    llm = ScriptedLLM(list(SCRIPT))
    run1 = run_react(session, llm, max_steps=2)  # 不够步数，停在 max_steps
    assert run1.stopped_reason == "max_steps"
    assert run1.n_steps == 2

    run2 = run_react(session, llm, max_steps=8, checkpoint=run1.checkpoint)
    assert run2.finished is True
    assert run2.n_steps == 3
    # 不重放：前两步是恢复的，动作与原文一致
    assert [s.action for s in run2.steps[:2]] == ["preprocess", "detect_peaks"]
    # 工具调用累计正确，finish 追加在末尾
    assert run2.tool_calls == ["preprocess", "detect_peaks", "finish"]


def test_resume_after_budget_exceeded(base_cfg, synth_ds):
    session = _session(base_cfg, synth_ds)
    llm = ScriptedLLM(list(SCRIPT))
    run1 = run_react(session, llm, max_steps=8, budget=Budget(max_steps=2))
    assert run1.stopped_reason == "budget_exceeded"
    assert run1.n_steps == 2  # 实际执行 2 步
    # 账本计"进入循环的步数"（含被挡下的第 3 步），这是 M8 既有语义
    assert run1.budget_state.steps == 3

    run2 = run_react(
        session, llm, max_steps=8,
        budget=Budget(max_steps=8), checkpoint=run1.checkpoint,
    )
    assert run2.finished is True
    assert run2.n_steps == 3
    # 预算账本从断点恢复并继续累计（恢复 3 + 续跑 1）
    assert run2.budget_state is not None
    assert run2.budget_state.steps == 4


def test_resume_finished_is_idempotent(base_cfg, synth_ds):
    session = _session(base_cfg, synth_ds)
    llm = ScriptedLLM(list(SCRIPT))
    run1 = run_react(session, llm, max_steps=8)
    assert run1.finished is True
    calls_before = llm.n_calls

    run2 = run_react(session, llm, max_steps=8, checkpoint=run1.checkpoint)
    # 幂等：不发起任何 LLM / 工具调用，步数不变
    assert run2.finished is True
    assert run2.n_steps == run1.n_steps
    assert llm.n_calls == calls_before
    assert run2.tool_calls == run1.tool_calls


def test_resume_from_loaded_file_is_consistent(tmp_path, base_cfg, synth_ds):
    session = _session(base_cfg, synth_ds)
    llm = ScriptedLLM(list(SCRIPT))
    run1 = run_react(session, llm, max_steps=2)
    path = tmp_path / "ckpt.json"
    run1.checkpoint.save(str(path))

    # 续跑需复用同一个 LLM 实例（脚本型后端按调用次数续播）
    loaded = Checkpoint.load(str(path))
    run2 = run_react(_session(base_cfg, synth_ds), llm, max_steps=8, checkpoint=loaded)
    assert run2.finished is True
    assert run2.n_steps == 3
    assert [s.action for s in run2.steps[:2]] == ["preprocess", "detect_peaks"]
