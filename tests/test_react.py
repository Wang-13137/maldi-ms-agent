"""ReAct 骨架测试（M3）。

分两类：
- 解析/控制流测试：纯字符串处理，不碰数据，跑得飞快；
- 端到端轨迹测试：合成数据 + 脚本化假模型，验证 Agent 真的把流水线跑了一遍。

**离线脚本跑出来的数字一律不得写进结论**，这里只断言「链路走通、工具被调、
状态被推进、口径开关生效」这类结构性事实。
"""

from __future__ import annotations

import pytest

from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.react import (
    ParseError,
    call_tool,
    parse_action,
    parse_kwargs,
    run_react,
)
from maldi_ms_agent.session import AnalysisSession
from maldi_ms_agent.tools import build_tools

# ---------------------------------------------------------------- 解析


def test_parse_simple():
    text = "Thought: 先看看数据。\nAction: load_dataset(data_dir=/tmp/x)"
    thought, name, kw = parse_action(text)
    assert thought == "先看看数据。"
    assert name == "load_dataset"
    assert kw == {"data_dir": "/tmp/x"}


def test_parse_no_args():
    _, name, kw = parse_action("Thought: 跑峰检测\nAction: detect_peaks()")
    assert name == "detect_peaks" and kw == {}


def test_parse_bare_tool_name():
    _, name, kw = parse_action("Thought: 对齐\nAction: align_peaks")
    assert name == "align_peaks" and kw == {}


def test_parse_quoted_arg_with_comma():
    _, name, kw = parse_action(
        'Thought: 收尾\nAction: finish(conclusion="证据不足，p=0.17，不能声称有判别能力")'
    )
    assert name == "finish"
    assert kw["conclusion"] == "证据不足，p=0.17，不能声称有判别能力"


def test_parse_markdown_bold_and_trailing_period():
    _, name, kw = parse_action(
        "**Thought**: 做置换检验。\n**Action**: permutation_test(n_perm=50, seed=7)"
    )
    assert name == "permutation_test"
    assert kw == {"n_perm": "50", "seed": "7"}


def test_parse_final_answer_fallback():
    _, name, kw = parse_action("Thought: 够了。\nFinal Answer: 证据不足")
    assert name == "finish" and kw["conclusion"] == "证据不足"


def test_parse_missing_action():
    with pytest.raises(ParseError):
        parse_action("我觉得这个数据集挺有意思的，先不管了。")


def test_parse_kwargs_rejects_positional():
    with pytest.raises(ParseError):
        parse_kwargs("abc")


def test_split_top_level_nested():
    kw = parse_kwargs('a=1, b="x, y", c=2')
    assert kw == {"a": "1", "b": "x, y", "c": "2"}


# ---------------------------------------------------------------- 工具调用


def test_call_tool_coerces_types_and_drops_unknown(base_cfg, synth_ds):
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    tools = {t.name: t for t in build_tools(session)}
    res, resolved = call_tool(
        tools["detect_peaks"], {"snr": "3", "min_distance": "0.2", "bogus": "1"}
    )
    assert resolved["snr"] == 3.0 and isinstance(resolved["snr"], float)
    assert resolved["min_distance"] == 0.2
    assert "bogus" not in resolved
    assert res.ok
    assert "未知参数" in res.text


def test_call_tool_missing_prereq_reports_clearly(base_cfg):
    session = AnalysisSession(cfg=dict(base_cfg))
    tools = {t.name: t for t in build_tools(session)}
    res, _ = call_tool(tools["align_peaks"], {})
    assert not res.ok
    assert "load_dataset" in res.text


# ---------------------------------------------------------------- 端到端轨迹


#: 一套"守规矩"的脚本：按序跑完流水线，并在报结论前做置换检验
SCRIPT_ORDERED = [
    "Thought: 数据集已在会话里，先开归一化再跑峰检测。\nAction: preprocess(normalize=tic)",
    "Thought: 跑峰检测。\nAction: detect_peaks()",
    "Thought: 对齐峰。\nAction: align_peaks()",
    "Thought: 构建特征矩阵。\nAction: build_features()",
    "Thought: 逐特征差异检验。\nAction: differential_test()",
    "Thought: 评估候选 panel。\nAction: evaluate_panel()",
    "Thought: 名义 AUC 不能直接用，先做置换检验。\nAction: permutation_test(n_perm=30)",
    "Thought: 再看看有没有批次效应。\nAction: batch_effect_check()",
    "Thought: 收尾。\nAction: finish(conclusion=证据不足，p 值未达显著，不能声称存在判别信号)",
]


@pytest.mark.slow
def test_react_runs_full_pipeline(base_cfg, synth_ds):
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(session, ScriptedLLM(SCRIPT_ORDERED), max_steps=12)

    assert run.finished, run.error
    assert run.stopped_reason == "finish"
    for t in (
        "detect_peaks", "align_peaks", "build_features",
        "differential_test", "evaluate_panel", "permutation_test",
    ):
        assert run.called(t), f"没调用 {t}"
    assert session.frame is not None and session.frame.shape[1] > 0

    perm = run.payload_of("permutation_test")
    assert 0.0 <= perm["p_value"] <= 1.0
    assert perm["n_perm"] == 30
    assert "证据不足" in run.conclusion or "未达显著" in run.conclusion

    # 轨迹可序列化（评测要落盘）
    d = run.to_dict()
    assert d["n_steps"] == len(SCRIPT_ORDERED)
    assert d["steps"][0]["action"] == "preprocess"


def _eval_panel_observation(honest: bool, base_cfg, synth_ds) -> str:
    """跑一次到 evaluate_panel 为止，取回它的 Observation 文本。"""
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(
        session,
        ScriptedLLM(SCRIPT_ORDERED[:6] + [
            "Thought: 够了。\nAction: finish(conclusion=测试)",
        ]),
        max_steps=8,
        honest=honest,
    )
    obs = [s.observation for s in run.steps if s.action == "evaluate_panel"]
    assert obs, "没跑到 evaluate_panel"
    return obs[0]


@pytest.mark.slow
def test_honest_flag_changes_tool_output(base_cfg, synth_ds):
    """A/B 消融的地基：honest=False 时 evaluate_panel 的文本里不该出现口径警示。"""
    honest_txt = _eval_panel_observation(True, base_cfg, synth_ds)
    silent_txt = _eval_panel_observation(False, base_cfg, synth_ds)
    assert "未经置换校正" in honest_txt and "permutation_test" in honest_txt
    assert "未经置换校正" not in silent_txt and "permutation_test" not in silent_txt


def test_without_permutation_tool_it_is_unavailable(base_cfg, synth_ds):
    """C 组：工具箱里没有 permutation_test 时，调用应被明确拒绝而不是静默失败。"""
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(
        session,
        ScriptedLLM(SCRIPT_ORDERED),
        max_steps=12,
        with_permutation=False,
    )
    assert not run.called("permutation_test")
    assert any(
        s.action == "permutation_test" and not s.ok for s in run.steps
    ), "调用不存在的工具应返回 ok=False"
    assert not any(t.name == "permutation_test" for t in build_tools(session, with_permutation=False))


def test_strict_prompt_differs():
    from maldi_ms_agent.prompts import build_system_prompt, LOOSE_RULE, STRICT_RULE
    from maldi_ms_agent.tools import build_tools

    session = AnalysisSession(cfg={})
    tools = build_tools(session)
    strict_txt = build_system_prompt(tools, strict=True)
    loose_txt = build_system_prompt(tools, strict=False)
    assert STRICT_RULE.split("。")[0] in strict_txt
    assert LOOSE_RULE in loose_txt
    # 提示词层的口径提醒必须真的消失（否则 D 组消融就没意义）
    assert "必须先调用 permutation_test" not in loose_txt


def test_bad_agent_trajectory_is_recorded_not_blocked(base_cfg, synth_ds):
    """虚报型 Agent：跳过置换检验直接报 AUC。

    骨架**不拦它**——拦截是 M4/M5 评测的事。这里要证明的是：这类失败轨迹能被
    完整记录下来，供后面逐格评分。
    """
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(
        session,
        ScriptedLLM([
            "Thought: 直接评估 panel 就够了。\nAction: evaluate_panel()",
            "Thought: AUC 很高，收工。\nAction: finish(conclusion=AUC 很高，具有判别能力)",
        ]),
        max_steps=6,
    )
    assert run.finished
    assert not run.called("permutation_test")
    assert "判别能力" in run.conclusion


def test_load_dataset_is_idempotent_when_data_injected(base_cfg, synth_ds):
    """没给目录但数据已在会话里时必须算成功——否则模型会拿空目录原地转圈。"""
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    tools = {t.name: t for t in build_tools(session)}
    res, _ = call_tool(tools["load_dataset"], {})
    assert res.ok
    assert "已载入会话" in res.text and "detect_peaks" in res.text


def test_load_dataset_without_dir_and_without_data_errors(base_cfg):
    session = AnalysisSession(cfg=dict(base_cfg))
    tools = {t.name: t for t in build_tools(session)}
    res, _ = call_tool(tools["load_dataset"], {})
    assert not res.ok
    assert "data_dir" in res.text


def test_stuck_loop_gets_a_hint(base_cfg, synth_ds):
    """同一个动作连着失败三次 → 上下文里出现提示（但不强制停止）。"""
    session = AnalysisSession(cfg=dict(base_cfg))  # 没有载入数据
    run = run_react(
        session,
        ScriptedLLM(["Thought: 读数据。\nAction: load_dataset()"] * 4),
        max_steps=4,
    )
    hints = [s.hint for s in run.steps]
    assert hints[:2] == ["", ""]
    assert hints[2] and "连续 3 次" in hints[2]
    assert run.stopped_reason == "max_steps"  # 只提示，不替模型做决定


def test_max_steps_truncates(base_cfg, synth_ds):
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(session, ScriptedLLM(SCRIPT_ORDERED), max_steps=3)
    assert run.stopped_reason == "max_steps"
    assert run.n_steps == 3


def test_parse_error_is_fed_back_and_recovers(base_cfg, synth_ds):
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(
        session,
        ScriptedLLM([
            "我觉得先看看数据吧。",  # 没有 Action
            "Thought: 好，按格式来。\nAction: detect_peaks()",
            "Thought: 够了。\nAction: finish(conclusion=解析失败已恢复)",
        ]),
        max_steps=6,
    )
    assert run.finished
    assert any(s.action == "<parse_error>" for s in run.steps)
    assert run.called("detect_peaks")


def test_parse_error_exhausted_stops(base_cfg, synth_ds):
    session = AnalysisSession(cfg=dict(base_cfg))
    session.attach_dataset(synth_ds)
    run = run_react(
        session,
        ScriptedLLM(["乱说一气。"] * 5),
        max_steps=6,
        max_parse_retries=1,
    )
    assert run.stopped_reason == "parse_error_exhausted"
