"""规则型 Agent 测试。

它存在的意义是**给评测框架做区分力自检**：如果四组消融跑出来分数一样，
说明这套评测根本测不出东西。所以这些测试断言的不是"Agent 有多聪明"，
而是"框架能不能区分出差别"。
"""

from __future__ import annotations

import pytest

from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.rule_agent import RuleBasedLLM
from maldi_ms_agent.tasks import TASKS, run_task

TASK_NULL = next(t for t in TASKS if t.id == "T1-null")


@pytest.mark.slow
def test_full_constraints_make_it_check(base_cfg):
    """A 组：提示词 + 工具警示 + 有工具 → 会做置换检验，且判对。"""
    tr = run_task(TASK_NULL, base_cfg, RuleBasedLLM(n_perm=30), max_steps=14)
    assert tr.grade.called_permutation
    assert tr.grade.correct, tr.grade.failure_mode


@pytest.mark.slow
def test_prompt_alone_is_enough(base_cfg):
    """B 组：关掉工具层警示，提示词还在 → 仍然会做（提示词能兜住）。"""
    tr = run_task(
        TASK_NULL, base_cfg, RuleBasedLLM(n_perm=30), max_steps=14, honest=False
    )
    assert tr.grade.called_permutation
    assert tr.grade.correct, tr.grade.failure_mode


@pytest.mark.slow
def test_tool_warning_alone_is_enough(base_cfg):
    """D 组：提示词不提醒，工具层警示还在 → 仍然会做（工具契约能兜住）。"""
    tr = run_task(
        TASK_NULL, base_cfg, RuleBasedLLM(n_perm=30), max_steps=14, strict=False
    )
    assert tr.grade.called_permutation
    assert tr.grade.correct, tr.grade.failure_mode


@pytest.mark.slow
def test_without_the_tool_it_overclaims(base_cfg):
    """C 组：工具箱里没有置换检验 → 想做也做不了，直接虚报。"""
    tr = run_task(
        TASK_NULL, base_cfg, RuleBasedLLM(n_perm=30),
        max_steps=14, with_permutation=False,
    )
    assert not tr.grade.called_permutation
    assert not tr.grade.correct
    assert "虚报" in tr.grade.failure_mode


def test_rule_agent_is_stateless_across_runs():
    """每个任务必须用**新的**规则 Agent 实例——它有状态，复用会串场。"""
    a, b = RuleBasedLLM(n_perm=10), RuleBasedLLM(n_perm=10)
    assert a.i == b.i == 0
    assert a.complete([{"role": "system", "content": "x"}]) == b.complete(
        [{"role": "system", "content": "x"}]
    )


def test_scripted_llm_ignores_context():
    """反例：固定脚本**不读上下文**，所以四组消融跑出来一模一样——
    这正是需要 RuleBasedLLM 的原因（评测框架必须先自证有区分力）。"""
    strict_sys = {"role": "system", "content": "必须先调用 permutation_test"}
    loose_sys = {"role": "system", "content": "按 AUC 判断即可"}
    a = ScriptedLLM(["Thought: x\nAction: finish(conclusion=结论 A)"])
    b = ScriptedLLM(["Thought: x\nAction: finish(conclusion=结论 A)"])
    assert a.complete([strict_sys]) == b.complete([loose_sys])
