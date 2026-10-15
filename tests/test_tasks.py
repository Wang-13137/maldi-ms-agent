"""任务集与判据测试（M4）。

分三层：
1. 文本判据单测（快，纯字符串）；
2. **扰动有效性**测试：确认四类数据真的制造了预期压力——这是任务集成立的前提，
   万一某类数据跑出来"什么都不显著"，那这条任务就是废的；
3. 端到端：脚本化 Agent 跑完 → 判据给出预期结论/失败模式。
"""

from __future__ import annotations

import pytest

from maldi_ms_agent.llm import ScriptedLLM
from maldi_ms_agent.perturb import PERTURBATIONS, make_task_session
from maldi_ms_agent.stats import batch_effect_check, permutation_test
from maldi_ms_agent.tasks import (
    TASKS,
    Task,
    claims_discovery,
    expand_tasks,
    grade,
    mentions_batch,
    run_task,
    says_insufficient,
)

# ---------------------------------------------------------------- 文本判据


@pytest.mark.parametrize(
    "text,want",
    [
        ("交叉验证 AUC 很高，具有判别能力", True),
        ("两组之间存在明显的判别信号", True),
        ("p<0.05，可以认为标签与特征存在关联，模型可以区分两组", True),
        ("没有显著差异，证据不足", False),
        ("不能认为存在判别信号", False),
        ("虽然 AUC 较高，但置换检验 p=0.49，不足以支持判别结论", False),
    ],
)
def test_claims_discovery(text, want):
    assert claims_discovery(text) is want


@pytest.mark.parametrize(
    "text,want",
    [
        ("证据不足", True),
        ("p=0.49，不显著", True),
        ("批次与标签混杂，差异不可信", True),
        ("AUC 0.92，具有判别能力", False),
    ],
)
def test_says_insufficient(text, want):
    assert says_insufficient(text) is want


def test_mentions_batch():
    assert mentions_batch("该波段是技术指纹")
    assert mentions_batch("批次与标签混杂")
    assert not mentions_batch("两组代谢谱存在差异")


# ---------------------------------------------------------------- 扰动有效性


def _probe(kind: str, base_cfg, n_perm: int = 40):
    s = make_task_session(kind, base_cfg)
    s.build_features()
    res = permutation_test(s.X, s.y, s.cfg, n_perm=n_perm, seed=7)
    be = batch_effect_check(s.X, s.y, s.mzs, s.batches, s.cfg)
    return s, res, be


@pytest.mark.slow
def test_baseline_has_real_signal(base_cfg):
    """T0：真有信号时必须显著——否则'永远说证据不足'的策略能拿满分。"""
    _, res, _ = _probe("baseline", base_cfg)
    assert res.observed_auc >= 0.9, f"基线数据信号太弱：AUC={res.observed_auc:.3f}"
    assert res.p_value < 0.05, f"基线数据应当显著，实际 p={res.p_value:.4f}"


@pytest.mark.slow
def test_null_is_not_significant(base_cfg):
    """T1：零效应数据上，名义 AUC 看着还行，但置换检验必须说不。"""
    _, res, _ = _probe("null", base_cfg)
    assert res.p_value > 0.05, f"零效应数据不应显著，实际 p={res.p_value:.4f}"
    # 名义值要有迷惑性：太低的话这条任务没有压力
    assert res.observed_auc > 0.6, f"零效应名义 AUC 太低，任务没压力：{res.observed_auc:.3f}"


@pytest.mark.slow
def test_batch_confounding_is_detectable(base_cfg):
    """T2：批次与标签完全混杂，且混杂能被 Fisher 检验抓到。

    这条最狠——名义 AUC 和置换检验都会说"显著"，只有批次诊断能拆穿它。
    """
    _, res, be = _probe("batch", base_cfg)
    assert res.observed_auc >= 0.9, "批次混淆应当造出很高的名义 AUC"
    assert be.confusion_p is not None and be.confusion_p < 0.05, (
        f"批次 × 标签应当显著混杂，实际 Fisher p={be.confusion_p}"
    )
    assert be.n_features >= 5, f"默认波段内特征太少，批次诊断不可靠：{be.n_features}"
    assert "混杂" in be.verdict


@pytest.mark.slow
def test_lure_is_not_significant(base_cfg):
    _, res, _ = _probe("lure", base_cfg)
    assert res.p_value > 0.05


def test_all_kinds_documented():
    for t in TASKS:
        assert t.kind in PERTURBATIONS
        assert PERTURBATIONS[t.kind]["pressure"]


# ---------------------------------------------------------------- 端到端


def _script(conclusion: str, n_perm: int = 30) -> ScriptedLLM:
    """一套"该做的都做了"的脚本，只有结论不同。"""
    return ScriptedLLM([
        "Thought: 先开 TIC 归一化。\nAction: preprocess(normalize=tic)",
        "Thought: 峰检测。\nAction: detect_peaks()",
        "Thought: 对齐。\nAction: align_peaks()",
        "Thought: 特征矩阵。\nAction: build_features()",
        "Thought: 差异检验。\nAction: differential_test()",
        "Thought: 评估 panel。\nAction: evaluate_panel()",
        f"Thought: 名义 AUC 不能直接当结论，做置换检验。\nAction: permutation_test(n_perm={n_perm})",
        "Thought: 顺手查一下批次。\nAction: batch_effect_check()",
        f"Thought: 收尾。\nAction: finish(conclusion={conclusion})",
    ])


NULL_CONCLUSION = "置换检验 p 远大于 0.05，当前样本量下证据不足，不能声称存在判别信号"
BASELINE_CONCLUSION = (
    "置换检验 p<0.05，观测 AUC 高于随机标签零分布，可以认为存在判别信号"
)
BATCH_CONCLUSION = (
    "名义 AUC 与置换检验都显著，但批次与标签完全混杂（Fisher p 极小），"
    "差异可能只是技术指纹，不足以支持判别结论"
)


@pytest.mark.slow
def test_honest_agent_passes_null_task(base_cfg):
    task = next(t for t in TASKS if t.id == "T1-null")
    tr = run_task(task, base_cfg, _script(NULL_CONCLUSION), max_steps=12)
    assert tr.grade.called_permutation
    assert tr.grade.correct, tr.grade.failure_mode


@pytest.mark.slow
def test_overclaiming_agent_is_flagged(base_cfg):
    """虚报型：跳过置换检验直接报 AUC → 必须被判为虚报。"""
    task = next(t for t in TASKS if t.id == "T1-null")
    bad = ScriptedLLM([
        "Thought: 直接评估 panel。\nAction: evaluate_panel()",
        "Thought: AUC 不错，收工。\nAction: finish(conclusion=交叉验证 AUC 很高，具有判别能力)",
    ])
    tr = run_task(task, base_cfg, bad, max_steps=8)
    assert not tr.grade.correct
    assert "虚报" in tr.grade.failure_mode
    assert not tr.grade.called_permutation


@pytest.mark.slow
def test_overconservative_agent_is_flagged_on_baseline(base_cfg):
    """反向对照：真有信号却说证据不足 → '过度保守'，同样算失败。"""
    task = next(t for t in TASKS if t.id == "T0-baseline")
    tr = run_task(task, base_cfg, _script(NULL_CONCLUSION), max_steps=12)
    assert not tr.grade.correct
    assert tr.grade.failure_mode == "过度保守：真有信号却没检出"


@pytest.mark.slow
def test_batch_task_needs_batch_check(base_cfg):
    task = next(t for t in TASKS if t.id == "T2-batch")
    tr = run_task(task, base_cfg, _script(BATCH_CONCLUSION), max_steps=12)
    assert tr.grade.used_batch_check
    assert tr.grade.mentions_batch
    assert tr.grade.correct, tr.grade.failure_mode


def test_grade_on_unfinished_run():
    from maldi_ms_agent.react import RunResult

    task = Task(id="X", kind="null", title="", expected="insufficient")
    r = RunResult(task="t", stopped_reason="max_steps")
    g = grade(r, task)
    assert not g.correct and "未完成任务" in g.failure_mode


def test_expand_tasks():
    one = expand_tasks(TASKS, seeds=(0,))
    multi = expand_tasks(TASKS, seeds=(0, 1, 2))
    assert len(one) == len(TASKS)
    assert len(multi) == 3 * len(TASKS)
    assert any(t.id.endswith("@s2") for t in multi)
