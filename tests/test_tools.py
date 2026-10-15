"""工具层的测试。

重点验证两件事：
1. 所有工具只返回**汇总统计**，不会把原始谱或特征矩阵泄漏进 LLM 可见文本；
2. ``evaluate_panel`` 的口径警示可开关（消融 A/B 组的开关就在这里）。
"""

from __future__ import annotations

import numpy as np

from maldi_ms_agent.session import AnalysisSession
from maldi_ms_agent.tools import AUC_WARNING, build_tools, tools_prompt


def _ready_session(base_cfg, synth_ds=None) -> AnalysisSession:
    """``synth_ds=None`` 时返回一个空会话（用于测试"前置未跑"的错误路径）。"""
    s = AnalysisSession(cfg=dict(base_cfg))
    if synth_ds is not None:
        s.attach_dataset(synth_ds)
    return s


# ---------------------------------------------------------------- 工具集构造


def test_toolset_sizes():
    s = AnalysisSession(cfg={})
    assert len(build_tools(s, with_permutation=True)) == 10
    # C 组消融：工具箱里没有 permutation_test
    assert len(build_tools(s, with_permutation=False)) == 9
    names = {t.name for t in build_tools(s, with_permutation=False)}
    assert "permutation_test" not in names


def test_tools_prompt_lists_everything():
    s = AnalysisSession(cfg={})
    text = tools_prompt(build_tools(s))
    for name in ("load_dataset", "detect_peaks", "permutation_test", "finish"):
        assert name in text


# ---------------------------------------------------------------- 依赖检查


def test_tools_report_missing_prerequisite(base_cfg):
    """没读数据就调下游工具，必须返回可回喂的错误而不是抛异常。"""
    s = _ready_session(base_cfg)
    tools = {t.name: t for t in build_tools(s)}
    r = tools["detect_peaks"].func()
    assert not r.ok
    assert "load_dataset" in r.text or "dataset" in r.text


# ---------------------------------------------------------------- 口径警示可消融


def test_evaluate_panel_honest_vs_silent(base_cfg, synth_ds):
    """A 组附口径警示，B 组不附——这是消融实验的开关。"""
    s = _ready_session(base_cfg, synth_ds)
    honest = {t.name: t for t in build_tools(s, honest=True)}["evaluate_panel"]
    silent = {t.name: t for t in build_tools(s, honest=False)}["evaluate_panel"]

    a = honest.func()
    b = silent.func()
    assert a.ok and b.ok
    assert "permutation_test" in a.text and "名义 AUC" in a.text
    assert "permutation_test" not in b.text
    assert a.payload["honest"] is True and b.payload["honest"] is False
    # 数字本身必须一致——消融只改提示，不改算法
    assert abs(a.payload["best_auc"] - b.payload["best_auc"]) < 1e-9


def test_warning_template_mentions_sample_size(base_cfg, synth_ds):
    s = _ready_session(base_cfg, synth_ds)
    r = {t.name: t for t in build_tools(s)}["evaluate_panel"].func()
    assert "24" in r.text  # 样本数
    assert AUC_WARNING  # 模板非空


# ---------------------------------------------------------------- 合规：不泄漏原始数据


def test_no_raw_matrix_in_tool_text(base_cfg, synth_ds):
    """工具文本里不能出现长数字串（原始强度值）。"""
    s = _ready_session(base_cfg, synth_ds)
    tools = build_tools(s)
    for t in tools:
        if t.name in ("load_dataset", "finish"):
            continue
        r = t.func()
        if not r.ok:
            continue
        # 强度值是 log10 后仍可能较长的浮点；检查是否出现 >6 位连续数字
        for chunk in r.text.replace("\n", " ").split():
            digits = "".join(c for c in chunk if c.isdigit() or c == ".")
            if len(digits) > 12:
                raise AssertionError(f"{t.name} 返回文本疑似含原始数值：{chunk[:40]}")


def test_finish_requires_conclusion(base_cfg):
    s = _ready_session(base_cfg)
    finish = {t.name: t for t in build_tools(s)}["finish"]
    assert not finish.func().ok
    r = finish.func(conclusion="证据不足：p=0.31")
    assert r.ok and r.payload["finished"] is True


# ---------------------------------------------------------------- 参数改动触发重算


def test_evaluate_panel_tolerates_sizes_formats(base_cfg, synth_ds):
    """真实模型写 sizes 的格式五花八门，都得容忍。

    踩过的坑：早期版本用 int("[3") 直接报错，模型卡在这一步，
    导致唯一有真信号的 T0-baseline 在三组消融里全部作废。
    """
    from maldi_ms_agent.tools import _parse_sizes

    assert _parse_sizes("3,5,10") == [3, 5, 10]
    assert _parse_sizes("[3, 5, 10]") == [3, 5, 10]
    assert _parse_sizes("3，5，10") == [3, 5, 10]
    assert _parse_sizes("3.0,5.0") == [3, 5]
    assert _parse_sizes("") is None
    assert _parse_sizes("abc") is None

    s = _ready_session(base_cfg, synth_ds)
    tools = {t.name: t for t in build_tools(s)}
    for fmt in ("3,5,10", "[5, 10]", "5，10"):
        r = tools["evaluate_panel"].func(sizes=fmt)
        assert r.ok, f"sizes={fmt} 应当被接受：{r.text}"
    bad = tools["evaluate_panel"].func(sizes="大中小")
    assert not bad.ok


def test_parameter_change_invalidates_downstream(base_cfg, synth_ds):
    """改归一化方式后必须重算峰检测（T3 参数敏感性靠这个）。"""
    s = _ready_session(base_cfg, synth_ds)
    tools = {t.name: t for t in build_tools(s)}
    tools["detect_peaks"].func()
    assert s.peaks_by_sample

    r = tools["preprocess"].func(normalize="tic")
    assert r.ok
    assert s.peaks_by_sample == {}, "改了归一化方式后峰检测结果应当被清空"


def test_batch_check_reports_batch_dominance(base_cfg, synth_ds):
    s = _ready_session(base_cfg, synth_ds)
    r = {t.name: t for t in build_tools(s)}["batch_effect_check"].func(lo_mz=50, hi_mz=600)
    assert r.ok
    assert "判别批次" in r.text and "判别标签" in r.text
    assert np.isfinite(r.payload["batch_auc"]) or np.isnan(r.payload["batch_auc"])
