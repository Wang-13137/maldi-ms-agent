"""会话状态与全流程链路的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from maldi_ms_agent.session import AnalysisSession, StepMissing


# base_cfg / synth_ds 由 conftest.py 提供（session 级，跨测试文件共享）


def test_attach_and_summarize(base_cfg, synth_ds):
    s = AnalysisSession(cfg=dict(base_cfg))
    s.attach_dataset(synth_ds)
    text = s.summarize_dataset()
    assert "样本 24 例" in text
    assert "感染 12" in text
    assert "采集批次分布" in text


def test_missing_step_raises(base_cfg, synth_ds):
    """前置未跑时必须报 StepMissing，且错误消息要指明补哪一步。"""
    s = AnalysisSession(cfg=dict(base_cfg))
    with pytest.raises(StepMissing) as ei:
        s.summarize_dataset()
    assert "load_dataset" in str(ei.value) or "attach" in str(ei.value)

    with pytest.raises(StepMissing):
        s.detect()


def test_lazy_steps_auto_chain(base_cfg, synth_ds):
    """下游步骤会自动补跑上游，不必按序手动调用。"""
    s = AnalysisSession(cfg=dict(base_cfg))
    s.attach_dataset(synth_ds)
    s.build_features()  # 一步到位，内部应自动 detect + align
    assert "detect_peaks" in s.log
    assert "align_peaks" in s.log


def test_full_chain(base_cfg, synth_ds):
    """合成谱必须能走完 读→峰检测→对齐→过滤→特征矩阵。"""
    s = AnalysisSession(cfg=dict(base_cfg))
    s.attach_dataset(synth_ds)
    s.build_features()
    assert s.frame is not None
    assert s.frame.shape[0] == 24
    assert s.frame.shape[1] >= 5, f"合成谱应当能对齐出足够多的特征，实际 {s.frame.shape[1]}"
    assert s.X.shape == s.frame.shape
    assert set(np.unique(s.y)) == {0, 1}


def test_invalidate_forces_recompute(base_cfg, synth_ds):
    """改上游参数后必须重算下游，否则 T3 参数敏感性任务没有意义。"""
    s = AnalysisSession(cfg=dict(base_cfg))
    s.attach_dataset(synth_ds)
    s.build_features()
    first = s.frame.shape[1]

    s.invalidate("features")
    assert s.frame is None and s.X is None
    s.build_features()
    assert s.frame.shape[1] == first

    # 改检出率阈值后特征数应当变化（或至少重算过）
    s.cfg["filtering"]["min_detection_ratio"] = 0.5
    s.invalidate("features")
    s.build_features()
    assert s.log.count("[invalidate from features]") >= 2


def test_labels_match_groups(base_cfg, synth_ds):
    s = AnalysisSession(cfg=dict(base_cfg))
    s.attach_dataset(synth_ds)
    s.build_features()
    names = list(s.frame.index)
    expect = np.array([1 if synth_ds.samples[n].group == "感染组" else 0 for n in names])
    assert np.array_equal(s.y, expect)
