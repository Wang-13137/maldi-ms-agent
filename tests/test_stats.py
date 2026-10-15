"""置换检验与批次效应诊断的测试。

这里最重要的一条是 ``test_permutation_detects_no_signal``：
在**完全没有真实信号**的数据上，置换检验必须给出"证据不足"。
如果这个测试挂了，整个项目的立论就不成立。
"""

from __future__ import annotations

import numpy as np
import pytest

from maldi_ms_agent.session import AnalysisSession
from maldi_ms_agent.stats import (
    batch_effect_check,
    permutation_test,
    summarize_null,
)
from maldi_ms_agent.synth import SynthConfig, make_synth_dataset


def _session(cfg: dict, effect_size: float, seed: int = 11, **synth_kw) -> AnalysisSession:
    ds = make_synth_dataset(
        SynthConfig(n_case=12, n_ctrl=12, effect_size=effect_size, seed=seed, **synth_kw)
    )
    s = AnalysisSession(cfg=dict(cfg))
    s.attach_dataset(ds)
    s.build_features()
    return s


# ---------------------------------------------------------------- 置换检验


def test_permutation_detects_no_signal(base_cfg):
    """无真实信号（effect_size=0）→ 必须判"证据不足"。"""
    s = _session(base_cfg, effect_size=0.0, seed=3)
    res = permutation_test(s.X, s.y, s.cfg, n_perm=30, seed=0)
    assert not res.significant, (
        f"无信号数据上不应显著：p={res.p_value:.4f}, auc={res.observed_auc:.4f}"
    )
    assert "证据不足" in res.verdict


def test_permutation_detects_strong_signal(base_cfg):
    """强真实信号（effect_size=2.0）→ 应当显著。"""
    s = _session(base_cfg, effect_size=2.0, seed=5)
    res = permutation_test(s.X, s.y, s.cfg, n_perm=30, seed=0)
    assert res.significant, (
        f"强信号数据上应当显著：p={res.p_value:.4f}, auc={res.observed_auc:.4f}"
    )
    assert res.percentile > 0.9, f"观测值应远高于零分布，实际分位 {res.percentile:.2f}"


def test_p_value_floor(base_cfg):
    """加一修正保证 p 不会是 0，且下界为 1/(1+n_perm)。"""
    s = _session(base_cfg, effect_size=3.0, seed=9)
    n = 20
    res = permutation_test(s.X, s.y, s.cfg, n_perm=n, seed=0)
    assert res.p_value >= 1.0 / (1.0 + n) - 1e-9
    assert res.p_value > 0.0


def test_result_fields_are_consistent(base_cfg):
    s = _session(base_cfg, effect_size=1.0, seed=13)
    res = permutation_test(s.X, s.y, s.cfg, n_perm=25, seed=1)
    assert res.n_perm == len(res.null_samples)
    assert 0.0 <= res.percentile <= 1.0
    assert res.null_p95 >= res.null_mean - 1e-6
    assert res.null_max >= res.null_p95 - 1e-6
    assert isinstance(summarize_null(res), str) and "零分布" in summarize_null(res)


def test_reproducible_with_seed(base_cfg):
    s = _session(base_cfg, effect_size=1.0, seed=17)
    a = permutation_test(s.X, s.y, s.cfg, n_perm=15, seed=42)
    b = permutation_test(s.X, s.y, s.cfg, n_perm=15, seed=42)
    assert a.p_value == b.p_value
    assert a.observed_auc == b.observed_auc


# ---------------------------------------------------------------- 批次效应


def test_batch_effect_needs_two_batches(base_cfg):
    s = _session(base_cfg, effect_size=1.0, seed=21, n_batches=1)
    with pytest.raises(ValueError):
        batch_effect_check(s.X, s.y, s.mzs, s.batches, s.cfg)


def test_batch_effect_confusion_is_detected(base_cfg):
    """批次与标签完全混淆时，Fisher p 应当很小（提示混杂）。"""
    s = _session(
        base_cfg, effect_size=1.0, seed=23,
        batch_label_coupling=1.0, batch_shift=1.0,
    )
    res = batch_effect_check(s.X, s.y, s.mzs, s.batches, s.cfg, lo_mz=50, hi_mz=600)
    assert res.n_features > 0
    assert res.confusion_p is not None
    assert res.confusion_p < 0.05, f"完全混淆时应当检出混杂，实际 p={res.confusion_p:.4f}"


def test_batch_effect_empty_band(base_cfg):
    """波段内没有特征时不应崩溃，而是返回 n_features=0。"""
    s = _session(base_cfg, effect_size=1.0, seed=29)
    res = batch_effect_check(s.X, s.y, s.mzs, s.batches, s.cfg, lo_mz=1e6, hi_mz=1e6 + 1)
    assert res.n_features == 0
    assert np.isnan(res.batch_auc)
