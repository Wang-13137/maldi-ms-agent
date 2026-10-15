"""合成谱生成器。

存在理由：37 例真实谱与 47×37 特征矩阵是未发表数据，**不能进公开仓库**。
但评测必须能在没有真实数据的情况下复现，所以要有可控的合成数据——
效应量、批次效应、噪声特征数量都能调，用来构造三类扰动。

设计上刻意让合成谱走完 pipeline 的完整链路（读谱 → 峰检测 → 对齐 → 过滤 → 建模），
这样跑出来的数字才有意义：峰要够宽（>=3 个数据点）、够高（SNR>3），
不能是随手造的矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from maldi_ms.io import Dataset, Spectrum


@dataclass
class SynthConfig:
    """合成数据配置。"""

    n_case: int = 20            # 感染组例数
    n_ctrl: int = 20            # 健康组例数
    mz_lo: float = 50.0
    mz_hi: float = 600.0
    n_points: int = 20000       # 每条谱的数据点数
    n_signal_peaks: int = 10    # 真实有差异的峰
    n_null_peaks: int = 10      # 纯噪声峰（两组无差异）
    effect_size: float = 1.5    # 效应量（log10 强度差）；0 = 完全无差异
    batch_shift: float = 0.0    # 批次效应强度（log10）；0 = 无批次效应
    n_batches: int = 2
    # 批次与标签的相关程度：0=完全无关，1=批次 A 全是病例、B 全是对照（批次混淆）
    batch_label_coupling: float = 0.0
    noise_level: float = 0.02   # 基线噪声（相对强度）
    peak_sigma_da: float = 0.15
    seed: int = 42

    @property
    def step_da(self) -> float:
        return (self.mz_hi - self.mz_lo) / self.n_points


def _gaussian(mz: np.ndarray, center: float, amp: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((mz - center) / sigma) ** 2)


def make_synth_dataset(cfg: Optional[SynthConfig] = None) -> Dataset:
    """生成一份合成数据集（样本谱 + 阴性对照谱）。"""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    mz = np.linspace(cfg.mz_lo, cfg.mz_hi, cfg.n_points)

    # 峰位：先均匀撒点，再强制最小间隔 0.4 Da（峰检测要求 >=0.2 Da）
    n_all = cfg.n_signal_peaks + cfg.n_null_peaks
    centers = np.sort(rng.uniform(cfg.mz_lo + 5, cfg.mz_hi - 5, n_all))
    min_gap = 0.4
    for i in range(1, len(centers)):
        if centers[i] - centers[i - 1] < min_gap:
            centers[i] = centers[i - 1] + min_gap
    signal_idx = np.sort(rng.choice(n_all, size=cfg.n_signal_peaks, replace=False))
    signal_mask = np.zeros(n_all, dtype=bool)
    signal_mask[signal_idx] = True

    # 基础强度（log10 尺度），峰要远高于基线噪声才能被 SNR>3 检出
    base_amp = 10 ** rng.uniform(2.5, 4.0, size=n_all)

    # 组分配：0=对照，1=病例
    y = np.array([1] * cfg.n_case + [0] * cfg.n_ctrl)

    # 批次分配：按 coupling 决定与标签的相关程度
    batches = np.zeros(len(y), dtype=int)
    if cfg.n_batches >= 2:
        if cfg.batch_label_coupling >= 1.0:
            batches = y.copy()  # 完全混淆
        elif cfg.batch_label_coupling <= 0:
            batches = rng.integers(0, cfg.n_batches, size=len(y))
        else:
            # 以 coupling 的概率让批次跟随标签
            follow = rng.random(len(y)) < cfg.batch_label_coupling
            batches = np.where(follow, y, rng.integers(0, cfg.n_batches, size=len(y)))
            if len(np.unique(batches)) < 2:
                batches[0] = 1 - batches[1]

    samples: Dict[str, Spectrum] = {}
    for i, yi in enumerate(y):
        noise = rng.normal(0, cfg.noise_level, cfg.n_points)
        intensity = np.abs(noise) + cfg.noise_level

        amps = base_amp.copy()
        # 生物信号：病例组在 signal 峰上抬高 effect_size 个 log10 单位
        if cfg.effect_size != 0:
            amps[signal_mask] *= 10 ** (cfg.effect_size * yi)
        # 批次效应：整批样本的峰强度整体偏移
        if cfg.batch_shift != 0 and cfg.n_batches >= 2:
            amps *= 10 ** (cfg.batch_shift * (batches[i] / max(cfg.n_batches - 1, 1)))
        # 个体波动
        amps *= 10 ** rng.normal(0, 0.15, size=n_all)

        for c, a in zip(centers, amps):
            intensity += _gaussian(mz, c, a, cfg.peak_sigma_da)

        name = f"UTI{100 + i}"
        samples[name] = Spectrum(
            name=name, mz=mz.copy(), intensity=intensity,
            group="感染组" if yi == 1 else "健康组",
            source_file=f"batch{batches[i]}.xlsx",
        )

    # 阴性对照：只有基线噪声 + 极低强度的背景峰
    negatives: Dict[str, Spectrum] = {}
    for k in range(2):
        intensity = np.abs(rng.normal(0, cfg.noise_level, cfg.n_points)) + cfg.noise_level
        for c in centers[::3]:
            intensity += _gaussian(mz, c, base_amp[0] * 0.01, cfg.peak_sigma_da)
        nname = f"neg_CN_{k}"
        negatives[nname] = Spectrum(
            name=nname, mz=mz.copy(), intensity=intensity,
            group="阴性对照", source_file="blank.xlsx",
        )

    return Dataset(samples=samples, negatives=negatives)


def synth_config(cfg: dict) -> dict:
    """给合成数据配一份合适的 pipeline 配置（覆盖默认 uti8_50 的部分参数）。

    合成谱的点数远少于真实谱（2 万 vs 21 万），峰更宽，所以 SNR 与最小峰宽要跟着调。
    """
    import copy

    out = copy.deepcopy(cfg)
    out.setdefault("peak_detection", {}).update({
        "snr_threshold": 3,
        "min_peak_distance": 0.2,
        "min_intensity": 0.0,           # 合成谱强度尺度不同，不设绝对下限
        "min_peak_width_pts": 3,
    })
    out.setdefault("dataset", {}).update({
        "infection_uti": [],            # 合成数据靠 group 字段，不靠编号
        "sample_pattern": r"UTI(\d+)",
    })
    # 置换检验要跑几百上千遍建模，500 棵树的 RF 太慢；合成数据上 60 棵足够排序。
    # 注意：这只用于合成/测试配置，真实数据仍用 pipeline 默认的 500 棵。
    out.setdefault("modeling", {}).setdefault("rf", {})["n_estimators"] = 60
    return out


__all__ = ["SynthConfig", "make_synth_dataset", "synth_config"]
