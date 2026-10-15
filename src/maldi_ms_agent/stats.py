"""判别证据的统计核验：标签置换检验与批次效应诊断。

这两个函数是本项目的核心新增能力——``maldi-ms-pipeline`` 里没有置换检验
（``grep -n "permut\\|shuffle\\|null"`` 在它的 ``src/`` 下只命中一处
``StratifiedKFold(shuffle=True)``），而这正是 37 例 pilot 结论被证伪的办法。

设计原则（合规硬约束）：本模块只返回汇总统计量（AUC、p 值、分位数），
绝不把原始谱或特征矩阵带出函数调用边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import fisher_exact

from maldi_ms.modeling import evaluate_panels


# ---------------------------------------------------------------- 置换检验


@dataclass
class PermutationResult:
    """标签置换检验的结果。"""

    observed_auc: float
    p_value: float
    n_perm: int
    null_mean: float
    null_std: float
    null_p95: float
    null_max: float
    # 观测值在零分布中的分位排名（0~1）；接近 1 说明观测值比几乎所有随机标签都高
    percentile: float
    # 零分布里 >= 观测值的次数（不含观测值自身）
    n_ge: int
    # 判定：p >= alpha 即"证据不足"
    alpha: float = 0.05
    # 每次置换的原始 AUC，用于画零分布直方图
    null_samples: List[float] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def verdict(self) -> str:
        """给 Agent 读的结论文本。措辞是"证据不足"而不是"证明无效"。"""
        if self.significant:
            return (
                f"显著：p={self.p_value:.4f} < {self.alpha}，"
                f"观测 AUC {self.observed_auc:.4f} 高于零分布 {self.percentile:.1%} 分位，"
                f"可以认为标签与特征之间存在超出随机的关联。"
            )
        return (
            f"证据不足：p={self.p_value:.4f} >= {self.alpha}。"
            f"观测 AUC {self.observed_auc:.4f}，随机标签零分布均值 {self.null_mean:.4f}、"
            f"95 分位 {self.null_p95:.4f}，有 {self.n_ge}/{self.n_perm} 次随机标签达到或超过观测值。"
            f"当前样本量下**不能**得出「具有判别能力」的结论。"
        )


def _panel_auc_for_labels(X: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    """对给定标签跑一遍完整的 panel 选择 + CV，返回名义 AUC。

    注意这里跑的是**全流程**（RF 重要性排序 → 逐个 panel 规模 → SVM CV 取最优），
    与真实标签上跑的流程完全一致。只重跑 CV 而沿用真实标签选出的特征会低估零分布
    （选择偏差），得到的 p 值偏乐观。
    """
    cfg = dict(cfg)  # 浅拷贝，避免下面改 modeling 配置影响调用方
    res = evaluate_panels(X, y, cfg, verbose=False)
    return float(res.best_auc)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    n_perm: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    scorer: Optional[Callable[[np.ndarray, np.ndarray, dict], float]] = None,
) -> PermutationResult:
    """同流程标签置换检验。

    把标签随机打乱 ``n_perm`` 次，每次**重跑整套流程**（特征选择 + 交叉验证），
    得到零分布；再用观测 AUC 在零分布中的位置算 p 值。

    p 值用 ``(1 + #{null >= obs}) / (1 + n_perm)``（加一修正），保证不会得到 0。

    性能参考：37 例 × 47 特征、RF 500 树时单次置换约 0.3~0.5 s，
    200 次约 1~2 分钟，1000 次约 6~8 分钟。正式结论建议 n_perm >= 1000。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    score = scorer or _panel_auc_for_labels

    observed = float(score(X, y, cfg))

    rng = np.random.default_rng(seed)
    null: List[float] = []
    y_work = y.copy()
    for _ in range(int(n_perm)):
        rng.shuffle(y_work)
        # 极端情况下打乱后只剩一类（样本极少时），跳过该次
        if len(np.unique(y_work)) < 2:
            continue
        try:
            null.append(float(score(X, y_work, cfg)))
        except Exception:
            # 单次失败不应中断整轮检验
            continue

    if not null:
        raise RuntimeError("置换检验未能产生任何零分布样本")

    null_arr = np.asarray(null, dtype=float)
    n_ge = int(np.sum(null_arr >= observed - 1e-12))
    n_eff = len(null_arr)
    p = (1.0 + n_ge) / (1.0 + n_eff)

    return PermutationResult(
        observed_auc=observed,
        p_value=float(p),
        n_perm=n_eff,
        null_mean=float(np.mean(null_arr)),
        null_std=float(np.std(null_arr)),
        null_p95=float(np.percentile(null_arr, 95)),
        null_max=float(np.max(null_arr)),
        percentile=float(np.mean(null_arr < observed)),
        n_ge=n_ge,
        alpha=alpha,
        null_samples=[float(v) for v in null_arr],
    )


# ---------------------------------------------------------------- 批次效应


@dataclass
class BatchEffectResult:
    """某个 m/z 波段上的批次效应诊断结果。"""

    lo_mz: float
    hi_mz: float
    n_features: int
    # 用该波段特征判别「采集批次」的 AUC
    batch_auc: float
    # 用该波段特征判别「生物学标签」的 AUC
    label_auc: float
    # 批次 × 标签 的 Fisher 精确检验 p 值；p 小 = 批次与标签混杂
    confusion_p: Optional[float]
    # 2x2 列联表 [[batch0&y0, batch0&y1], [batch1&y0, batch1&y1]]
    table: Optional[List[List[int]]]

    @property
    def verdict(self) -> str:
        gap = self.batch_auc - self.label_auc
        base = (
            f"{self.lo_mz}–{self.hi_mz} Da（{self.n_features} 个特征）："
            f"判别批次 AUC={self.batch_auc:.4f}，判别标签 AUC={self.label_auc:.4f}，"
            f"差值 {gap:+.4f}。"
        )
        if gap > 0.15:
            base += (
                "该波段更像**技术指纹**而非生物信号：模型更容易区分样本来自哪个批次，"
                "而不是它属于哪一组。基于该波段的判别结论不可信。"
            )
        elif self.confusion_p is not None and self.confusion_p < 0.05:
            base += "⚠️ 批次与标签存在显著混杂（Fisher p=%.4f），二者无法分离。" % self.confusion_p
        else:
            base += "未见明显批次主导迹象。"
        return base


def _auc_for_target(X: np.ndarray, target: np.ndarray, cfg: dict) -> float:
    """对任意二分类目标跑一遍 panel 流程，返回名义 AUC。"""
    return _panel_auc_for_labels(X, np.asarray(target).astype(int), cfg)


def batch_effect_check(
    X: np.ndarray,
    y: np.ndarray,
    mzs: Sequence[float],
    batches: Sequence[int | str],
    cfg: dict,
    lo_mz: float = 250.0,
    hi_mz: float = 499.7,
) -> BatchEffectResult:
    """诊断指定 m/z 波段是生物信号还是技术指纹。

    做法（来自项目一 maldi-dl 的发现）：用同一批特征分别去判别「采集批次」和
    「生物学标签」。若判别批次远比判别标签容易，说明该波段携带的是技术变异。

    参考实测：250–499.7 Da 波段上判别批次 AUC=0.938、判别标签仅 0.697
    （Fisher p=0.63），该波段不含 47 个候选特征中的任何一个。
    """
    X = np.asarray(X, dtype=float)
    mzs = np.asarray(mzs, dtype=float)
    y = np.asarray(y).astype(int)
    b_raw = np.asarray(batches)

    mask = (mzs >= float(lo_mz)) & (mzs <= float(hi_mz))
    n_features = int(mask.sum())

    # 批次编码成 0/1；多于两个批次时取"是否为第一个批次"
    levels = list(dict.fromkeys(list(b_raw)))
    if len(levels) < 2:
        raise ValueError("batch_effect_check 需要至少两个采集批次")
    b = np.array([0 if v == levels[0] else 1 for v in b_raw], dtype=int)

    if n_features == 0:
        return BatchEffectResult(
            lo_mz=float(lo_mz), hi_mz=float(hi_mz), n_features=0,
            batch_auc=float("nan"), label_auc=float("nan"),
            confusion_p=None, table=None,
        )

    Xb = X[:, mask]
    # 只有一个批次水平出现在该波段的有效样本中时，AUC 无意义
    batch_auc = float("nan")
    if len(np.unique(b)) >= 2:
        try:
            batch_auc = _auc_for_target(Xb, b, cfg)
        except Exception:
            batch_auc = float("nan")

    label_auc = float("nan")
    if len(np.unique(y)) >= 2:
        try:
            label_auc = _auc_for_target(Xb, y, cfg)
        except Exception:
            label_auc = float("nan")

    # 批次 × 标签 列联表
    confusion_p = None
    table: Optional[List[List[int]]] = None
    try:
        tbl = np.zeros((2, 2), dtype=int)
        for bi in (0, 1):
            for yi in (0, 1):
                tbl[bi, yi] = int(np.sum((b == bi) & (y == yi)))
        table = [[int(tbl[0, 0]), int(tbl[0, 1])], [int(tbl[1, 0]), int(tbl[1, 1])]]
        _, confusion_p = fisher_exact(tbl)
        confusion_p = float(confusion_p)
    except Exception:
        pass

    return BatchEffectResult(
        lo_mz=float(lo_mz), hi_mz=float(hi_mz), n_features=n_features,
        batch_auc=batch_auc, label_auc=label_auc,
        confusion_p=confusion_p, table=table,
    )


def summarize_null(res: PermutationResult, bins: int = 10) -> str:
    """把零分布压成一行直方图文本（给 LLM 看的紧凑表示）。"""
    if not res.null_samples:
        return "（无零分布样本）"
    arr = np.asarray(res.null_samples, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    parts = []
    for i, c in enumerate(counts):
        parts.append(f"{edges[i]:.2f}:{'#' * max(1, int(c))}({int(c)})")
    return (
        f"零分布 n={res.n_perm} 均值 {res.null_mean:.3f} "
        f"95分位 {res.null_p95:.3f} 最大 {res.null_max:.3f}；"
        f"观测 {res.observed_auc:.3f}（分位 {res.percentile:.1%}）\n"
        + " ".join(parts)
    )


__all__ = [
    "PermutationResult",
    "BatchEffectResult",
    "permutation_test",
    "batch_effect_check",
    "summarize_null",
]
