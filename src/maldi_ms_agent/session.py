"""跨工具共享的分析会话状态。

为什么必须有这个东西：读谱 + 峰检测在 37 例上是分钟级的，如果每个工具都从原始
xlsx 重跑一遍，Agent 走完 T1 要几十分钟。这里按依赖顺序保存每一步的中间产物，
工具按需惰性计算，前置未跑则返回明确错误让 Agent 自己补上。

合规硬约束：本模块持有原始谱與特征矩阵，但**只有 summarize* 系列方法产出的文本
允许进入 LLM 请求体**。任何返回 numpy 数组 / DataFrame 的方法都只供工具内部使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from maldi_ms.io import Dataset, read_directory
from maldi_ms import alignment, filtering, modeling
from maldi_ms.modeling import PanelResult
from maldi_ms.peak_detection import pick_peaks
from maldi_ms.preprocessing import prepare_spectrum


class StepMissing(RuntimeError):
    """前置步骤未执行。错误消息会被回喂给 Agent，所以它必须说清缺哪一步。"""


@dataclass
class AnalysisSession:
    """一次判别分析的完整状态。"""

    cfg: dict = field(default_factory=dict)
    data_dir: Optional[str] = None

    # --- 各阶段产物（惰性填充）---
    dataset: Optional[Dataset] = None
    sample_names: List[str] = field(default_factory=list)
    peaks_by_sample: Dict[str, List[dict]] = field(default_factory=dict)
    negative_peaks: List[dict] = field(default_factory=list)
    clusters: List[List[dict]] = field(default_factory=list)
    frame_aligned: Optional[pd.DataFrame] = None   # 对齐后、过滤前
    frame: Optional[pd.DataFrame] = None           # 过滤后的最终特征矩阵
    mzs: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    batches: Optional[np.ndarray] = None
    diff: Optional[pd.DataFrame] = None
    panel: Optional[PanelResult] = None

    # 已执行的步骤日志（给 Agent 自己看它做过什么）
    log: List[str] = field(default_factory=list)

    # ------------------------------------------------------------ 读取

    def load(self, data_dir: str) -> None:
        self.dataset = read_directory(data_dir, self.cfg)
        self.sample_names = sorted(self.dataset.samples.keys())
        self.data_dir = data_dir
        # 批次：优先用调用方显式指定的；否则用来源文件做代理
        # （同一批采集的样本通常落在同一个 xlsx 里）
        if self.batches is None:
            src = [self.dataset.samples[n].source_file for n in self.sample_names]
            levels = list(dict.fromkeys(src))
            self.batches = np.array([levels.index(s) for s in src], dtype=int)
        self.log.append(f"load_dataset({data_dir})")

    def attach_dataset(self, ds: Dataset, batches: Optional[Sequence[int]] = None) -> None:
        """直接注入一个已构造好的 Dataset（不经过 ``read_directory``）。

        合成数据与单元测试走这条路——它们没有 xlsx 文件，只有内存里的 numpy 数组。
        """
        self.dataset = ds
        self.sample_names = sorted(ds.samples.keys())
        if batches is not None:
            self.batches = np.asarray(list(batches), dtype=int)
        elif self.batches is None:
            src = [ds.samples[n].source_file for n in self.sample_names]
            levels = list(dict.fromkeys(src))
            self.batches = np.array([levels.index(s) for s in src], dtype=int)
        self.log.append("attach_dataset")

    # ------------------------------------------------------------ 依赖检查

    # ------------------------------------------------------------ 失效

    #: 阶段顺序，靠后的依赖靠前的
    STAGES = ("peaks", "align", "features", "diff", "panel")

    def invalidate(self, stage: str) -> None:
        """清空 ``stage`` 及其之后的所有产物。

        改了上游参数（归一化方式、SNR、ppm、检出率）必须调用它，
        否则工具会返回用旧参数算出来的结果——T3 参数敏感性任务正是靠这个才有意义。
        """
        if stage not in self.STAGES:
            raise ValueError(f"未知阶段 {stage}，可选：{self.STAGES}")
        i = self.STAGES.index(stage)
        for s in self.STAGES[i:]:
            if s == "peaks":
                self.peaks_by_sample = {}
                self.negative_peaks = []
            elif s == "align":
                self.clusters = []
                self.frame_aligned = None
            elif s == "features":
                self.frame = None
                self.mzs = None
                self.X = None
                self.y = None
            elif s == "diff":
                self.diff = None
            elif s == "panel":
                self.panel = None
        # 日志里留痕，Agent 能看到自己触发了重算
        self.log.append(f"[invalidate from {stage}]")

    def require(self, what: str, how: str) -> None:
        if getattr(self, what) is None or (
            hasattr(getattr(self, what), "__len__") and len(getattr(self, what)) == 0
        ):
            raise StepMissing(f"还没有 {what}。请先调用 `{how}`。")

    # ------------------------------------------------------------ 峰检测

    def detect(self) -> None:
        """预处理 + 峰检测。幂等：已跑过则跳过。"""
        self.require("dataset", "load_dataset")
        if self.peaks_by_sample:
            return

        for name in self.sample_names:
            spec = self.dataset.samples[name]
            prepared = prepare_spectrum(spec.intensity, self.cfg)
            peaks, _noise = pick_peaks(spec.mz, prepared, self.cfg)
            self.peaks_by_sample[name] = peaks

        for nname in sorted(self.dataset.negatives.keys()):
            spec = self.dataset.negatives[nname]
            prepared = prepare_spectrum(spec.intensity, self.cfg)
            peaks, _ = pick_peaks(spec.mz, prepared, self.cfg)
            self.negative_peaks.extend(peaks)

        self.log.append("detect_peaks")

    # ------------------------------------------------------------ 对齐

    def align(self) -> None:
        self.detect()
        if self.frame_aligned is not None:
            return
        flat: List[dict] = []
        for name in self.sample_names:
            for p in self.peaks_by_sample[name]:
                flat.append({"sample": name, "mz": p["mz"], "intensity": p["intensity"]})
        self.clusters = alignment.align_peaks(
            flat, self.cfg["alignment"]["ppm_tolerance"]
        )
        self.frame_aligned = alignment.build_feature_table(self.clusters, self.sample_names)
        self.log.append("align_peaks")

    # ------------------------------------------------------------ 过滤

    def build_features(self) -> None:
        self.align()
        if self.frame is not None:
            return
        frame = filtering.filter_background(
            self.frame_aligned, self.negative_peaks, self.cfg, verbose=False
        )
        frame = filtering.filter_by_detection(frame, self.cfg, verbose=False)
        self.frame = frame
        self.mzs = alignment.feature_mz(frame)
        self.y = np.array(
            [1 if self.dataset.samples[n].group == "感染组" else 0 for n in frame.index],
            dtype=int,
        )
        self.X = modeling.transform(frame)
        self.log.append("build_features")

    # ------------------------------------------------------------ 差异检验

    def differential(self) -> None:
        self.build_features()
        if self.diff is not None:
            return
        self.diff = modeling.differential_test(self.frame, self.y, self.mzs)
        self.log.append("differential_test")

    # ------------------------------------------------------------ 给 LLM 的摘要

    def summarize_dataset(self) -> str:
        self.require("dataset", "load_dataset")
        ds = self.dataset
        n_inf = int(self.y.sum()) if self.y is not None else sum(
            1 for s in ds.samples.values() if s.group == "感染组"
        )
        n_hea = len(ds.samples) - n_inf
        pts = sum(s.n_points for s in ds.samples.values())
        mz_lo = min(float(s.mz.min()) for s in ds.samples.values())
        mz_hi = max(float(s.mz.max()) for s in ds.samples.values())
        lines = [
            f"样本 {len(ds.samples)} 例（感染 {n_inf} / 健康 {n_hea}），"
            f"阴性对照 {len(ds.negatives)} 例",
            f"每例 {ds.samples[self.sample_names[0]].n_points} 数据点，合计约 {pts/1e6:.1f} M 点",
            f"m/z 范围 {mz_lo:.1f}–{mz_hi:.1f} Da",
        ]
        if self.batches is not None:
            vals, counts = np.unique(self.batches, return_counts=True)
            dist = "、".join(f"批次{v}: {c} 例" for v, c in zip(vals, counts))
            lines.append(f"采集批次分布（按来源文件推断）：{dist}")
            lines.append("⚠️ 样本量提示：感染组仅 %d 例，任何判别结论都需要做置换检验。" % n_inf)
        return "\n".join(lines)

    def summarize_peaks(self) -> str:
        self.detect()
        counts = [len(p) for p in self.peaks_by_sample.values()]
        total = sum(counts)
        return (
            f"每样本峰数：中位 {int(np.median(counts))}，"
            f"范围 {min(counts)}–{max(counts)}，合计 {total} 个检出峰；"
            f"阴性对照检出 {len(self.negative_peaks)} 个峰。"
        )

    def summarize_features(self) -> str:
        self.build_features()
        n_before = self.frame_aligned.shape[1]
        n_after = self.frame.shape[1]
        zero_ratio = float((self.frame.values == 0).mean())
        det_rate = (self.frame > 0).mean(axis=0).values
        return (
            f"对齐特征 {n_before} → 过滤后 {n_after} 个（压缩 {1 - n_after/max(n_before,1):.0%}）；"
            f"矩阵 {self.frame.shape[0]}×{n_after}，零值占比 {zero_ratio:.0%}（极度稀疏）；"
            f"特征检出率中位 {np.median(det_rate):.0%}、最高 {det_rate.max():.0%}。"
        )

    def ttest_order(self) -> List[int]:
        """按 t 检验 p 值排序的特征索引，供 evaluate_panels 使用。"""
        self.differential()
        return list(range(len(self.diff)))


def new_session(cfg: dict, batches: Optional[Sequence[int]] = None) -> AnalysisSession:
    s = AnalysisSession(cfg=cfg)
    if batches is not None:
        s.batches = np.asarray(list(batches), dtype=int)
    return s
