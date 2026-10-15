"""把 maldi-ms-pipeline 的各阶段包装成 Agent 可调用的工具。

设计上最关键的一处是 ``tool_evaluate_panel`` 的输出：它返回名义 AUC 时**强制附带
口径警示**，提醒 Agent 先做置换检验。这比在 system prompt 里说教更可靠，而且可消融
（``build_tools(honest=False)`` 去掉警示，用于对照实验 B 组）。

合规硬约束：所有工具**只返回汇总统计文本**（样本数、峰数、AUC、p 值、m/z 列表），
原始谱与完整特征矩阵绝不出现在返回文本里，因此可以安全地进入 LLM 请求体。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from maldi_ms.modeling import evaluate_panels

from .session import AnalysisSession, StepMissing
from .stats import batch_effect_check, permutation_test, summarize_null


# ---------------------------------------------------------------- 数据结构


@dataclass
class ToolResult:
    """工具返回值。

    ``text`` 是给 LLM 看的；``payload`` 是结构化数据，**默认不进 LLM**，
    只用于评测与落盘（落盘前还要再过滤一遍数组字段）。
    """

    text: str
    payload: Dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.payload.get("error")


@dataclass
class Tool:
    name: str
    description: str
    args_hint: str
    func: Callable[..., ToolResult]


#: 口径警示。把它放在工具输出里而不是提示词里，是本项目的核心设计。
AUC_WARNING = (
    "\n⚠️ 以上为**名义 AUC**，未经置换校正。样本量 {n}（感染 {k}）时，"
    "随机标签也能达到相近水平。报告判别能力前请先调用 `permutation_test`。"
)


def _err(msg: str) -> ToolResult:
    """工具级错误。文本会被回喂给 Agent，让它自己纠正。"""
    return ToolResult(f"[错误] {msg}", {"error": msg})


# ---------------------------------------------------------------- 1. 读取


def tool_load_dataset(session: AnalysisSession, data_dir: str = "", **kw) -> ToolResult:
    try:
        session.load(data_dir)
    except Exception as exc:  # noqa: BLE001
        return _err(f"读取 {data_dir} 失败：{exc}")
    return ToolResult(
        session.summarize_dataset(),
        {"n_samples": len(session.sample_names), "data_dir": data_dir},
    )


# ---------------------------------------------------------------- 2. 预处理


def tool_preprocess(
    session: AnalysisSession,
    normalize: str = "",
    baseline: str = "",
    **kw,
) -> ToolResult:
    """设置预处理参数并报告归一化诊断。

    这里刻意暴露"样本间强度求和差异可达上百倍"这个坑——不做归一化就直接比峰面积
    是错的，而这个数字能让 Agent 自己看到问题。
    """
    p = session.cfg.setdefault("preprocessing", {})
    if normalize:
        if normalize not in ("none", "tic"):
            return _err(f"normalize 只能是 none 或 tic，收到 {normalize}")
        if p.get("normalization") != normalize:
            p["normalization"] = normalize
            session.invalidate("peaks")
    if baseline:
        flag = baseline.lower() in ("1", "true", "yes", "on")
        if p.get("baseline_correction") != flag:
            p["baseline_correction"] = flag
            session.invalidate("peaks")

    session.require("dataset", "load_dataset")
    ints = [float(np.sum(s.intensity)) for s in session.dataset.samples.values()]
    lo, hi = min(ints), max(ints)
    ratio = hi / max(lo, 1e-12)
    note = ""
    if ratio > 10 and p.get("normalization", "none") == "none":
        note = (
            "\n⚠️ 未做归一化时样本间总强度相差 %.0f 倍，直接比较峰面积会引入系统性偏差，"
            "建议 normalization=tic。" % ratio
        )
    return ToolResult(
        f"预处理设置：normalization={p.get('normalization', 'none')}，"
        f"baseline_correction={p.get('baseline_correction', False)}；"
        f"样本总强度中位 {np.median(ints):.3g}，最大/最小 = {ratio:.1f} 倍。" + note,
        {"normalization": p.get("normalization"), "tic_ratio": ratio},
    )


# ---------------------------------------------------------------- 3. 峰检测


def tool_detect_peaks(
    session: AnalysisSession, snr: float | None = None, min_distance: float | None = None, **kw
) -> ToolResult:
    pd_cfg = session.cfg.setdefault("peak_detection", {})
    changed = False
    if snr is not None:
        if pd_cfg.get("snr_threshold") != float(snr):
            pd_cfg["snr_threshold"] = float(snr)
            changed = True
    if min_distance is not None:
        if pd_cfg.get("min_peak_distance") != float(min_distance):
            pd_cfg["min_peak_distance"] = float(min_distance)
            changed = True
    if changed:
        session.invalidate("peaks")

    t0 = time.time()
    try:
        session.detect()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"峰检测失败：{exc}")
    return ToolResult(
        f"峰检测完成（SNR≥{pd_cfg.get('snr_threshold')}，"
        f"最小间隔 {pd_cfg.get('min_peak_distance')} Da，耗时 {time.time()-t0:.1f}s）：\n"
        + session.summarize_peaks(),
        {"total_peaks": sum(len(p) for p in session.peaks_by_sample.values())},
    )


# ---------------------------------------------------------------- 4. 对齐


def tool_align_peaks(
    session: AnalysisSession, ppm_tolerance: float | None = None, **kw
) -> ToolResult:
    a_cfg = session.cfg.setdefault("alignment", {})
    if ppm_tolerance is not None and a_cfg.get("ppm_tolerance") != float(ppm_tolerance):
        a_cfg["ppm_tolerance"] = float(ppm_tolerance)
        session.invalidate("align")

    try:
        session.align()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"对齐失败：{exc}")

    sizes = [len(c) for c in session.clusters]
    singletons = sum(1 for s in sizes if s == 1)
    n_total = sum(sizes)
    return ToolResult(
        f"对齐完成（{a_cfg.get('ppm_tolerance')} ppm）：{n_total} 个检出峰归并成 "
        f"{len(session.clusters)} 个特征簇；平均每簇覆盖 "
        f"{np.mean(sizes):.1f} 条记录，仅出现于单个样本的簇 {singletons} 个"
        f"（{singletons/max(len(session.clusters),1):.0%}）。",
        {"n_clusters": len(session.clusters), "n_singletons": singletons},
    )


# ---------------------------------------------------------------- 5. 特征矩阵


def tool_build_features(
    session: AnalysisSession, detection_rate: float | None = None, **kw
) -> ToolResult:
    f_cfg = session.cfg.setdefault("filtering", {})
    if detection_rate is not None and f_cfg.get("min_detection_ratio") != float(detection_rate):
        f_cfg["min_detection_ratio"] = float(detection_rate)
        session.invalidate("features")

    try:
        session.build_features()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"构建特征矩阵失败：{exc}")
    return ToolResult(
        session.summarize_features(),
        {"n_features": int(session.frame.shape[1]), "n_samples": int(session.frame.shape[0])},
    )


# ---------------------------------------------------------------- 6. 差异检验


def tool_differential_test(
    session: AnalysisSession, alpha: float = 0.05, **kw
) -> ToolResult:
    try:
        session.differential()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"差异检验失败：{exc}")

    diff = session.diff
    n_sig = int((diff["p_value"] < float(alpha)).sum())
    n_fdr = int((diff["FDR"] < float(alpha)).sum())
    top = diff.head(10)
    lines = [
        f"差异检验（两组 t 检验，log10 空间）：共 {len(diff)} 个特征，"
        f"p<{alpha} 的 {n_sig} 个，FDR<{alpha} 的 {n_fdr} 个。",
        "p 值最小的 10 个特征（m/z, p, FDR, log2FC）：",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"  {r['m/z']:.2f} Da  p={r['p_value']:.4f}  FDR={r['FDR']:.4f}  "
            f"log2FC={r['log2(FC)']:+.2f}"
        )
    if n_fdr == 0:
        lines.append(
            "⚠️ 没有任何特征通过 FDR 校正。在未做置换检验前，不能据此判断两组有差异。"
        )
    return ToolResult("\n".join(lines), {"n_sig": n_sig, "n_fdr": n_fdr})


# ---------------------------------------------------------------- 7. panel 评估


def tool_evaluate_panel(
    session: AnalysisSession,
    sizes: str = "",
    honest: bool = True,
    **kw,
) -> ToolResult:
    """评估候选 panel，返回名义 AUC 曲线。

    ``honest=True`` 时附带口径警示（A 组）；``False`` 时只给数字（B 组，用于消融）。
    """
    if sizes:
        try:
            parsed = [int(s) for s in str(sizes).replace("，", ",").split(",") if s.strip()]
        except ValueError:
            return _err(f"sizes 解析失败：{sizes}")
        m_cfg = session.cfg.setdefault("modeling", {})
        if m_cfg.get("panel_sizes") != parsed:
            m_cfg["panel_sizes"] = parsed

    try:
        session.differential()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))

    try:
        session.panel = evaluate_panels(
            session.X, session.y, session.cfg,
            ttest_order=session.ttest_order(), verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"panel 评估失败：{exc}")
    session.log.append("evaluate_panel")

    p = session.panel
    curve = "  ".join(f"Top{k}: {v:.3f}" for k, v in sorted(p.curve.items()))
    text = (
        f"panel 评估（SVM {session.cfg['modeling'].get('cv_folds', 5)} 折 CV）：\n"
        f"  按 RF 重要性排序：{curve}\n"
        f"  最优 panel = Top {p.best_size}，名义 AUC = {p.best_auc:.4f}"
    )
    if honest:
        n = int(session.frame.shape[0])
        k = int(session.y.sum())
        text += AUC_WARNING.format(n=n, k=k)
    return ToolResult(
        text,
        {"best_size": p.best_size, "best_auc": float(p.best_auc), "honest": honest},
    )


# ---------------------------------------------------------------- 8. 置换检验


def tool_permutation_test(
    session: AnalysisSession, n_perm: int = 200, seed: int = 42, **kw
) -> ToolResult:
    """同流程标签置换检验——本项目的杀手锏工具。"""
    try:
        session.build_features()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))

    t0 = time.time()
    try:
        res = permutation_test(
            session.X, session.y, session.cfg,
            n_perm=int(n_perm), seed=int(seed),
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"置换检验失败：{exc}")
    elapsed = time.time() - t0
    session.log.append(
        f"permutation_test(n={res.n_perm}, p={res.p_value:.4f}, auc={res.observed_auc:.4f})"
    )

    text = (
        f"标签置换检验（{res.n_perm} 次，整套流程重跑：特征选择 + CV，耗时 {elapsed:.1f}s）：\n"
        f"  观测 AUC = {res.observed_auc:.4f}\n"
        f"  p = {res.p_value:.4f}（{res.n_ge}/{res.n_perm} 次随机标签 ≥ 观测值）\n"
        f"  零分布：均值 {res.null_mean:.4f}，标准差 {res.null_std:.4f}，"
        f"95 分位 {res.null_p95:.4f}，最大 {res.null_max:.4f}\n"
        f"  观测值分位排名：{res.percentile:.1%}\n"
        f"结论：{res.verdict}\n"
        f"{summarize_null(res)}"
    )
    return ToolResult(
        text,
        {
            "observed_auc": res.observed_auc,
            "p_value": res.p_value,
            "null_mean": res.null_mean,
            "significant": res.significant,
            "n_perm": res.n_perm,
        },
    )


# ---------------------------------------------------------------- 9. 批次效应


def tool_batch_effect_check(
    session: AnalysisSession, lo_mz: float = 250.0, hi_mz: float = 499.7, **kw
) -> ToolResult:
    try:
        session.build_features()
    except StepMissing as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))

    if session.batches is None or len(np.unique(session.batches)) < 2:
        return _err("当前数据集没有可用的批次信息（只识别到 1 个批次），无法做批次效应诊断。")

    try:
        res = batch_effect_check(
            session.X, session.y, session.mzs, session.batches,
            session.cfg, lo_mz=float(lo_mz), hi_mz=float(hi_mz),
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"批次效应诊断失败：{exc}")
    session.log.append(f"batch_effect_check({res.lo_mz}-{res.hi_mz})")

    return ToolResult(
        f"批次效应诊断：{res.verdict}",
        {
            "batch_auc": res.batch_auc,
            "label_auc": res.label_auc,
            "confusion_p": res.confusion_p,
            "n_features": res.n_features,
        },
    )


# ---------------------------------------------------------------- 10. 结束


def tool_finish(session: AnalysisSession, conclusion: str = "", **kw) -> ToolResult:
    if not conclusion.strip():
        return _err("conclusion 不能为空，请写出你的最终结论。")
    session.log.append("finish")
    return ToolResult(f"[结束] {conclusion}", {"conclusion": conclusion, "finished": True})


# ---------------------------------------------------------------- 工具集


def build_tools(
    session: AnalysisSession,
    honest: bool = True,
    with_permutation: bool = True,
) -> List[Tool]:
    """构造工具集。

    - ``honest=True``  ：``evaluate_panel`` 附口径警示（A 组）
    - ``honest=False`` ：只返回数字（B 组，用于消融"工具契约 vs 沉默工具"）
    - ``with_permutation=False`` ：工具箱里不含 ``permutation_test``（C 组，
      用于回答"没有工具时 Agent 会不会自己想办法"）
    """
    tools: List[Tool] = [
        Tool(
            "load_dataset",
            "读取质谱数据目录，返回样本数、两组例数、批次分布与数据量。",
            "data_dir=<数据目录路径>",
            partial(tool_load_dataset, session),
        ),
        Tool(
            "preprocess",
            "设置预处理参数（归一化、基线校正）并报告样本间总强度差异。",
            "normalize=none|tic, baseline=true|false",
            partial(tool_preprocess, session),
        ),
        Tool(
            "detect_peaks",
            "执行预处理与峰检测，返回每样本峰数与阴性对照峰数。",
            "snr=<信噪比阈值>, min_distance=<Da>",
            partial(tool_detect_peaks, session),
        ),
        Tool(
            "align_peaks",
            "跨样本 m/z 对齐，返回特征簇数与单例簇占比。",
            "ppm_tolerance=<ppm>",
            partial(tool_align_peaks, session),
        ),
        Tool(
            "build_features",
            "背景过滤与检出率过滤，构建特征矩阵并返回维度与稀疏度。",
            "detection_rate=<0~1>",
            partial(tool_build_features, session),
        ),
        Tool(
            "differential_test",
            "逐特征两组 t 检验 + BH 校正，返回显著特征数与 top10。",
            "alpha=<显著性水平>",
            partial(tool_differential_test, session),
        ),
        Tool(
            "evaluate_panel",
            "评估候选特征 panel 的交叉验证 AUC。",
            "sizes=<如 3,5,10,20>",
            partial(tool_evaluate_panel, session, honest=honest),
        ),
    ]
    if with_permutation:
        tools.append(
            Tool(
                "permutation_test",
                "同流程标签置换检验：把标签随机打乱后重跑整套流程，给出观测 AUC 的 p 值"
                "与随机标签零分布。用于判断判别结论能否与随机区分。",
                "n_perm=<置换次数>, seed=<随机种子>",
                partial(tool_permutation_test, session),
            )
        )
    tools.append(
        Tool(
            "batch_effect_check",
            "诊断指定 m/z 波段是生物信号还是技术指纹：比较判别采集批次与判别生物标签的 AUC。",
            "lo_mz=<下界>, hi_mz=<上界>",
            partial(tool_batch_effect_check, session),
        )
    )
    tools.append(
        Tool(
            "finish",
            "给出最终结论并结束分析。结论必须说明判别证据是否充分及其依据。",
            "conclusion=<结论文本>",
            partial(tool_finish, session),
        )
    )
    return tools


def tools_prompt(tools: Sequence[Tool]) -> str:
    """把工具集渲染成提示词里的工具清单。"""
    lines = []
    for t in tools:
        lines.append(f"- {t.name}({t.args_hint})：{t.description}")
    return "\n".join(lines)


__all__ = [
    "Tool",
    "ToolResult",
    "AUC_WARNING",
    "build_tools",
    "tools_prompt",
    "tool_load_dataset",
    "tool_preprocess",
    "tool_detect_peaks",
    "tool_align_peaks",
    "tool_build_features",
    "tool_differential_test",
    "tool_evaluate_panel",
    "tool_permutation_test",
    "tool_batch_effect_check",
    "tool_finish",
]
