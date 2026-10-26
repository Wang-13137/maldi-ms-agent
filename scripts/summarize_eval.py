"""把 run_eval.py 的多 seed 报告汇总成可直接贴 README 的 markdown。

设计说明（为什么比 run_eval.py 自带的汇总更进一步）：

- **配对分析**：同一个 (task, seed) 在四组消融里各跑一次，所以 A vs C 是配对设计，
  应该用 McNemar（精确二项）看不一致对，而不是把 64 条当独立样本做 Fisher。
- **两种方法都报**：pooled Fisher（读者熟悉） + McNemar（统计上更正确），不一致时以 McNemar 为准。
- **Wilson 置信区间**：给通过率一个区间，避免只看点估计。
- **逐 seed 一致性**：有多少个 seed 里 A 严格优于 C。全部一致才说明不是某个 seed 的偶然。

用法：
    python scripts/summarize_eval.py reports/eval_openai_multiseed.json
    python scripts/summarize_eval.py reports/eval_openai_multiseed.json --out docs/eval-multiseed.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest, fisher_exact
except ImportError:  # pragma: no cover
    fisher_exact = None
    binomtest = None

GROUP_LABEL = {
    "A_full": "A 全套",
    "B_no_warning": "B 关工具警示",
    "C_no_tool": "C 关置换工具",
    "D_loose_prompt": "D 关提示词要求",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间，小样本下比 normal approximation 稳。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def key_of(run: dict) -> str:
    """配对键：task id + seed 后缀。"""
    return run["task"]["id"]


def strip_seed(tid: str) -> str:
    return tid.split("@")[0]


def seed_of(tid: str) -> str:
    return tid.split("@s")[1] if "@s" in tid else "0"


def mcnemar(pairs_a: dict, pairs_b: dict) -> dict:
    """配对 McNemar：只看不一致对 (A过C挂 / A挂C过)，精确二项检验。"""
    b_only = sum(1 for k, v in pairs_a.items() if v and k in pairs_b and not pairs_b[k])
    c_only = sum(1 for k, v in pairs_b.items() if v and k in pairs_a and not pairs_a[k])
    n_disc = b_only + c_only
    if binomtest is None or n_disc == 0:
        return {"a_only": b_only, "other_only": c_only, "discordant": n_disc, "p": None}
    p = binomtest(b_only, n_disc, 0.5).pvalue
    return {"a_only": b_only, "other_only": c_only, "discordant": n_disc, "p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = json.loads(Path(args.report).read_text(encoding="utf-8"))
    runs = d["runs"]
    offline = d.get("offline", True)

    # ---------------------------------------------------------------- 分组
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    order: list[str] = []
    for r in runs:
        g = r.get("group", "")
        # 旧版报告没有 group 字段，用 '(task,seed)' 推断会失败，这里容错
        if g not in GROUP_LABEL:
            continue
        if g not in order:
            order.append(g)
        by_group[g][key_of(r)] = r["grade"]

    if offline:
        print("⚠️  这是离线后端的报告，数字不得引用。")

    lines: list[str] = []
    lines.append(f"# 多 seed 实测汇总（{d.get('model') or d.get('agent')}）\n")
    lines.append(
        f"- 运行组数 {len(order)}；seeds {d.get('seeds')}；每组的样本条数 n "
        f"= {max((len(v) for v in by_group.values()), default=0)}"
    )
    lines.append(f"- 报告来源 `{Path(args.report).name}`\n")

    # ---------------------------------------------------------------- 主表
    lines.append("## 一、各组通过率\n")
    lines.append("| 组 | 通过 | Wilson 95% CI | 调用置换检验 | 调用批次诊断 |")
    lines.append("|---|---|---|---|---|")
    summary = {}
    for g in order:
        grades = by_group[g]
        n = len(grades)
        ok = sum(gr["correct"] for gr in grades.values())
        lo, hi = wilson_ci(ok, n)
        perm = sum(gr["called_permutation"] for gr in grades.values())
        batch = sum(gr["used_batch_check"] for gr in grades.values())
        summary[g] = {"n": n, "correct": ok}
        lines.append(
            f"| {GROUP_LABEL[g]} | **{ok}/{n}** ({ok/n:.0%}) | "
            f"{lo:.0%}–{hi:.0%} | {perm}/{n} | {batch}/{n} |"
        )

    # ---------------------------------------------------------------- 显著性
    lines.append("\n## 二、A 组 vs 其余（两种算法）\n")
    lines.append("| 对比 | pooled Fisher p | McNemar p | A 独有的成功 | 对方独有的成功 |")
    lines.append("|---|---|---|---|---|")
    comp = {}
    if "A_full" in summary and fisher_exact is not None:
        ga = by_group["A_full"]
        for g in order:
            if g == "A_full":
                continue
            gb = by_group[g]
            tbl = [
                [summary["A_full"]["correct"], summary["A_full"]["n"] - summary["A_full"]["correct"]],
                [summary[g]["correct"], summary[g]["n"] - summary[g]["correct"]],
            ]
            _, pf = fisher_exact(tbl)
            mc = mcnemar(
                {k: v["correct"] for k, v in ga.items()},
                {k: v["correct"] for k, v in gb.items()},
            )
            comp[g] = {"fisher_p": float(pf), **mc}
            pp = "—" if mc["p"] is None else f"{mc['p']:.4f}"
            lines.append(
                f"| A vs {GROUP_LABEL[g]} | {pf:.4f} | **{pp}** | "
                f"{mc['a_only']} | {mc['other_only']} |"
            )

    # ---------------------------------------------------------------- 逐任务
    lines.append("\n## 三、逐任务通过情况（分母 = seed 数）\n")
    tasks = sorted({strip_seed(tid) for g in order for tid in by_group[g]})
    lines.append("| 任务 | " + " | ".join(GROUP_LABEL[g] for g in order) + " |")
    lines.append("|---" * (len(order) + 1) + "|")
    for t in tasks:
        row = [f"`{t}`"]
        for g in order:
            sub = {k: v for k, v in by_group[g].items() if strip_seed(k) == t}
            if not sub:
                row.append("—")
                continue
            ok = sum(v["correct"] for v in sub.values())
            row.append(f"{ok}/{len(sub)}")
        lines.append("| " + " | ".join(row) + " |")

    # ---------------------------------------------------------------- 失败模式
    lines.append("\n## 四、失败模式分布\n")
    for g in order:
        modes: dict[str, int] = defaultdict(int)
        for gr in by_group[g].values():
            if gr.get("failure_mode"):
                modes[gr["failure_mode"]] += 1
        if modes:
            bullets = "；".join(f"{m} ×{c}" for m, c in sorted(modes.items(), key=lambda x: -x[1]))
            lines.append(f"- **{GROUP_LABEL[g]}**：{bullets}")
        else:
            lines.append(f"- **{GROUP_LABEL[g]}**：无失败")

    # ---------------------------------------------------------------- 老实交代
    lines.append("\n## 五、这份数字能用到什么程度\n")
    n_per_group = max((len(v) for v in by_group.values()), default=0)
    lines.append(
        f"- 每组 n={n_per_group} 条，来自 {len(tasks)} 类任务 × {len({seed_of(tid) for g in order for tid in by_group[g]})} 个 seed 的合成数据集；"
    )
    lines.append("- **单模型**（多模型未做），所以结论限于该模型；")
    lines.append("- 任务之间是不同类型的扰动，runs 并非完全独立样本，pooled Fisher 会略微高估自由度；")
    lines.append("- 配对 McNemar 是主要依据，pooled Fisher 仅供对照。")

    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
