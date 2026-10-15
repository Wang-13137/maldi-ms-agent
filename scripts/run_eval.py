"""跑完整评测：4 组消融 × 任务集 × 多 seed（M5）。

```bash
# 离线自检：规则型 Agent，验证评测框架本身有区分力
python scripts/run_eval.py --agent rule

# 接了 Key 之后：真实模型
python scripts/run_eval.py --agent openai --model glm-4-flash --seeds 0,1,2
```

四组消融：

| 组 | strict | honest | with_permutation |
|---|---|---|---|
| A | ✅ | ✅ | ✅ |
| B | ✅ | ❌ | ✅ |
| C | ✅ | ✅ | ❌ |
| D | ❌ | ✅ | ✅ |

**离线（rule / script）跑出来的数字一律不得写进简历或 README 结论**——
规则 Agent 不代表真实模型的水平，它只用来证明"这套评测能区分好坏"。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os as _os

PIPE_SRC = Path(
    _os.environ.get("MALDI_PIPELINE_SRC", str(ROOT.parent / "maldi-ms-pipeline" / "src"))
)
sys.path.insert(0, str(PIPE_SRC))

from maldi_ms.io import load_config  # noqa: E402

from maldi_ms_agent.llm import ScriptedLLM, create_llm  # noqa: E402
from maldi_ms_agent.rule_agent import RuleBasedLLM  # noqa: E402
from maldi_ms_agent.tasks import TASKS, Task, expand_tasks, run_task  # noqa: E402

ABLATIONS = {
    "A_full": dict(honest=True, with_permutation=True, strict=True),
    "B_no_warning": dict(honest=False, with_permutation=True, strict=True),
    "C_no_tool": dict(honest=True, with_permutation=False, strict=True),
    "D_loose_prompt": dict(honest=True, with_permutation=True, strict=False),
}

#: 跟 run_agent.py 里同款脚本（--agent script 时用）
SCRIPT_HONEST = [
    "Thought: 开 TIC 归一化。\nAction: preprocess(normalize=tic)",
    "Thought: 峰检测。\nAction: detect_peaks()",
    "Thought: 对齐。\nAction: align_peaks()",
    "Thought: 特征矩阵。\nAction: build_features()",
    "Thought: 差异检验。\nAction: differential_test()",
    "Thought: 评估 panel。\nAction: evaluate_panel()",
    "Thought: 做置换检验。\nAction: permutation_test(n_perm={n_perm})",
    "Thought: 查批次。\nAction: batch_effect_check()",
    "Thought: 收尾。\nAction: finish(conclusion=置换检验 p={placeholder}，证据不足)",
]


def make_llm(agent: str, n_perm: int, model: str, base_url: str):
    if agent == "rule":
        return RuleBasedLLM(n_perm=n_perm)
    if agent == "script":
        return ScriptedLLM([s.format(n_perm=n_perm, placeholder="?") for s in SCRIPT_HONEST])
    return create_llm("openai", **({"model": model, "base_url": base_url} if model else {}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="rule", choices=["rule", "script", "openai"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--seeds", default="0", help="逗号分隔，如 0,1,2")
    ap.add_argument("--groups", default=",".join(ABLATIONS))
    ap.add_argument("--tasks", default="", help="只跑指定任务 id 前缀，逗号分隔")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument(
        "--config",
        default=str(ROOT.parent / "maldi-ms-pipeline" / "configs" / "uti8_50.yaml"),
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    tasks: list[Task] = TASKS
    if args.tasks:
        prefixes = [t.strip() for t in args.tasks.split(",")]
        tasks = [t for t in TASKS if any(t.id.startswith(p) for p in prefixes)]

    cfg = load_config(args.config)
    t0 = time.time()

    results: dict[str, list] = defaultdict(list)
    for gname in groups:
        flags = ABLATIONS[gname]
        for t in expand_tasks(tasks, seeds=seeds):
            llm = make_llm(args.agent, args.n_perm, args.model, args.base_url)
            tr = run_task(
                t, cfg, llm,
                max_steps=args.max_steps,
                honest=flags["honest"],
                with_permutation=flags["with_permutation"],
                strict=flags["strict"],
                verbose=args.verbose,
            )
            results[gname].append(tr)
            print(
                f"[{gname:14s}] {t.id:16s} correct={str(tr.grade.correct):5s} "
                f"perm={str(tr.grade.called_permutation):5s} "
                f"batch={str(tr.grade.used_batch_check):5s} "
                f"{tr.grade.failure_mode}"
            )

    # ---------------------------------------------------------------- 汇总
    print("\n=== 汇总 ===")
    summary = {}
    for gname in groups:
        gs = [tr.grade for tr in results[gname]]
        n = len(gs)
        ok = sum(g.correct for g in gs)
        modes = Counter(g.failure_mode for g in gs if g.failure_mode)
        summary[gname] = {
            "n": n,
            "correct": ok,
            "rate": ok / n if n else 0.0,
            "failure_modes": dict(modes),
            "called_permutation": sum(g.called_permutation for g in gs),
            "used_batch_check": sum(g.used_batch_check for g in gs),
        }
        print(
            f"{gname:14s} 通过 {ok}/{n} ({summary[gname]['rate']:.0%})  "
            f"做置换检验 {summary[gname]['called_permutation']}/{n}  "
            f"做批次诊断 {summary[gname]['used_batch_check']}/{n}"
        )
        for m, c in modes.most_common():
            print(f"                 └ {m} ×{c}")

    # ---------------------------------------------------------------- 显著性
    comparisons = {}
    if "A_full" in summary:
        try:
            from scipy.stats import fisher_exact

            a = summary["A_full"]
            for gname in groups:
                if gname == "A_full":
                    continue
                b = summary[gname]
                tbl = [
                    [a["correct"], a["n"] - a["correct"]],
                    [b["correct"], b["n"] - b["correct"]],
                ]
                _, p = fisher_exact(tbl)
                comparisons[f"A_full vs {gname}"] = {
                    "table": tbl, "fisher_p": float(p),
                }
                print(
                    f"\nA vs {gname}: {a['correct']}/{a['n']} vs {b['correct']}/{b['n']} "
                    f"Fisher p={p:.4f}"
                    + ("（样本量太小，只能看趋势）" if a["n"] + b["n"] < 20 else "")
                )
        except ImportError:  # pragma: no cover
            pass

    out = Path(args.out) if args.out else ROOT / "reports" / f"eval_{args.agent}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "agent": args.agent,
                "model": args.model,
                "n_perm": args.n_perm,
                "seeds": seeds,
                "offline": args.agent != "openai",
                "warning": (
                    "离线后端（rule/script）跑出的数字只用于验证评测框架的区分力，"
                    "不代表真实模型水平，不得引用。"
                    if args.agent != "openai"
                    else ""
                ),
                "summary": summary,
                "comparisons": comparisons,
                "runs": [tr.to_dict() for g in groups for tr in results[g]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n总耗时 {time.time()-t0:.0f}s；结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
