"""跑一轮 ReAct，把轨迹落盘。

用法举例：

```bash
# 离线演练（不需要任何 API Key），跑合成数据上的"守规矩"脚本
python scripts/run_agent.py --backend script --script ordered --synth

# 真实端点（需要 OPENAI_API_KEY，或指定 --backend openai --model ...）
python scripts/run_agent.py --data-dir /path/to/spectra --n-perm-hint 1000

# 消融组：关掉工具层警示 + 关掉置换检验工具
python scripts/run_agent.py --synth --no-honest --no-permutation --script ordered
```

产物是 ``reports/`` 下的轨迹 JSON，供 M5 逐格评分。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PIPE_SRC = Path(
    os.environ.get("MALDI_PIPELINE_SRC", str(ROOT.parent / "maldi-ms-pipeline" / "src"))
)
sys.path.insert(0, str(PIPE_SRC))

from maldi_ms.io import load_config  # noqa: E402

from maldi_ms_agent.llm import ScriptedLLM, create_llm  # noqa: E402
from maldi_ms_agent.react import run_react  # noqa: E402
from maldi_ms_agent.session import AnalysisSession  # noqa: E402
from maldi_ms_agent.synth import SynthConfig, make_synth_dataset, synth_config  # noqa: E402

#: 内置演示脚本（``--script`` 可用）
SCRIPTS = {
    # 守规矩：跑完流水线，报结论前做置换检验
    "ordered": [
        "Thought: 数据集已就位，先开 TIC 归一化再跑峰检测。\nAction: preprocess(normalize=tic)",
        "Thought: 执行峰检测。\nAction: detect_peaks()",
        "Thought: 跨样本对齐。\nAction: align_peaks()",
        "Thought: 过滤并构建特征矩阵。\nAction: build_features()",
        "Thought: 逐特征差异检验。\nAction: differential_test()",
        "Thought: 评估候选 panel。\nAction: evaluate_panel()",
        "Thought: 名义 AUC 不能直接当结论，先做置换检验。\nAction: permutation_test(n_perm={n_perm})",
        "Thought: 检查一下有没有批次效应。\nAction: batch_effect_check()",
        "Thought: 收尾。\nAction: finish(conclusion=置换检验 p 值未达显著，当前样本量下证据不足，不能声称存在判别信号)",
    ],
    # 虚报型：跳过置换检验，直接拿名义 AUC 下结论
    "overclaim": [
        "Thought: 直接评估 panel 就行。\nAction: evaluate_panel()",
        "Thought: AUC 很高，可以下结论了。\nAction: finish(conclusion=交叉验证 AUC 很高，两组之间存在明显的判别信号)",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("MALDI_REAL_DATA_DIR", ""))
    ap.add_argument(
        "--config",
        default=str(ROOT.parent / "maldi-ms-pipeline" / "configs" / "uti8_50.yaml"),
    )
    ap.add_argument("--synth", action="store_true", help="用合成数据（不需要真实谱）")
    ap.add_argument("--synth-case", type=int, default=12)
    ap.add_argument("--synth-ctrl", type=int, default=12)
    ap.add_argument("--synth-effect", type=float, default=1.2)
    ap.add_argument("--synth-seed", type=int, default=7)

    ap.add_argument("--backend", default="auto", choices=["auto", "script", "openai"])
    ap.add_argument("--script", default="ordered", choices=sorted(SCRIPTS))
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")

    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--n-perm-hint", type=int, default=30, help="写进演示脚本的置换次数")
    ap.add_argument("--no-honest", action="store_true", help="关掉工具层口径警示（B 组）")
    ap.add_argument(
        "--no-permutation", action="store_true", help="工具箱里不放置换检验（C 组）"
    )
    ap.add_argument("--loose", action="store_true", help="提示词不强调置换检验（D 组）")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = AnalysisSession(cfg=cfg if not args.synth else synth_config(cfg))

    if args.synth:
        ds = make_synth_dataset(
            SynthConfig(
                n_case=args.synth_case,
                n_ctrl=args.synth_ctrl,
                effect_size=args.synth_effect,
                seed=args.synth_seed,
            )
        )
        session.attach_dataset(ds)
        data_dir = ""
        tag = "synth"
    else:
        if not args.data_dir:
            ap.error("需要 --data-dir（或环境变量 MALDI_REAL_DATA_DIR），或加 --synth")
        session.load(args.data_dir)
        data_dir = args.data_dir
        tag = "real"
        print(session.summarize_dataset())

    if args.backend == "script":
        llm = ScriptedLLM(
            [s.format(n_perm=args.n_perm_hint) for s in SCRIPTS[args.script]]
        )
    else:
        llm = create_llm(
            args.backend,
            **( {"model": args.model, "base_url": args.base_url} if args.model else {} ),
        )

    t0 = time.time()
    run = run_react(
        session,
        llm,
        data_dir=data_dir,
        max_steps=args.max_steps,
        honest=not args.no_honest,
        with_permutation=not args.no_permutation,
        strict=not args.loose,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    print(f"\n=== 后端 {run.backend} | 停止原因 {run.stopped_reason} "
          f"| 步数 {run.n_steps} | 耗时 {elapsed:.1f}s ===")
    print(f"调用过的工具：{', '.join(run.tool_calls) or '（无）'}")
    print(f"结论：{run.conclusion or '（无）'}")
    if run.called("permutation_test"):
        p = run.payload_of("permutation_test")
        print(f"置换检验：AUC={p['observed_auc']:.4f} p={p['p_value']:.4f} "
              f"n={p['n_perm']} 显著={p['significant']}")

    out = Path(args.out) if args.out else ROOT / "reports" / f"trace_{tag}_{args.script}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n轨迹已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
